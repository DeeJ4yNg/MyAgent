import os
import sqlite3
import json
import time
from typing import List, Optional, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document

try:
    from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None
    AzureOpenAIEmbeddings = None

try:
    from langchain_ollama import OllamaEmbeddings
except Exception:
    OllamaEmbeddings = None

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

DB_FILENAME = os.path.join(os.getcwd(), "knowledge.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT,
    chunk_index INTEGER,
    content TEXT,
    embedding_json TEXT,
    source TEXT,
    metadata_json TEXT,
    created_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    content,
    doc_id,
    chunk_id,
    tokenize = 'porter'
);
CREATE INDEX IF NOT EXISTS idx_knowledge_doc ON knowledge_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunk ON knowledge_chunks(doc_id, chunk_index);
"""


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_FILENAME
    conn = sqlite3.connect(path)
    return conn


def init_db(db_path: Optional[str] = None) -> None:
    conn = _connect(db_path)
    cur = conn.cursor()
    for stmt in SCHEMA_SQL.strip().split(";"):
        s = stmt.strip()
        if s:
            cur.execute(s)
    conn.commit()
    conn.close()


def _embeddings_provider():
    provider = (os.getenv("EMBEDDINGS_PROVIDER", "").lower()).strip()
    if provider == "openai" and OpenAIEmbeddings:
        model = os.getenv("OPENAI_EMBEDDINGS_MODEL", os.getenv("EMBEDDINGS_MODEL", "")) or None
        return OpenAIEmbeddings(model=model) if model else OpenAIEmbeddings()
    if provider == "azure" and AzureOpenAIEmbeddings:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT", os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small"))
        return AzureOpenAIEmbeddings(azure_endpoint=endpoint, api_key=key, model=deployment)
    if provider == "ollama" and OllamaEmbeddings:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_EMBED_MODEL", os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text"))
        return OllamaEmbeddings(base_url=base_url, model=model)
    if provider == "huggingface" and HuggingFaceEmbeddings:
        model_name = os.getenv("HF_EMBEDDINGS_MODEL", os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
        return HuggingFaceEmbeddings(model_name=model_name)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and OpenAIEmbeddings:
        model = os.getenv("OPENAI_EMBEDDINGS_MODEL")
        return OpenAIEmbeddings(model=model) if model else OpenAIEmbeddings()
    if OllamaEmbeddings:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(base_url=base_url, model=model)
    if HuggingFaceEmbeddings:
        model_name = os.getenv("HF_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)
    raise RuntimeError("No embeddings provider available. Configure EMBEDDINGS_PROVIDER or set OPENAI_API_KEY/OLLAMA/HF.")


def _load_documents_from_path(path: str) -> List[Document]:
    docs: List[Document] = []
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                docs.extend(_load_single_file(fp))
        return docs
    return _load_single_file(path)


def _load_single_file(file_path: str) -> List[Document]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    if ext in [".pdf"]:
        loader = PyPDFLoader(file_path)
        return loader.load()
    loader = UnstructuredFileLoader(file_path)
    return loader.load()


def _split_documents(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def _compute_embeddings(texts: List[str]) -> List[List[float]]:
    emb = _embeddings_provider()
    vectors = emb.embed_documents(texts)
    return vectors


def _upsert_chunks(conn: sqlite3.Connection, doc_id: str, chunks: List[Document], vectors: List[List[float]], source: str) -> None:
    cur = conn.cursor()
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
        cid = f"{doc_id}:{idx}"
        cur.execute(
            """
            INSERT OR REPLACE INTO knowledge_chunks(id, doc_id, chunk_index, content, embedding_json, source, metadata_json, created_at)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                cid,
                doc_id,
                idx,
                chunk.page_content,
                json.dumps(vec),
                source,
                json.dumps(chunk.metadata or {}),
                now,
            ),
        )
        cur.execute(
            "INSERT INTO knowledge_fts(content, doc_id, chunk_id) VALUES(?,?,?)",
            (chunk.page_content, doc_id, cid),
        )
    conn.commit()


def ingest_path(path: str, doc_id: Optional[str] = None, db_path: Optional[str] = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    init_db(db_path)
    docs = _load_documents_from_path(path)
    if not docs:
        return {"status": "empty", "path": path}
    chunks = _split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [c.page_content for c in chunks]
    vectors = _compute_embeddings(texts)
    conn = _connect(db_path)
    did = doc_id or os.path.basename(path)
    _upsert_chunks(conn, did, chunks, vectors, source=path)
    conn.close()
    return {"status": "ok", "doc_id": did, "chunks": len(chunks)}


def ingest_text(text: str, doc_id: str, db_path: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, chunk_size: int = 1000, chunk_overlap: int = 200) -> Dict[str, Any]:
    init_db(db_path)
    docs = [Document(page_content=text, metadata=metadata or {})]
    chunks = _split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = [c.page_content for c in chunks]
    vectors = _compute_embeddings(texts)
    conn = _connect(db_path)
    _upsert_chunks(conn, doc_id, chunks, vectors, source="inline")
    conn.close()
    return {"status": "ok", "doc_id": doc_id, "chunks": len(chunks)}


def delete_by_doc_id(doc_id: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    init_db(db_path)
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM knowledge_fts WHERE doc_id=?", (doc_id,))
    cur.execute("DELETE FROM knowledge_chunks WHERE doc_id=?", (doc_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "doc_id": doc_id}


def delete_by_source(source: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    init_db(db_path)
    conn = _connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, doc_id FROM knowledge_chunks WHERE source=?", (source,))
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    doc_ids = list(set([r[1] for r in rows]))
    for d in doc_ids:
        cur.execute("DELETE FROM knowledge_fts WHERE doc_id=?", (d,))
    cur.execute("DELETE FROM knowledge_chunks WHERE source=?", (source,))
    conn.commit()
    conn.close()
    return {"status": "ok", "deleted_chunk_ids": ids}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init", "ingest_path", "ingest_text", "delete_doc", "delete_source"])
    parser.add_argument("--path", dest="path")
    parser.add_argument("--text", dest="text")
    parser.add_argument("--doc_id", dest="doc_id")
    parser.add_argument("--source", dest="source")
    parser.add_argument("--db", dest="db")
    args = parser.parse_args()
    if args.action == "init":
        init_db(args.db)
        print("ok")
    elif args.action == "ingest_path":
        if not args.path:
            print("error: path required")
        else:
            print(json.dumps(ingest_path(args.path, args.doc_id, args.db)))
    elif args.action == "ingest_text":
        if not args.text or not args.doc_id:
            print("error: text and doc_id required")
        else:
            print(json.dumps(ingest_text(args.text, args.doc_id, args.db)))
    elif args.action == "delete_doc":
        if not args.doc_id:
            print("error: doc_id required")
        else:
            print(json.dumps(delete_by_doc_id(args.doc_id, args.db)))
    elif args.action == "delete_source":
        if not args.source:
            print("error: source required")
        else:
            print(json.dumps(delete_by_source(args.source, args.db)))