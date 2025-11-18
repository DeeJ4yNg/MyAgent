import os
import sqlite3
import json
import math
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field
from langchain.tools import BaseTool

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


def _connect(db_path: Optional[str]) -> sqlite3.Connection:
    path = db_path or os.path.join(os.getcwd(), "knowledge.db")
    return sqlite3.connect(path)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class KnowledgeSearchInput(BaseModel):
    query: str = Field(..., description="Question or search query")
    top_k: int = Field(5, description="Number of contexts to return")
    db_path: Optional[str] = Field(None, description="SQLite database path")


class KnowledgeSearchTool(BaseTool):
    name: str = "knowledge_search"
    description: str = "Search internal knowledge using SQLite + embeddings and return top contexts"
    args_schema: type = KnowledgeSearchInput

    def _run(self, query: str, top_k: int = 5, db_path: Optional[str] = None) -> str:
        conn = _connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT chunk_id, content FROM knowledge_fts WHERE knowledge_fts MATCH ? LIMIT 200",
                (query,)
            )
            rows = cur.fetchall()
            if not rows:
                return json.dumps({"contexts": [], "note": "no lexical matches"})
            chunk_ids = [r[0] for r in rows]
            placeholders = ",".join(["?"] * len(chunk_ids))
            cur.execute(
                f"SELECT id, content, embedding_json, source, metadata_json FROM knowledge_chunks WHERE id IN ({placeholders})",
                chunk_ids,
            )
            chunks = cur.fetchall()
            emb = _embeddings_provider()
            qvec = emb.embed_query(query)
            scored: List[Dict[str, Any]] = []
            for cid, content, ejson, source, mjson in chunks:
                vec = json.loads(ejson)
                score = _cosine(qvec, vec)
                scored.append({"id": cid, "content": content, "source": source, "metadata": json.loads(mjson or "{}"), "score": score})
            scored.sort(key=lambda x: x["score"], reverse=True)
            top = scored[:max(1, top_k)]
            return json.dumps({"contexts": top})
        except Exception as e:
            return json.dumps({"error": str(e)})
        finally:
            try:
                conn.close()
            except Exception:
                pass