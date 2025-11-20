import os
import sqlite3
import json
import math
import re
import random
from typing import Optional, List, Dict, Any, Tuple

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


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def _bm25_scores(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return [0.0] * len(docs)
    N = max(1, len(docs))
    tokenized_docs = [ _tokenize(d) for d in docs ]
    vocab_in_docs: List[set] = [ set(td) for td in tokenized_docs ]
    df: Dict[str, int] = { t: sum(1 for vd in vocab_in_docs if t in vd) for t in set(q_tokens) }
    lengths = [ len(td) for td in tokenized_docs ]
    avgdl = (sum(lengths) / max(1, len(lengths))) if lengths else 1.0
    scores: List[float] = []
    for td in tokenized_docs:
        dl = len(td)
        s = 0.0
        tf: Dict[str, int] = {}
        for w in td:
            tf[w] = tf.get(w, 0) + 1
        for t in q_tokens:
            tf_t = tf.get(t, 0)
            if tf_t == 0:
                continue
            df_t = df.get(t, 0)
            idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1.0)
            denom = tf_t + k1 * (1 - b + b * (dl / max(1.0, avgdl)))
            s += idf * ((tf_t * (k1 + 1)) / denom)
        scores.append(s)
    return scores


def _ann_simhash_scores(qvec: List[float], doc_vecs: List[List[float]], num_planes: int = 16) -> List[float]:
    dim = len(qvec)
    if dim == 0 or not doc_vecs:
        return [0.0] * len(doc_vecs)
    rng = random.Random(42)
    planes: List[List[float]] = [ [rng.gauss(0, 1) for _ in range(dim)] for _ in range(num_planes) ]
    def _sign(v: List[float], p: List[float]) -> int:
        return 1 if sum(x*y for x, y in zip(v, p)) >= 0 else 0
    q_bits = [ _sign(qvec, p) for p in planes ]
    scores: List[float] = []
    for dv in doc_vecs:
        bits = [ _sign(dv, p) for p in planes ]
        match = sum(1 for a, b in zip(q_bits, bits) if a == b)
        scores.append(match / float(num_planes))
    return scores


def _mmr_rerank(qvec: List[float], items: List[Tuple[str, str, List[float], Dict[str, Any], float]], top_k: int, lam: float = 0.5) -> List[Dict[str, Any]]:
    selected: List[int] = []
    remaining = list(range(len(items)))
    sims = [ _cosine(qvec, it[2]) for it in items ]
    while len(selected) < min(top_k, len(items)) and remaining:
        best_idx = None
        best_score = -1e9
        for i in remaining:
            relevance = sims[i]
            redundancy = 0.0
            if selected:
                redundancy = max(_cosine(items[i][2], items[j][2]) for j in selected)
            score = (1 - lam) * relevance - lam * redundancy
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    result: List[Dict[str, Any]] = []
    for i in selected:
        cid, content, vec, meta, combined = items[i]
        m = {"id": cid, "content": content, "score": combined}
        m.update(meta or {})
        result.append(m)
    return result


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
            max_candidates = int(os.getenv("RAG_MAX_CANDIDATES", "1000"))
            cur.execute(
                "SELECT id, content, embedding_json, source, metadata_json FROM knowledge_chunks ORDER BY created_at DESC LIMIT ?",
                (max_candidates,)
            )
            chunks = cur.fetchall()
            if not chunks:
                return json.dumps({"contexts": [], "note": "empty_knowledge"})
            emb = _embeddings_provider()
            qvec = emb.embed_query(query)
            contents = [c[1] for c in chunks]
            bm25_list = _bm25_scores(query, contents)
            vecs: List[List[float]] = [ json.loads(c[2]) for c in chunks ]
            ann_list = _ann_simhash_scores(qvec, vecs)
            max_bm25 = max(bm25_list) if bm25_list else 1.0
            max_ann = max(ann_list) if ann_list else 1.0
            alpha = float(os.getenv("RAG_ALPHA", "0.5"))
            beta = float(os.getenv("RAG_BETA", "0.5"))
            combined_items: List[Tuple[str, str, List[float], Dict[str, Any], float]] = []
            for i, (cid, content, ejson, source, mjson) in enumerate(chunks):
                bm = bm25_list[i] / max(1e-9, max_bm25)
                an = ann_list[i] / max(1e-9, max_ann)
                combined = alpha * an + beta * bm
                meta = {"source": source, "metadata": json.loads(mjson or "{}")}
                combined_items.append((cid, content, json.loads(ejson), meta, combined))
            combined_items.sort(key=lambda x: x[4], reverse=True)
            pool_k = int(os.getenv("RAG_POOL_K", "20"))
            lam = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))
            reranked = _mmr_rerank(qvec, combined_items[:max(pool_k, top_k)], top_k, lam)
            return json.dumps({"contexts": reranked})
        except Exception as e:
            return json.dumps({"error": str(e)})
        finally:
            try:
                conn.close()
            except Exception:
                pass