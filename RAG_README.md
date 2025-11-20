# RAG Knowledge DB and Retrieval Tool

This document explains how to use the SQLite-based RAG knowledge database manager and the retrieval tool.

## Overview

- Stores document chunks with embeddings in `knowledge_chunks` and maintains an `fts5` table `knowledge_fts` for lexical search.
- Retrieval uses a hybrid strategy:
  - BM25 computed over loaded candidate contents
  - ANN similarity via SimHash-style random hyperplanes on embeddings
  - MMR reranker balances relevance and diversity
- Embeddings provider is pluggable via environment variables (OpenAI, Azure OpenAI, Ollama, HuggingFace).

## Requirements

- Python 3.10+
- SQLite with FTS5 (bundled with modern Python/Windows builds)
- Install dependencies:
  
  ```powershell
  pip install -r requirements.txt
  ```

## Environment Configuration

Set your embedding provider and optional tuning parameters.

- Provider selection
  - `EMBEDDINGS_PROVIDER`: `openai` | `azure` | `ollama` | `huggingface`
- OpenAI
  - `OPENAI_API_KEY`
  - `OPENAI_EMBEDDINGS_MODEL` (optional)
- Azure OpenAI
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT`
- Ollama
  - `OLLAMA_BASE_URL` (default `http://localhost:11434`)
  - `OLLAMA_EMBED_MODEL` (e.g., `nomic-embed-text`)
- HuggingFace
  - `HF_EMBEDDINGS_MODEL` (e.g., `sentence-transformers/all-MiniLM-L6-v2`)
- Retrieval tuning
  - `RAG_MAX_CANDIDATES` (default `1000`)
  - `RAG_ALPHA` (ANN weight, default `0.5`)
  - `RAG_BETA` (BM25 weight, default `0.5`)
  - `RAG_POOL_K` (pre-rerank pool size, default `20`)
  - `RAG_MMR_LAMBDA` (diversity tradeoff, default `0.5`)

Example (PowerShell):

```powershell
$env:EMBEDDINGS_PROVIDER = "openai"
$env:OPENAI_API_KEY = "sk-..."
$env:RAG_MAX_CANDIDATES = "1000"
$env:RAG_ALPHA = "0.6"
$env:RAG_BETA = "0.4"
$env:RAG_POOL_K = "30"
$env:RAG_MMR_LAMBDA = "0.4"
```

## Initialize the Database

```powershell
python tools/rag_db_manager.py init --db "E:\Knowledge\knowledge.db"
```

## Ingest Content

- Ingest raw text as a document

```powershell
python tools/rag_db_manager.py ingest_text --doc_id "doc1" --text "Python is a programming language used for many tasks." --db "E:\Knowledge\knowledge.db"
```

- Ingest a folder of files (PDFs, text, etc.)

```powershell
python tools/rag_db_manager.py ingest_path --doc_id "docs" --path "E:\Docs" --db "E:\Knowledge\knowledge.db"
```

- Delete content by document id

```powershell
python tools/rag_db_manager.py delete_doc --doc_id "doc1" --db "E:\Knowledge\knowledge.db"
```

- Delete content by source

```powershell
python tools/rag_db_manager.py delete_source --source "E:\Docs" --db "E:\Knowledge\knowledge.db"
```

## Retrieve Contexts (Python)

```python
import json
from tools.rag_retriever import KnowledgeSearchTool

tool = KnowledgeSearchTool()
result = tool._run(query="What is Python?", top_k=5, db_path="E:\\Knowledge\\knowledge.db")
payload = json.loads(result)
contexts = payload.get("contexts", [])
for c in contexts:
    print(c["id"], c["score"])
    print(c["content"][:200])
```

- Output fields per context:
  - `id`: chunk id (`doc_id:index`)
  - `content`: chunk text
  - `score`: hybrid score after rerank
  - `source`: original file/folder source (if available)
  - `metadata`: any loader metadata

## How Retrieval Works

- Candidate loading
  - Loads recent chunks from `knowledge_chunks` (`ORDER BY created_at DESC LIMIT RAG_MAX_CANDIDATES`)
- BM25
  - Computes DF/TF over loaded contents
- ANN
  - Uses random-hyperplane SimHash across embedding vectors to approximate similarity
- Hybrid score
  - `score = RAG_ALPHA * ann_norm + RAG_BETA * bm25_norm`
- Rerank
  - Applies MMR with `RAG_MMR_LAMBDA` to select a diverse and relevant final set

## Code References
- DB manager actions: `tools/rag_db_manager.py:219-253`
- Retrieval tool entry: `tools/rag_retriever.py:163-214`
- BM25 scoring helper: `tools/rag_retriever.py:78-110`
- ANN SimHash helper: `tools/rag_retriever.py:113-127`
- MMR reranker: `tools/rag_retriever.py:130-154`

## Troubleshooting

- Empty results
  - Ensure the DB path is correct and content was ingested
  - Increase `RAG_MAX_CANDIDATES` if the DB is large
- Embeddings errors
  - Verify provider env variables are set
- Performance
  - Tune `RAG_MAX_CANDIDATES`, `RAG_POOL_K`, and provider embedding model for speed/quality

## Tests

Run the basic tests (LangChain required for ingestion):

```powershell
python -m unittest discover -s tests -v
```