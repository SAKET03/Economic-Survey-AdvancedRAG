# 📘 Economic Survey RAG + Knowledge Graph Pipeline

This repository implements an end-to-end Retrieval-Augmented Generation (RAG) pipeline and Knowledge Graph (KG) construction for India's Economic Survey. The system ingests raw survey documents, processes them into structured chunks, builds a Neo4j-powered knowledge graph with embeddings, and enables intelligent Q\&A through hybrid retrieval and reranking.

---

## 🚀 Pipeline Overview

### Ingestion (`01_ingestion.ipynb`)

* Load Economic Survey PDFs.
* Extract raw text.
* Store in a structured format.

### Chunking (`02_pdf_chunking.ipynb`)

* Split text into manageable chunks (chapter/subchapter level).
* Annotate with metadata (chapter\_no, subchapter\_no, tags).

### LLM Processing (`03_llm_processing.ipynb`)

* Use LLMs to enrich chunks (summaries, tagging, embedding preparation).
* Save processed chunks as JSON.

### JSON → Knowledge Graph (`04_json_to_kg.py`)

* Build a Neo4j knowledge graph from chunk JSONs.
* **Nodes**: Chapter, Chunk, Concept
* **Relationships**:

  * (\:Chapter)-\[:CONTAINS]->(\:Chunk)
  * (\:Chunk)-\[:RELATES\_TO]->(\:Concept)
  * (\:Concept)-\[:CO\_OCCURS\_WITH]->(\:Concept)

### Vector Indexing (`05_kg_vector.ipynb`)

* Generate embeddings using HuggingFace (`BAAI/bge-base-en-v1.5`).
* Store vectors in Neo4j using `langchain-neo4j`.
* Hybrid search enabled (full-text + embedding).

### RAG Q\&A (`06_rag_answers.py`)

* Retrieve relevant chunks (top-25 candidates).
* Re-rank with CrossEncoder (`BAAI/bge-reranker-base`).
* Query ChatGroq LLM (`LLaMA 4 Scout 17B`) with context.
* Save results as JSON.



## ⚙️ Setup

### Prerequisites

* Python 3.12+
* Neo4j (community/enterprise, with APOC enabled)
* GPU recommended for embedding + reranking

### System Dependencies

Run `bash setup.sh`

```bash 
apt update && apt upgrade -y
apt-get update && apt-get install -y wget curl && rm -rf /var/lib/apt/lists/*

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Required libraries
apt-get update
apt-get install -y libgl1
apt-get install -y libglib2.0-0
```

### Python Environment Setup with uv

This project uses **uv** for dependency management. Ensure you have `pyproject.toml` and `uv.lock` files in your repository.

```bash
# Create a virtual environment
uv venv

# Sync dependencies from pyproject.toml and uv.lock
uv sync

```

### Configure Environment

Create a `.env` file with:

```env
URI=bolt://localhost:7687
USER=neo4j
PASSWORD=yourpassword
GROQ_API_KEY="api-key"
```


## 📊 Outputs

* **Neo4j Browser**: Explore KG at [http://localhost:7474](http://localhost:7474)
* **Vector Index**: Stored inside Neo4j
* **Answers**: JSON file in `runs/{timestamp}/answers.json`

