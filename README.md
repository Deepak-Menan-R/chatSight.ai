# RAG + FAISS Table Retriever

This project is a **FastAPI-based Retrieval-Augmented Generation (RAG) service**.  
It ingests tabular data (JSON rows), stores them in a **FAISS vector database** using OpenAI (or LiteLLM proxy) embeddings, and answers natural language questions using GPT models with retrieved context.

---

## Features
1. Ingest any tabular dataset (as JSON rows).
2. Store embeddings in **FAISS** for semantic similarity search.
3. Query data with natural language.
4. GPT will **only answer using provided rows** and cite sources.
5. Metadata store persisted alongside FAISS.
6. REST API with Swagger docs at `/docs`.

---

## Requirements
1. Python 3.9+ recommended  
2. [OpenAI API key](https://platform.openai.com/account/api-keys) (or LiteLLM proxy endpoint)

---

## Setup Instructions

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd <project-folder>
