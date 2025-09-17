# app.py
import os
import json
import pickle
from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import numpy as np
import pandas as pd

# LangChain & OpenAI
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./faiss_index")
METADATA_STORE_PATH = os.environ.get("METADATA_STORE_PATH", "./faiss_metadata.pkl")
TOP_K = int(os.environ.get("TOP_K", "5"))

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in environment")

app = FastAPI(title="RAG + FAISS Table Retriever")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class IngestRequest(BaseModel):
    table_name: str
    rows: List[Dict[str, Any]]  # list of JSON rows

class QueryRequest(BaseModel):
    question: str
    table_name: str = None
    top_k: int = TOP_K

# Initialize embeddings and LLM

llm = OpenAI(
    model_name=CHAT_MODEL,
    openai_api_key=OPENAI_API_KEY,
    base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1"),
    temperature=0.0,
)

embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    openai_api_key=OPENAI_API_KEY,
    base_url=os.getenv("LITELLM_BASE_URL", "http://localhost:4000/v1"),
)
# We'll keep a metadata list mapping doc_ids -> metadata to persist alongside FAISS
metadata_store: Dict[str, Dict[str, Any]] = {}

# Helper to serialize a row into readable text for embedding
def row_to_text(row: Dict[str, Any], table_name: str = "") -> str:
    # simple deterministic serialization: "col1: val1 col2: val2 ..."
    parts = []
    for k, v in row.items():
        parts.append(f"{k}:{v}")
    return f"table:{table_name} " + " ".join(parts)

# Ingest function: build Document list and add to FAISS
def ingest_table(table_name: str, rows: List[Dict[str, Any]]):
    global metadata_store, faiss_index
    docs: List[Document] = []
    for i, row in enumerate(rows):
        doc_id = f"table:{table_name}:row:{row.get('id', i)}"
        text = row_to_text(row, table_name)
        meta = {"doc_id": doc_id, "table": table_name, "row": row}
        metadata_store[doc_id] = meta
        docs.append(Document(page_content=text, metadata=meta))

    # If FAISS index exists on disk, load; otherwise create new
    if os.path.exists(FAISS_INDEX_PATH):
        # load existing datastore
        faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
        # add documents
        faiss_index.add_documents(docs)
    else:
        # create new index
        faiss_index = FAISS.from_documents(docs, embeddings)

    # persist index + metadata
    faiss_index.save_local(FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    return {"ingested_count": len(docs), "table_name": table_name}

# Load persisted metadata & faiss on startup if available
faiss_index = None
if os.path.exists(METADATA_STORE_PATH) and os.path.exists(FAISS_INDEX_PATH):
    try:
        with open(METADATA_STORE_PATH, "rb") as f:
            metadata_store = pickle.load(f)
        faiss_index = FAISS.load_local(FAISS_INDEX_PATH, embeddings)
        print("Loaded FAISS index and metadata store from disk.")
    except Exception as e:
        print("Could not load persisted FAISS index:", e)


@app.post("/ingest")
def ingest(req: IngestRequest):
    result = ingest_table(req.table_name, req.rows)
    return {"status": "ok", **result}


def build_prompt(question: str, rows: List[Dict[str, Any]]) -> str:
    """
    Build a retrieval-augmented prompt that instructs the model to use only the provided rows.
    Also instruct the model to cite doc_ids.
    """
    context_blocks = []
    for meta in rows:
        doc_id = meta["doc_id"]
        row = meta["row"]
        # Put a human-friendly representation of the row
        row_items = ", ".join([f"{k}={v}" for k, v in row.items()])
        context_blocks.append(f"[{doc_id}] {row_items}")

    context_text = "\n".join(context_blocks)
    prompt = f"""
You are an assistant that must answer using ONLY the facts present in the provided table rows below.
If the answer cannot be determined from the provided rows, reply exactly: "I don't know based on the provided data."
Cite (in square brackets) the doc_id(s) you used to answer.

Context rows:
{context_text}

Question:
{question}

Answer concisely, and then provide a one-line list of doc_id(s) you used (as 'SOURCES: [docid,...]').
"""
    return prompt

@app.post("/query")
def query(req: QueryRequest):
    if faiss_index is None:
        raise HTTPException(status_code=400, detail="No FAISS index loaded. Please ingest data first.")

    top_k = req.top_k or TOP_K
    # Simple semantic search: get Documents with metadata
    docs = faiss_index.similarity_search(req.question, k=top_k)

    # We return the original rows as the 'related table'
    related_metas = []
    for d in docs:
        meta = d.metadata
        related_metas.append(meta)

    prompt = build_prompt(req.question, related_metas)
    llm_resp = llm.invoke(prompt)
    answer_text = llm_resp.strip() if isinstance(llm_resp, str) else str(llm_resp)

    rows = [m["row"] for m in related_metas]
    if rows:
        df = pd.DataFrame(rows)
        markdown_table = df.to_markdown(index=False)
    else:
        markdown_table = "No related rows found."

    sources = [m["doc_id"] for m in related_metas]

    return {
        "answer": answer_text,
        "related_rows": rows,
        "markdown_table": markdown_table,
        "sources": sources,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
