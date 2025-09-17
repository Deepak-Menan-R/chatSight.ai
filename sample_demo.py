# app.py
import os
import json
import pickle
import requests
import faiss
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# =====================================
# 🔹 Load Environment
# =====================================
load_dotenv()

client = OpenAI(
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)
embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index")
METADATA_STORE_PATH = os.getenv("METADATA_STORE_PATH", "./faiss_metadata.pkl")
TOP_K = int(os.getenv("TOP_K", "5"))

# =====================================
# 🔹 FAISS + Metadata
# =====================================
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_STORE_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH, "rb") as f:
        metadata_store = pickle.load(f)
    print("✅ Loaded FAISS index & metadata")
else:
    index = faiss.IndexFlatL2(1536)  # embedding dim for text-embedding-3-small
    metadata_store = []
    print("⚠️ No FAISS index found, starting fresh")

# =====================================
# 🔹 Helpers
# =====================================
# def record_to_text(record: dict) -> str:
#     print(record)
#     ids = json.loads(record.get("id", ""))
#     tittle = json.loads(record.get("tittle", ""))
#     description = json.loads(record.get("description", ""))
#     category = json.loads(record.get("category", ""))
#     price = json.loads(record.get("price", ""))
#     discountPercentage = json.loads(record.get("discountPercentage", ""))
#     rating = json.loads(record.get("rating", ""))
#     stocks = json.loads(record.get("stocks", ""))
#     tags = json.loads(record.get("tags", "[]"))
    
#     return f"""
# id = {ids}
#     tittle = {tittle}
#     description = {description}
#     category = {category}
#     price = {price}
#     discountPercentage = {discountPercentage}
#     rating = {rating}
#     stocks = {stocks}
#     tags = {tags}
# """.strip()
    

def embed_texts(texts: List[str]):
    response = client.embeddings.create(model=embed_model, input=texts)
    return [d.embedding for d in response.data]

def record_to_text(product: dict) -> str:
    """Convert product dict into text block (id → tags, plus context)."""
    return f"""
id = {product.get("id")}
title = {product.get("title")}
description = {product.get("description")}
category = {product.get("category")}
price = {product.get("price")}
discountPercentage = {product.get("discountPercentage")}
rating = {product.get("rating")}
stock = {product.get("stock")}
tags = {", ".join(product.get("tags", []))}
""".strip()


def ingest_records(data):
    """Ingest products JSON into FAISS index."""
    global index, metadata_store

    # Ensure data is dict
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            print("❌ Failed to parse string input as JSON")
            return

    products = data.get("products", [])
    if not products:
        print("⚠️ No products found in data")
        return

    # Convert each product into text
    docs = [record_to_text(p) for p in products]

    # Create embeddings
    embeddings = embed_texts(docs)
    vectors = np.array(embeddings).astype("float32")

    # Add vectors to FAISS
    index.add(vectors)

    # Save metadata: only id and context (id + tags)
    for i, p in enumerate(products):
        metadata_store.append({
            "id": p.get("id"),
            "context": {
                "title": p.get("title"),
                "description": p.get("description"),
                "category": p.get("category"),
                "tags": p.get("tags", [])
            }
        })

    # Persist index + metadata
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH, "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"✅ Ingested {len(products)} products into FAISS")


def load_data_on_startup():
    if len(metadata_store) == 0 and os.path.exists("ecommerce-sample-data.txt"):
        with open("ecommerce-sample-data.txt", "r", encoding="utf-8") as f:
            data = json.load(f)
        try:
            response = requests.get("https://dummyjson.com/products")
            response.raise_for_status()
            # Try to parse as JSON
            try:
                data = response.json()
                text = json.dumps(data, indent=2)
            except Exception:
                text = response.text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to fetch or parse URL: {e}")
        ingest_records(text)

# =====================================
# 🔹 FastAPI
# =====================================
app = FastAPI(title="RAG + FAISS Table Retriever")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_data_on_startup()

# Pydantic model for JSON request
class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K

@app.post("/query")
def query(req: QueryRequest):
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No FAISS index loaded. Please add data first.")

    q_embedding = embed_texts([req.question])[0]
    q_vector = np.array([q_embedding]).astype("float32")
    D, I = index.search(q_vector, req.top_k)

    if len(I[0]) == 0:
        return {"answer": "I don't know based on the provided data.", "sources": []}

    # retrieved = [metadata_store[i] for i in I[0]]
    # context = "\n".join([f"[{r['interactionId']}] {r['text']}" for r in retrieved])
    # print("Retrieved content: ", context)
    
    retrieved = [metadata_store[i] for i in I[0]]

    context = "\n".join([
        f"[{r['id']}] {r['context']}" for r in retrieved
    ])

#     prompt = f"""
# You are a data assistant. Only use the context rows below to answer. Try to understand miss spells but dont forget to mention that you assume this is the input and for that input you are giving this output.
# If answer is not found, reply exactly: "I don't know based on the provided data."
# IF input mentions anything about date/ Time then first convert the epoch value to standard format and answer it.

# Context:
# {context}

# Question:
# {req.question}

# Answer:
# """


    prompt = f"""
    You are a data assistant. Only use the context rows below to answer. 
    If answer is not found, reply exactly: "I don't know based on the provided data."
    

    Context:
    {context}

    Question:
    {req.question}

    Answer:
    """
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": resp.choices[0].message.content
    }

# =====================================
# 🔹 Run App
# =====================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
