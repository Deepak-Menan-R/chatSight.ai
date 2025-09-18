# app.py

import os
import json
import pickle
import faiss
import requests
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

import time


load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "Gemini-2.5-flash-lite")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "./faiss_index")
METADATA_STORE_PATH = os.environ.get("METADATA_STORE_PATH", "./faiss_metadata.pkl")
ASSIST_SKEY = os.environ.get("ASSIST_SKEY", "")
ASSIST_AGENT_SESSION_ID = os.environ.get("ASSIST_AGENT_SESSION_ID", "")
ASSIST_TOKEN = os.environ.get("ASSIST_TOKEN", "")
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
    table_name: str = "interactions_table"

class QueryRequest(BaseModel):
    question: str
    table_name: str = None
    top_k: int = TOP_K

# Initialize embeddings and LLM

client = OpenAI(
    base_url=os.getenv("LITELLM_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)
embed_model = EMBED_MODEL
chat_model = CHAT_MODEL

if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_STORE_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH, "rb") as f:
        metadata_store = pickle.load(f)
    print("✅ Loaded FAISS index & metadata")
else:
    index = faiss.IndexFlatL2(1536)  # embedding dim for text-embedding-3-small
    metadata_store = []
    print("⚠️ No FAISS index found, starting fresh")


# Helper to serialize a row into readable text for embedding
def row_to_text(row: Dict[str, Any], table_name: str = "") -> str:  
    # simple deterministic serialization: "col1: val1 col2: val2 ..."
    parts = []
    for k, v in row.items():
        parts.append(f"{k}:{v}")
    result = f"table:{table_name} " + " ".join(parts)
    print(result)
    return result

def embed_texts(texts: List[str]):
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in response.data]

# Ingest function: build Document list and add to FAISS



def fetch_transcripts(interaction_ids: List[str], items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fetch transcripts for individual interaction IDs using the enhanced transcript API
    """
    transcripts_map = {}

    # Create a mapping of interaction_id -> vsId for visitor ID lookup
    interaction_to_vsid = {}
    for item in items:
        if isinstance(item, dict) and "interactionId" in item and "vsId" in item:
            interaction_to_vsid[item["interactionId"]] = item["vsId"]

    for interaction_id in interaction_ids:
        try:
            # Get visitor ID from the item data
            time.sleep(0.1)  # Sleep for 100ms between API calls
            visitor_id = interaction_to_vsid.get(interaction_id, "")  # fallback to default if not found

            # Prepare headers for the transcript API call
            headers = {
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9,en-IN;q=0.8',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'application/x-www-form-urlencoded',
                'Pragma': 'no-cache',
                'Referer': f'https://consoleusw1.portal.assist.staging.247-inc.net/en/console?_skey={ASSIST_SKEY}&locale=en_US&status=default',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-origin',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 Edg/140.0.0.0',
                'X-Requested-With': 'XMLHttpRequest',
                'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Microsoft Edge";v="140"',
                'sec-ch-ua-mobile': '?0',
                'sec-ch-ua-platform': '"Windows"'
            }

            cookies = {
                'JSESSIONID': 'node0ep66oynbz8wc1vrra0l9odifk1738.node0',
                '_tokenId': ASSIST_TOKEN,
                '_clientId': 'nemo-client-pedemo',
                '_domainId': 'pedemo',
                '_locale': 'en_US',
                '_userAuthType': 'local-auth-user',
                '_dc': 'consoleusw1',
                '_jsVersion': '3.30.0',
                '__asId': ASSIST_AGENT_SESSION_ID,
                'SERVER': 'usw1_1',
                'BAYEUX_BROWSER': '1fol0vdixnsrl1ic',
                'CSRF-TOKEN': '16975072348901221187012439763872'
            }

            # Build the URL with interaction ID in the path and visitor ID from item data
            url = f'https://consoleusw1.portal.assist.staging.247-inc.net/en/enhanced_interaction_transcript/rest/transcript/{interaction_id}/visitor/{visitor_id}'

            # Query parameters
            params = {
                'context': '',
                '_skey': ASSIST_SKEY,
                'session': ASSIST_AGENT_SESSION_ID,
                'diagSeq': '417',
                'clientId': 'nemo-client-pedemo',
                'locale': 'en-us',
                'dateRangeStart': '1',
                'dateRangeEnd': '1758220199980',
                'pageSize': '1',
                'pageNumber': '0',
                'isHistoryDataRequested': 'true',
                'queryCriteria': f'{{"orderBy":[{{"sortingOrder":"DESC","fieldName":"startTime"}}],"filters":[{{"fieldName":"vsId","operation":"EQ","value":"{visitor_id}"}},{{"fieldName":"InteractionId","operation":"EQ","value":"{interaction_id}"}}]}}'
            }

            response = requests.get(
                url,
                headers=headers,
                cookies=cookies,
                params=params
            )
            response.raise_for_status()

            transcript_response = response.json()
            print(f"DEBUG: Transcript API response for {interaction_id}: {list(transcript_response.keys()) if isinstance(transcript_response, dict) else 'Not a dict'}")

            # Extract transcript data from response
            if isinstance(transcript_response, dict):
                # Look for transcript data in various possible locations
                transcript_data = transcript_response.get('data', transcript_response.get('items', transcript_response.get('messages', transcript_response)))
                transcripts_map[interaction_id] = transcript_data
            else:
                transcripts_map[interaction_id] = transcript_response

        except Exception as e:
            print(f"ERROR: Failed to fetch transcript for {interaction_id}: {e}")
            transcripts_map[interaction_id] = None

    print(f"DEBUG: Extracted transcripts for {len([k for k, v in transcripts_map.items() if v is not None])} out of {len(interaction_ids)} interactions")
    return transcripts_map

def create_documents_from_api_response(table_name: str = "interactions_table"):
    """
    Fetch interactions from API and create Document objects with transcripts
    Returns a list of Document objects ready for FAISS indexing
    """
    global metadata_store

    # Convert curl command to requests
    headers = {
        'Accept': '*/*',
        'Accept-Language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Origin': 'https://consoleusw1.portal.assist.staging.247-inc.net',
        'Pragma': 'no-cache',
        'Referer': f'https://consoleusw1.portal.assist.staging.247-inc.net/en/console?_skey={ASSIST_SKEY}&locale=en_US&status=default',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Chromium";v="140", "Not=A?Brand";v="24", "Google Chrome";v="140"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"'
    }

    cookies = {
        'JSESSIONID': 'node0e9d6xojaxtyozvmyklj8wf9r1630.node0',
        '_tokenId': ASSIST_TOKEN,
        '_clientId': 'nemo-client-pedemo',
        '_domainId': 'pedemo',
        '_locale': 'en_US',
        '_userAuthType': 'local-auth-user',
        '_dc': 'consoleusw1',
        '_jsVersion': '3.30.0',
        '__asId': ASSIST_AGENT_SESSION_ID,
        'SERVER': 'usw1_1',
        'BAYEUX_BROWSER': '1gtmcff3fh9gr13f'
    }

    data = 'queryCriteria=%7B%22orderBy%22%3A%5B%7B%22sortingOrder%22%3A%22DESC%22%2C%22fieldName%22%3A%22endTime%22%7D%5D%2C%22filters%22%3A%5B%7B%22fieldName%22%3A%22interactionEndState%22%2C%22operation%22%3A%22EQ%22%2C%22value%22%3A%22DISPOSED%22%7D%5D%7D&queryMetadata=%7B%7D&timeType=endTime&dateRangeStart=1757442600000&dateRangeEnd=1758133799000&pageSize=25&pageNumber=0&isFilteringDone=false&navigationDirection=FIRST&timeBasedMarkerObj=null&isTranscriptRequested=false&mode=search&searchType=team&locale=en-us'

    response = requests.post(
        f'https://consoleusw1.portal.assist.staging.247-inc.net/en/enhanced_interaction_history/rest/team/pedemo-account-default-team-default-allteams?_skey={ASSIST_SKEY}&session={ASSIST_AGENT_SESSION_ID}&diagSeq=147&clientId=nemo-client-pedemo',
        headers=headers,
        cookies=cookies,
        data=data
    )
    response.raise_for_status()

    # Try to parse as JSON
    data = response.json()
    data = data['data']

    if not isinstance(data, dict) or "items" not in data:
        raise ValueError("API response does not contain expected 'items' structure")

    items = data["items"]

    # Extract interaction IDs for transcript fetching
    interaction_ids = []
    for item in items:
        if isinstance(item, dict) and "interactionId" in item:
            interaction_ids.append(item["interactionId"])

    # Fetch transcripts for all interactions
    transcripts_data = {}
    if interaction_ids:
        try:
            transcripts_data = fetch_transcripts(interaction_ids, items)
            print(f"DEBUG: Fetched transcripts for {len(transcripts_data)} interactions")
        except Exception as e:
            print(f"WARNING: Failed to fetch transcripts: {e}")

    # Create documents for each item using interactionId as doc_id
    docs = []
    for item in items:
        # if isinstance(item, dict) and "interactionId" in item:
        #     # Add transcript to the item if available
        #     interaction_id = item["interactionId"]
        #     if interaction_id in transcripts_data:
        #         item["transcript"] = transcripts_data[interaction_id]

        #     doc_id = f"table:{table_name}:interaction:{item['interactionId']}"
        #     text = json.dumps(item, indent=2)
        #     meta = {"doc_id": doc_id}
        #     doc = Document(page_content=text, metadata=meta)
        #     docs.append(doc)
        #     metadata_store[doc_id] = meta
        text = json.dumps(item, indent=2)
        docs.append(text)

    # Logging for ingestion
    print(f"Ingested {len(docs)} interactions:")
    for doc in docs[:3]:  # Show first 3 for brevity
        print(doc)
        # print(f"doc_id: {doc.metadata['doc_id']}")
        # print(f"content (first 200 chars): {doc.page_content[:200]}")

    return docs


    
def record_to_text(record: dict) -> str:
            
    return f"context: {json.dumps(record, ensure_ascii=False)}"    

def ingest_url(table_name: str = "interactions_table"):
    global metadata_store, faiss_index, index
    try:
        # Get documents from API
        docs = create_documents_from_api_response(table_name)
        with open('output.json', 'w') as f:
            json.dump(docs, f)
        docs1 = [json.loads(item) for item in docs]
        with open('output1.txt', 'w') as f:
            f.write(json.dumps(docs1))
        print("DONE")
        # Add to FAISS
        if docs:
            print(f"DEBUG: Starting FAISS indexing for {len(docs)} documents...")
            start_time = time.time()
            final_Doc = json.dumps(docs1)
            
            pre_embed_doc = [record_to_text(r) for r in final_Doc]
            print(type(pre_embed_doc))
            embeddings = embed_texts(pre_embed_doc)
            vectors = np.array(embeddings).astype("float32")
            index.add(vectors)

            end_time = time.time()
            print(f"DEBUG: FAISS indexing completed in {end_time - start_time:.2f} seconds")

            print("DEBUG: Saving FAISS index to disk...")
            
            for i, r in enumerate(pre_embed_doc):
                metadata_store.append({
                    "interactionId": r.get("interactionId"),
                    "visitorName": r.get("visitorName"),
                    "agentName": r.get("ownerAgentName"),
                    "queueName": r.get("queueName"),
                    "startTime": r.get("startTime"),
                    "endTime": r.get("endTime"),
                    "text": docs[i],
                })

            faiss.write_index(index, FAISS_INDEX_PATH)
            with open(METADATA_STORE_PATH, "wb") as f:
                pickle.dump(metadata_store, f)
            print(f"✅ Ingested {len(docs)} records into FAISS")
            
            print("DEBUG: FAISS index saved successfully")

        return {"ingested_count": len(docs), "table_name": table_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse URL: {e}")
    
# if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_STORE_PATH):
#     index = faiss.read_index(FAISS_INDEX_PATH)
#     with open(METADATA_STORE_PATH, "rb") as f:
#         metadata_store = pickle.load(f)
#     print("✅ Loaded FAISS index & metadata")
# else:
#     index = faiss.IndexFlatL2(1536)  # embedding dim for text-embedding-3-small
#     metadata_store = []
#     print("⚠️ No FAISS index found, starting fresh")

# def embed_texts(texts: List[str]):
#     print("Premb_doc:", texts[:3])
#     response = client.embeddings.create(model=embed_model, input=texts[:2])
#     return [d.embedding for d in response.data]

def embed_texts(texts: List[str]):
    # Just truncate to first 5000 characters
    safe_texts = [text[:5000] for text in texts[:2]]
    print("Premb_doc:", [f"{text[:50]}..." for text in safe_texts])
    response = client.embeddings.create(model=embed_model, input=safe_texts)
    return [d.embedding for d in response.data]

@app.post("/query")
def query(req: QueryRequest):
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No FAISS index loaded. Please add data first.")

    q_embedding = embed_texts([req.question])[0]
    q_vector = np.array([q_embedding]).astype("float32")
    D, I = index.search(q_vector, req.top_k)
    print(D, I)
    if len(I[0]) == 0: 
        return {"answer": "I don't know based on the provided data.", "sources": []}

    retrieved = [metadata_store[i] for i in I[0]]
    context = "\n".join([f"[{r['interactionId']}] {r['text']}" for r in retrieved])
    prompt = f"""
        You are a precise data assistant. Follow these rules strictly:

        1. Always answer using ONLY the provided context.
        2. If the question contains spelling mistakes, assume the corrected intent, state your assumption, and answer accordingly.
        3. If the question involves any date/time, convert epoch values in the context to human-readable standard format before answering.
        4. Always return a complete and properly formatted answer — never partial or incomplete.
        5. If the request requires visualization (bar chart, pie chart, line graph, etc.) and contains a string "Render", return the output as a valid, fully formatted HTML snippet with inline styling (no explanation).
        6. For all non-visual answers, provide a clean, well-structured response in plain text or markdown formatting only.

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
        "answer": resp.choices[0].message.content,
        "sources": [r["interactionId"] for r in retrieved],
        "related": retrieved,
    }
    
    
    
def build_prompt(question: str, metas: List[Dict[str, Any]]) -> str:

    context_blocks = []
    added_interaction_ids = set()  # Track which interaction IDs have been added

    for meta in metas:
        doc_id = meta["doc_id"]
        # Extract just the interaction ID for cleaner display
        if ":interaction:" in doc_id:
            interaction_id = doc_id.split(":interaction:")[-1]
            display_label = f"Interaction_{interaction_id}"

            # Check if this interaction ID has already been added
            if interaction_id in added_interaction_ids:
                print(f"DEBUG: Skipping duplicate interaction ID: {interaction_id}")
                continue

            # Mark this interaction ID as added
            added_interaction_ids.add(interaction_id)
        else:
            display_label = doc_id

        content = meta.get("page_content", "")
        # Try to parse content as JSON and show keys/values for interactions
        try:
            json_obj = json.loads(content)
            if isinstance(json_obj, dict):
                # Check if this content appears to be raw item data and skip it
                if "items" in json_obj and isinstance(json_obj["items"], list):
                    print(f"DEBUG: Skipping item data for {display_label}")
                    continue


                context_blocks.append(f"[{display_label}] CONTENT:\n{content}")
            else:
                context_blocks.append(f"[{display_label}]\nCONTENT TYPE: {type(json_obj).__name__}\nCONTENT:\n{content}")
        except Exception:
            context_blocks.append(f"[{display_label}]\nCONTENT (not JSON):\n{content[:500]}")

    context_text = "\n\n".join(context_blocks)
    prompt = f"""You are a Lead supervisor for all customer care executives in a customer care center.
    You are responsible for analyzing customer interactions and support data. Use the provided interaction information to answer the question accurately.
    You are eligible to do very detailed analysis and provide information in HTML, JSON or any other format as required.
    no explanation needed on how the html or image was generated

    IMPORTANT: Always provide a complete response. If generating HTML or code, ensure all tags are properly closed and the response is fully formed.

    JSON FORMATTING RULES:
    - When providing JSON responses, format them cleanly without code block markers
    - Do NOT wrap JSON in ```json or ``` code blocks
    - Present JSON directly as formatted text that looks clean in chat bubbles
    - Use proper indentation and spacing for readability

    Context Content:
    {context_text}

    Question:
    {question}

    Please provide a complete and detailed answer. If creating visualizations or HTML, ensure the response is fully complete with all closing tags.

    SOURCES: [List the doc_id(s) you used]
    """
    return prompt

@app.on_event("startup")
def load_data_on_startup():
    """Ingest sample file into FAISS at startup."""
    ingest_url()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)