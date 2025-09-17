# RAG + FAISS Table Retriever 

This project is a **FastAPI-based Retrieval-Augmented Generation (RAG) service**.  
It ingests tabular data (JSON rows), stores them in a **FAISS vector database** using OpenAI (or LiteLLM proxy) embeddings, and answers natural language questions using GPT models with retrieved context.  

---

## 🚀 Features  
1. Ingest any tabular dataset (as JSON rows).  
2. Store embeddings in **FAISS** for semantic similarity search.  
3. Query data with natural language.  
4. GPT will **only answer using provided rows** and cite sources.  
5. Metadata store persisted alongside FAISS.  
6. REST API with Swagger docs at `/docs`.  

---

## 📦 Requirements  
1. Python 3.9+ recommended  
2. [OpenAI API key](https://platform.openai.com/account/api-keys) (or LiteLLM proxy endpoint)  

---

## ⚙️ Setup Instructions  

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <project-folder>

# 2. Create virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create a .env file in the project root with the following content:
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXX 
LITELLM_BASE_URL=https://litellm-dev.cloud.247-inc.net/
OPENAI_CHAT_MODEL=gemini-2.5-flash
OPENAI_EMBED_MODEL=text-embedding-3-small      
FAISS_INDEX_PATH=./faiss_index
METADATA_STORE_PATH=./faiss_metadata.pkl
TOP_K=5

# 6. Run the service
python app.py
