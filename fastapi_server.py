# FastAPI server for question answering
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import faiss
import requests

INDEX_PATH = "faiss_index.bin"
META_PATH = "vector_db_meta.npz"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "llama3"

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# Load index and metadata at startup
index = faiss.read_index(INDEX_PATH)
meta = np.load(META_PATH, allow_pickle=True)
table = meta["table"]
row_idx = meta["row_idx"]
chunk_text = meta["chunk_text"]

def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": text}
    )
    response.raise_for_status()
    data = response.json()
    return np.array(data["embedding"], dtype=np.float32)

@app.post("/ask")
def ask_question(req: QueryRequest):
    question = req.question
    emb = get_embedding(question)
    emb = np.expand_dims(emb, axis=0)
    D, I = index.search(emb, k=3)  # top 3 results
    results = []
    for idx in I[0]:
        results.append({
            "table": str(table[idx]),
            "row_idx": int(row_idx[idx]),
            "chunk": str(chunk_text[idx])
        })
    return {"question": question, "results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_server:app", host="0.0.0.0", port=8000, reload=True)