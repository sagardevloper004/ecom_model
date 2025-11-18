from fastapi import FastAPI, Request
from pydantic import BaseModel
import faiss
import pandas as pd
import numpy as np
import os
from typing import List

app = FastAPI()

# Request model for POST endpoint
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3  # Number of results to return

# Load Faiss index and chunk sources
FAISS_INDEX_PATH = os.path.join(os.getcwd(), "faiss_index.bin")
CHUNK_SOURCES_PATH = os.path.join(os.getcwd(), "chunk_sources.csv")

# Dummy embedding function (replace with your actual embedding model)
def embed_text(text: str) -> np.ndarray:
    # Example: returns a random vector (replace with real embedding)
    return np.random.rand(768).astype('float32')

@app.post("/search")
def search_vector_db(request: QueryRequest):
    # Load Faiss index
    index = faiss.read_index(FAISS_INDEX_PATH)
    # Load chunk sources
    chunk_sources = pd.read_csv(CHUNK_SOURCES_PATH)
    # Embed the query
    query_vector = embed_text(request.query)
    # Search Faiss index
    D, I = index.search(np.expand_dims(query_vector, axis=0), request.top_k)
    # Get results from chunk_sources
    results = []
    for idx in I[0]:
        if idx < len(chunk_sources):
            row = chunk_sources.iloc[idx].to_dict()
            results.append(row)
    return {"results": results}
