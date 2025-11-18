import pandas as pd
import numpy as np
import faiss
import requests
from getData import fetch_data_from_db

# Ollama local embedding config
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "llama3"

# Helper: chunk text

def chunk_text(text, chunk_size=256):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i+chunk_size])

# Helper: get embedding

def get_embedding(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": text}
    )
    response.raise_for_status()
    data = response.json()
    return np.array(data["embedding"], dtype=np.float32)

# Main: create vector DB

def create_vector_db():
    data = fetch_data_from_db()
    all_chunks = []
    chunk_sources = []
    print("Starting chunking process...")
    for table_name, df in data.items():
        print(f"Chunking table: {table_name} ({len(df)} rows)")
        for idx, row in df.iterrows():
            row_text = " ".join([str(x) for x in row.values])
            for chunk in chunk_text(row_text):
                all_chunks.append(chunk)
                chunk_sources.append((table_name, idx))
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1} rows in {table_name}")
    print(f"Chunking complete. Total chunks: {len(all_chunks)}")
    print("Starting embedding process...")
    embeddings = []
    for i, chunk in enumerate(all_chunks):
        emb = get_embedding(chunk)
        embeddings.append(emb)
        if (i + 1) % 100 == 0 or (i + 1) == len(all_chunks):
            print(f"  Embedded {i + 1}/{len(all_chunks)} chunks")
    embeddings = np.array(embeddings).astype('float32')
    print("Embedding complete.")
    # Create Faiss index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    # Save index and sources
    faiss.write_index(index, "faiss_index.bin")
    pd.DataFrame(chunk_sources, columns=["table", "row_idx"]).to_csv("chunk_sources.csv", index=False)
    print(f"Vector DB created with {len(all_chunks)} chunks.")

if __name__ == "__main__":
    create_vector_db()
