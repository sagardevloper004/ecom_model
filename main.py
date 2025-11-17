from getData import fetch_data_from_db
import pandas as pd
from typing import List, Optional
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


if __name__ == "__main__":
    data = fetch_data_from_db()

    # Build a FAISS vector DB from the already-loaded `data` DataFrame (e.g. data["product"])

    def build_vector_db_from_dataframe(
        df: pd.DataFrame,
        text_columns: Optional[List[str]] = None,
        db_path: str = "data_faiss_index",
    ) -> FAISS:
        """
        Build and save a FAISS vector DB from a DataFrame.
        - df: DataFrame containing product rows
        - text_columns: list of columns to concatenate for embedding (default: all columns)
        - db_path: local path to save the FAISS index
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None")

        if text_columns is None:
            text_columns = df.columns.tolist()

        # prepare text chunks and metadata
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = []
        metadatas = []

        for idx, row in df.iterrows():
            parts = []
            for col in text_columns:
                if col in df.columns and pd.notna(row[col]):
                    parts.append(str(row[col]).strip())
            full_text = " ".join(parts).strip()
            if not full_text:
                continue
            chunks = text_splitter.split_text(full_text)
            for i, c in enumerate(chunks):
                texts.append(c)
                metadatas.append({"row_index": int(idx), "chunk_index": i})

        if not texts:
            raise ValueError("No text extracted from DataFrame for embeddings.")

        # embed and build FAISS
        embedding = OllamaEmbeddings(model="llama2")
        vector_db = FAISS.from_texts(texts, embedding, metadatas=metadatas)
        vector_db.save_local(db_path)
        return vector_db


    # --- Use the `data` variable already loaded above ---
    # Try product DF first; if `data` itself is a DataFrame use it directly.
    try:
        product_df = None
        if isinstance(data, dict):
            # try common product key names
            for key in ("product", "products", "items"):
                if key in data and isinstance(data[key], pd.DataFrame):
                    product_df = data[key]
                    break
        elif isinstance(data, pd.DataFrame):
            product_df = data

        if product_df is None:
            print("No product DataFrame found in `data`. Provide a DataFrame or use the 'product' key.")
        else:
            # choose which columns to include in the embedding text
            cols = ["title", "price", "rating", "orginalPrice"]
            cols = [c for c in cols if c in product_df.columns]
            if not cols:
                # fallback to all columns
                cols = product_df.columns.tolist()

            faiss_db = build_vector_db_from_dataframe(product_df, text_columns=cols, db_path="data_faiss_index")
            print("Saved FAISS index to 'data_faiss_index' (num_vectors ~= {})".format(len(faiss_db.index_to_docstore_id)))
    except Exception as e:
        print("Error building FAISS DB:", str(e))