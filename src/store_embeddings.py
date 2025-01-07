import os
import json
import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

def index_embeddings(chunk_file, faiss_index_file, metadata_file):
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Initialize FAISS index
    dimension = 1536
    index = faiss.IndexFlatL2(dimension)
    metadata = []

    # Process chunks
    for chunk in chunks:
        embedding = embedding_model.embed_query(chunk["text"])
        index.add(np.array([embedding], dtype=np.float32))
        metadata.append({
            "text": chunk["text"],
            "endpoint": chunk["metadata"].get("endpoint"),
            "body": chunk["metadata"].get("body"),
            "file_name": chunk["metadata"].get("file_name")
        })

    faiss.write_index(index, faiss_index_file)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Indexed {len(chunks)} chunks in FAISS.")
    print(f"FAISS index saved to {faiss_index_file}.")
    print(f"Metadata saved to {metadata_file}.")

if __name__ == "__main__":
    chunk_file = "data/chunks/chunked_data.json"
    faiss_index_file = "data/embeddings/faiss_index"
    metadata_file = "data/embeddings/metadata.json"

    os.makedirs(os.path.dirname(faiss_index_file), exist_ok=True)
    index_embeddings(chunk_file, faiss_index_file, metadata_file)
