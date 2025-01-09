import json
import os
import faiss
import numpy as np
from dotenv import load_dotenv
#from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

def index_embeddings(chunk_file, faiss_directory):
    # Load chunks and metadata
    with open(chunk_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = data["texts"]
    metadatas = data["metadata"]

    # Create FAISS index with LangChain
    vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)

    # Save FAISS index locally
    vectorstore.save_local(faiss_directory)
    print(f"Indexed {len(texts)} chunks in FAISS.")
    print(f"FAISS index saved to {faiss_directory}.")

if __name__ == "__main__":
    chunk_file = "data/chunks/chunked_data.json"
    faiss_directory = "data/embeddings"
    os.makedirs(faiss_directory, exist_ok=True)
    index_embeddings(chunk_file, faiss_directory)
