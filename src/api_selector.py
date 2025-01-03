from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
import json
import os

api_key = os.getenv("OPENAI_API_KEY")

class APISelectorAgent:
    def __init__(self, faiss_index_file, metadata_file):
        self.index = faiss.read_index(faiss_index_file)

        with open(metadata_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

    def select_api(self, query, top_k=1):
        query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # No match found
                continue
            result = {
                "text": self.metadata[idx],
                "distance": distances[0][i]
            }
            results.append(result)

        return results

if __name__ == "__main__":
    faiss_index_file = "data/embeddings/faiss_index"
    metadata_file = "data/embeddings/metadata.json"

    agent = APISelectorAgent(faiss_index_file, metadata_file)

    query = "How do I retrieve booking details?"
    results = agent.select_api(query, top_k=3)

    if results:
        for result in results:
            print(f"Text: {result['text']}")
            print(f"Distance: {result['distance']}\n")
    else:
        print("No relevant API found.")