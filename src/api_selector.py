from langchain_openai import OpenAIEmbeddings
import faiss
import numpy as np
import json
import os

class APISelectorAgent:
    def __init__(self, faiss_index_file, metadata_file, api_key):
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
                "text": self.metadata[idx]["text"],
                "metadata": self.metadata[idx],
                "distance": distances[0][i]
            }
            # Only include chunks with an "endpoint" key
            if "endpoint" in self.metadata[idx]:
                results.append(result)

        # Sort by distance for better relevance
        results = sorted(results, key=lambda x: x["distance"])
        return results

if __name__ == "__main__":
    faiss_index_file = "data/embeddings/faiss_index2"
    metadata_file = "data/embeddings/metadata2.json"

    api_key = os.getenv("OPENAI_API_KEY")
    agent = APISelectorAgent(faiss_index_file, metadata_file, api_key)

    query = "How do I retrieve booking details?"
    results = agent.select_api(query, top_k=3)

    if results:
        for result in results:
            print(f"Endpoint: {result['metadata'].get('endpoint', 'N/A')}")
            print(f"Text: {result['text']}")
            print(f"Distance: {result['distance']}\n")
    else:
        print("No relevant API found.")
