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

    def select_api(self, query, top_k=10):
        query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        seen_endpoints = set()

        min_distance = float(np.min(distances[0]))
        max_distance = float(np.max(distances[0]))
        range_distance = max_distance - min_distance if max_distance > min_distance else 1.0

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            chunk_metadata = self.metadata[idx]
            endpoint = chunk_metadata.get("endpoint", None)

            if not endpoint or endpoint in seen_endpoints:
                continue

            seen_endpoints.add(endpoint)

            description = chunk_metadata["text"]
            if len(description) > 200:
                description = description[:200] + "..."

            raw_body = chunk_metadata.get("body", None)
            try:
                body = json.loads(raw_body) if raw_body else None
            except json.JSONDecodeError:
                body = None

            normalized_distance = (distances[0][i] - min_distance) / range_distance
            relevance_score = 1 - normalized_distance

            results.append({
                "endpoint": endpoint,
                "description": description,
                "body": body,
                "file_name": chunk_metadata["file_name"],
                "distance": relevance_score
            })

        results = sorted(results, key=lambda x: x["distance"], reverse=True)
        return results


if __name__ == "__main__":
    faiss_index_file = "data/embeddings/faiss_index"
    metadata_file = "data/embeddings/metadata.json"

    api_key = os.getenv("OPENAI_API_KEY")
    agent = APISelectorAgent(faiss_index_file, metadata_file, api_key)

    query = "How to update passenger info?"
    results = agent.select_api(query, top_k=3)

    if results:
        for result in results:
            print(f"Endpoint: {result['endpoint']}")
            print(f"Description: {result['text']}")
            print(f"API Body: {result['body']}")
            print(f"File Name: {result['file_name']}")
            print(f"Relevance Score: {result['distance']}\n")
    else:
        print("No relevant API found.")
