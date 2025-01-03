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

    def select_api(self, query, top_k=3):
        query_embedding = np.array([self.embedding_model.embed_query(query)], dtype=np.float32)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k * 5)  # Fetch more results initially

        results = []
        seen_endpoints = set()  # To track unique endpoints

        # Normalize distances to calculate relevance scores
        min_distance = float(np.min(distances[0]))
        max_distance = float(np.max(distances[0]))
        range_distance = max_distance - min_distance if max_distance > min_distance else 1.0

        for i, idx in enumerate(indices[0]):
            if idx == -1:  # No match found
                continue

            # Metadata for the chunk
            chunk_metadata = self.metadata[idx]
            endpoint = chunk_metadata.get("endpoint", None)

            # Skip if there's no endpoint or duplicate endpoints
            if not endpoint or endpoint in seen_endpoints:
                continue

            seen_endpoints.add(endpoint)  # Mark endpoint as seen

            # Trim description to 200 characters
            description = chunk_metadata["text"]
            if len(description) > 200:
                description = description[:200] + "..."

            # Parse body into structured JSON if applicable
            raw_body = chunk_metadata.get("body", None)
            try:
                body = json.loads(raw_body) if raw_body else None
            except json.JSONDecodeError:
                body = None  # Skip if the body is not valid JSON

            # Calculate relevance score (inverse of normalized distance)
            normalized_distance = (distances[0][i] - min_distance) / range_distance
            relevance_score = 1 - normalized_distance  # Higher is better

            results.append({
                "endpoint": endpoint,
                "description": description,
                "body": body,
                "file_name": chunk_metadata["file_name"],
                "distance": relevance_score  # Use the new relevance score
            })

            # Stop when we have enough unique results
            if len(results) >= top_k:
                break

        # Sort results by the new relevance score in descending order
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
