import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class APISelectorAgent:
    def __init__(self, faiss_directory, api_key):
        self.vectorstore = FAISS.load_local(
            faiss_directory,
            OpenAIEmbeddings(openai_api_key=api_key),
            allow_dangerous_deserialization=True
        )

    def select_api(self, query, top_k=10):
        results = self.vectorstore.similarity_search_with_score(query, k=top_k)
        formatted_results = []

        for doc, score in results:
            metadata = doc.metadata
            formatted_results.append({
                "endpoint": metadata.get("endpoint"),
                "description": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "body": metadata.get("body") if metadata.get("body") else "No body available",
                "file_name": metadata.get("file_name"),
                "distance": score
            })

        return formatted_results
    

# if __name__ == "__main__":
#     faiss_directory = "data/embeddings"
#     api_key = os.getenv("OPENAI_API_KEY")

#     agent = APISelectorAgent(faiss_directory, api_key)

#     query = "How to update passenger info?"
#     results = agent.select_api(query, top_k=3)

#     if results:
#         for result in results:
#             print(f"Endpoint: {result['endpoint']}")
#             print(f"Description: {result['description']}")
#             print(f"API Body: {result['body']}")
#             print(f"File Name: {result['file_name']}")
#             print(f"Relevance Score: {result['distance']}\n")
#     else:
#         print("No relevant API found.")
