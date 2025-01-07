import os
import streamlit as st
from langchain_openai import ChatOpenAI
from src.api_selector import APISelectorAgent
from src.construct_prompt import ExplanationAgent

FAISS_INDEX_FILE = "data/embeddings/faiss_index"
METADATA_FILE = "data/embeddings/metadata.json"

api_key = os.getenv("OPENAI_API_KEY")
agent = APISelectorAgent(FAISS_INDEX_FILE, METADATA_FILE, api_key)

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

process_agent = ExplanationAgent(llm)

st.title("API Selector Chatbot")
st.subheader("Find the most relevant API documentation based on your query.")

# Input Query
user_query = st.text_input("Enter your query:", placeholder="E.g., How do I retrieve booking details?")

if st.button("Search API"):
    if user_query.strip():
        with st.spinner("Searching for the most relevant API..."):
            results = agent.select_api(user_query, top_k=3)

        if results:
            explanation = process_agent.generate_explanation(user_query, results)
            st.success("Here are the most relevant APIs:")
            st.markdown(explanation.content)
            # for i, result in enumerate(results):
            #     st.markdown(f"### Match {i+1}")
            #     if result['endpoint']:
            #         st.markdown(f"**Endpoint:** `{result['endpoint']}`")
            #     st.markdown(f"**Description:** {result['description']}")
            #     if result['body']:  # Only show body if it exists
            #         st.markdown("**API Body:**")
            #         st.json(result['body'])
            #     st.markdown(f"**File Name:** {result['file_name']}")
            #     st.markdown(f"**Relevance Score:** `{result['distance']:.4f}`")
            #     st.divider()  # Separator for readability
        else:
            st.warning("No relevant API found.")

    else:
        st.error("Please enter a valid query.")

st.markdown(
    """
    **Note:** This tool uses FAISS for local vector search and OpenAI embeddings for query understanding.
    """
)
