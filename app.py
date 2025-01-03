import streamlit as st
from src.api_selector import APISelectorAgent
import os

# File paths
FAISS_INDEX_FILE = "data/embeddings/faiss_index"
METADATA_FILE = "data/embeddings/metadata.json"

# Initialize API Selector Agent
api_key = os.getenv("OPENAI_API_KEY")
agent = APISelectorAgent(FAISS_INDEX_FILE, METADATA_FILE, api_key)

# Streamlit Interface
st.title("API Selector Chatbot")
st.subheader("Find the most relevant API documentation based on your query.")

# Input Query
user_query = st.text_input("Enter your query:", placeholder="E.g., How do I retrieve booking details?")

# Button to submit query
if st.button("Search API"):
    if user_query.strip():
        # Perform API selection
        with st.spinner("Searching for the most relevant API..."):
            results = agent.select_api(user_query, top_k=3)

        if results:
            st.success("Here are the most relevant APIs:")
            for i, result in enumerate(results):
                st.markdown(f"### Match {i+1}")
                if result['endpoint']:
                    st.markdown(f"**Endpoint:** `{result['endpoint']}`")
                st.markdown(f"**Description:** {result['description']}")
                if result['body']:  # Only show body if it exists
                    st.markdown("**API Body:**")
                    st.json(result['body'])
                st.markdown(f"**File Name:** {result['file_name']}")
                st.markdown(f"**Relevance Score:** `{result['distance']:.4f}`")
                st.divider()  # Separator for readability
        else:
            st.warning("No relevant API found.")

    else:
        st.error("Please enter a valid query.")

# Footer
st.markdown(
    """
    **Note:** This tool uses FAISS for local vector search and OpenAI embeddings for query understanding.
    """
)
