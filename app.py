import logging
import json
import os
import streamlit as st
from langchain_openai import ChatOpenAI
from streamlit_chat import message
from src.api_selector import APISelectorAgent
from src.construct_prompt import ExplanationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAISS_INDEX_FILE = "data/embeddings/faiss_index"
METADATA_FILE = "data/embeddings/metadata.json"

api_key = os.getenv("OPENAI_API_KEY")
agent = APISelectorAgent(FAISS_INDEX_FILE, METADATA_FILE, api_key)

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

process_agent = ExplanationAgent(llm)

## Streamlit App

st.title("API Selector Chatbot")
st.subheader("Find the most relevant API documentation based on your query.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How to retrieve Booking details?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get results from the API Selector Agent
    results = agent.select_api(prompt.strip(), top_k=10)
    if results:
        # Pass results directly to the Explanation Agent
        response = process_agent.generate_explanation(prompt.strip(), results, st.session_state.messages)

        # Display the explanation only
        st.chat_message("assistant").markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
    else:
        st.warning("No relevant API found.")
