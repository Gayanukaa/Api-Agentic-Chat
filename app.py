import os
import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from src.api_selector import APISelectorAgent
from src.construct_prompt import ExplanationAgent
import logging
FAISS_INDEX_FILE = "data/embeddings/faiss_index"
METADATA_FILE = "data/embeddings/metadata.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api_key = os.getenv("OPENAI_API_KEY")
agent = APISelectorAgent(FAISS_INDEX_FILE, METADATA_FILE, api_key)

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

process_agent = ExplanationAgent(llm)

import streamlit as st
from streamlit_chat import message

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
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    results = agent.select_api(prompt.strip(), top_k=3)
    if results:
        response = process_agent.generate_explanation(prompt.strip(), results)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response.content)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.content})
    else:
            st.warning("No relevant API found.")
