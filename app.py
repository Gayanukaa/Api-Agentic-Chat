import logging
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

# Streamlit App
st.title("API Selector Chatbot")
st.subheader("Find the most relevant API documentation based on your query.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("How to retrieve Booking details?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show loading animation
    with st.spinner("Processing your query..."):
        # Fetch relevant APIs
        results = agent.select_api(prompt.strip(), top_k=10)

        if results:
            # Check if the query is specific
            if "api" in prompt.lower() or "endpoint" in prompt.lower():
                # Focus on a single detailed API explanation
                specific_result = results[0]
                body = specific_result["body"]
                api_body = f"**Body:** ```json\n{body}```" if body else "**Body:** No body available"
                detailed_response = f"""
                ### Detailed API Information
                - **Endpoint:** `{specific_result['endpoint']}`
                - **Description:** {specific_result['description']}
                - **File Name:** {specific_result['file_name']}
                {api_body}
                """
                st.chat_message("assistant").markdown(detailed_response)
                st.session_state.messages.append({"role": "assistant", "content": detailed_response})
            else:
                # General step-by-step explanation
                response = process_agent.generate_explanation(prompt.strip(), results, st.session_state.messages)
                st.chat_message("assistant").markdown(response.content)
                st.session_state.messages.append({"role": "assistant", "content": response.content})
        else:
            st.warning("No relevant API found.")
