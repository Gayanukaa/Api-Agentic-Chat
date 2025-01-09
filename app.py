import os
import streamlit as st
from langchain_openai import ChatOpenAI
from streamlit_chat import message
from src.api_selector import APISelectorAgent
from src.construct_prompt import ExplanationAgent

FAISS_DIRECTORY = "data/embeddings"

api_key = os.getenv("OPENAI_API_KEY")
agent = APISelectorAgent(FAISS_DIRECTORY, api_key)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

process_agent = ExplanationAgent(llm)

st.title("Navitaire API Selector")
st.subheader("Find the most relevant API documentation based on your query.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about APIs:"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Processing your query..."):
        results = agent.select_api(prompt.strip(), top_k=10)

        if results:
            response = process_agent.generate_explanation(prompt.strip(), results, st.session_state.messages)
            st.chat_message("assistant").markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})
        else:
            st.warning("No relevant API found.")
