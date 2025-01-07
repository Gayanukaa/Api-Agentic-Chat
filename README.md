# API-Agentic-Chat

**API-Agentic-Chat** is a smart chatbot designed to assist users in navigating API documentation efficiently. 

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/Vector%20Storage-FAISS-brightgreen)](https://faiss.ai/)
[![OpenAI](https://img.shields.io/badge/Embeddings-OpenAI-blueviolet)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain-lightgrey)](https://langchain.readthedocs.io/)


## Features

- **Natural Language Query**: Users can ask API-related questions in plain language.
- **Intelligent Retrieval**: Identifies the most relevant APIs using embeddings and FAISS.
- **Step-by-Step Explanations**: Combines results into a coherent response using a Large Language Model (LLM).
- **Contextual Understanding**: Maintains context throughout the conversation to provide accurate and relevant information.

## Project Structure

```plaintext
API-Agentic-Chat/
├── reference_docs/          # API documentation PDFs
├── data/                    # Processed data
│   ├── chunks/              # Chunked data from PDFs
│   │   ├── chunked_data.json
│   ├── embeddings/          # FAISS index and metadata
│       ├── faiss_index
│       ├── metadata.json
├── src/                     
│   ├── api_selector.py      # API Selector Agent
│   ├── construct_prompt.py  # Explanation Agent
│   ├── process_docs.py      # PDF processing
│   ├── store_embeddings.py  # Embedding creation and FAISS indexing
│── app.py                   # Chatbot interface
├── .env                     
├── requirements.txt         
├── README.md                
```

## Getting Started

### Prerequisites

- Python 3.10 or higher
- OpenAI API key (for embeddings and explanation generation)

---

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/API-Agentic-Chat.git
   cd API-Agentic-Chat
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  
   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the project root and add your OpenAI API key:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

---

### Usage

1. **Process API Documentation**
   - Place your API documentation PDFs in the `reference_docs/` folder.
   - Run the document processing script to extract chunks:

     ```bash
     python src/process_docs.py
     ```

2. **Generate Embeddings**
   - Create FAISS index and metadata for chunked data:

     ```bash
     python src/store_embeddings.py
     ```

3. **Run the Streamlit App**
   - Start the chatbot interface:

     ```bash
     streamlit run src/streamlit_app.py
     ```

4. **Interact with the Chatbot**
   - Ask questions about API usage in plain language (e.g., "How to book a flight?").
   - Get detailed, step-by-step answers based on relevant API documentation.

---

## Agents

### 1. **API Selector Agent**

**File**: `src/api_selector.py`  
**Purpose**: Retrieves the most relevant API endpoints based on user queries using OpenAI embeddings and FAISS.

**Key Features**:

- Embeds user queries using OpenAI embeddings.
- Searches FAISS index for the most relevant chunks.
- Filters and ranks results by relevance score.

**Output**:

- A list of relevant API chunks, including:
  - Endpoint
  - Description
  - API Body (if available)
  - Relevance Score

---

### 2. **Explanation Agent**

**File**: `src/construct_prompt.py`  
**Purpose**: Generates step-by-step explanations using a Large Language Model (LLM) based on selected API results.

**Key Features**:

- Combines relevant API results into a coherent context.
- Generates clear and concise responses tailored to the user query.
- Ensures no unnecessary repetition or irrelevant details.

**Output**:

- A detailed step-by-step guide using the most relevant APIs.



## License

This project is licensed under the MIT License.