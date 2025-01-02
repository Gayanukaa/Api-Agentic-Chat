# Api-Agentic-Chat

```
├── .env                     # Environment variables for API keys, etc.
├── main.py                  # Entry point for the chatbot application
├── requirements.txt         # Python dependencies
├── reference_docs/          # Folder containing the dotRez API PDFs
│   ├── api_doc1.pdf
│   ├── api_doc2.pdf
│   ├── ...
├── src/                     # Source code
│   ├── __init__.py
│   ├── embeddings.py        # Code for embedding generation and vector storage
│   ├── retrieval.py         # Logic for searching relevant API documentation
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── api_selector.py  # Agent to select the relevant API
│   │   ├── simplifier.py    # Agent to summarize and simplify responses
├── data/                    # Processed data storage
│   ├── embeddings/          # Store generated embeddings
│   │   ├── vectors.pickle
│   ├── chunks/              # Processed chunks of text from PDFs
│   │   ├── chunked_data.json
├── tests/                   # Test cases for the application
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   ├── test_agents.py
└── README.md                # Project description and usage instructions
```