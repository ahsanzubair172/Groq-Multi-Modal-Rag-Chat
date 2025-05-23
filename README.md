# 🤖 Groq Multi-Modal Chat Application

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Built with Groq](https://img.shields.io/badge/built%20with-Groq-orange)](https://groq.ai/)
[![MIT License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

> A **versatile, intelligent conversational AI application** — powered by Groq API — supporting text-based conversations, document interactions, memory functionality for context management, and multiple implementation architectures.

* * *

## ✨ Key Features

- ✅ Text-based conversational AI with context awareness
- ✅ Document processing and question answering capabilities
- ✅ Memory functionality to maintain conversation context
- ✅ Support for multiple document formats including PDF
- ✅ FastAPI integration for building scalable chatbot APIs
- ✅ Streamlit interface for interactive document-based conversations

* * *
---
## 🏗️ Architecture Overview

Groq Chat Application
│
├── Core Components
│   ├── LLM (Groq API)
│   ├── Embedding Model (HuggingFace)
│   ├── Document Loaders (PyMuPDF, PyPDF2)
│   ├── Indexes (VectorStore, Tree, KeywordTable)
│   └── Memory Buffer (ChatMemoryBuffer)
│
├── Implementations
│   ├── Basic Text Chat (groqchat.py)
│   ├── Document-Aware Chat (mupdf.py, pypdf.py, rag.py)
│   ├── FastAPI Integration (pbfast.py, ragchat.py)
│   ├── Streamlit Interface (pcst2.py)
│   └── Advanced Architectures (graph.py, graph2.py)
│
└── Utilities
├── Node Parsers
├── Text Splitters
└── Configuration Management




## 🎮 Getting Started

### 📦 Requirements

- Python 3.8+
- Groq API key
- Various Python packages (listed in `requirements.txt`)

### 🔧 Installation
1. Clone the repository
git clone https://github.com/your-username/groq-multi-modal-chat.git
cd groq-multi-modal-chat
2. Install dependencies
pip install -r requirements.txt
3. Obtain a Groq API key from Groq Cloud Console
4. Set up your API key in API.py
echo "MyApi = 'your_api_key'" > API.py


### ▶️ Usage

#### Basic Text Chat
python groqchat.py


#### Document-Aware Chat
Using PyMuPDF
python mupdf.py
Using PyPDF2
python pypdf.py
Using RAG (Retrieval-Augmented Generation)
python rag.py


#### FastAPI Server
Using pbfast.py
uvicorn pbfast:app --reload --host 0.0.0.0 --port 4000
Using ragchat.py
uvicorn ragchat:app --reload --host 0.0.0.0 --port 8000


#### Streamlit Interface
streamlit run pcst2.py


* * *

## 📚 User Guide

### Basic Text Chat

1. Run the basic text chat script:
python groqchat.py


2. Type your queries in the console.
3. Type 'exit', 'quit', 'close', or 'bye' to end the conversation.

### Document-Aware Chat

1. Place your PDF documents in the specified data folder (`D:\Ahsan\Office\Data`).
2. Run the desired document chat script:
python mupdf.py
python pypdf.py
python rag.py


3. Type your queries related to the documents.
4. Type 'exit', 'quit', 'close', or 'bye' to end the conversation.

### FastAPI Integration

1. Start the FastAPI server:
uvicorn pbfast:app --reload --host 0.0.0.0 --port 4000


2. Send POST requests to `http://localhost:4000/ask` with a JSON body containing your query.

Example JSON body:

json
{
  "query": "What is the purpose of this application?"
}
Streamlit Interface
bash
streamlit run pcst2.py
Use the web interface for interaction

Upload PDF documents for document-aware Q&A

Navigate via: Home, Chat, Documents, Chat History

🧪 Developer Notes
Document loading via PyMuPDF, PyPDF2

Text chunking for indexing

VectorStore indexing

Conversational memory with token limits

System prompt configuration

Logging and error handling

Cross-platform compatibility

🩺 Troubleshooting
Issue	Solution
❌ Invalid API key	Check and update API.py
❌ Documents not loading	Ensure file paths are correct
❌ No response from bot	Verify API key, rate limits, internet
❌ App not starting	Check dependency installation

📜 License
MIT License — See LICENSE for full details.

🔒 Disclaimer
This project is for educational and personal use only.
Please respect the terms of service of the API providers.

🙌 Acknowledgements
Groq API

Llama Index

PyMuPDF
....
....
....
PyPDF2

FastAPI

Streamlit

HuggingFace Embeddings

Crafted by MYSELF
