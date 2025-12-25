<div align="center">

  <img src="IntelliSearch logo.png" alt="IntelliSearch Logo" width="400" height="auto" />  
  <h1>IntelliSearch: Hybrid RAG Assistant</h1>
  
  <p>
    <b>An advanced research assistant combining Document Retrieval (RAG) with Live Web Search.</b>
  </p>

  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  </a>
  <a href="https://python.langchain.com/">
    <img src="https://img.shields.io/badge/LangChain-v0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  </a>
  <a href="https://groq.com/">
    <img src="https://img.shields.io/badge/Groq-Llama_3-F55036?style=for-the-badge&logo=fastapi&logoColor=white" alt="Groq" />
  </a>
  <a href="https://tavily.com/">
    <img src="https://img.shields.io/badge/Tavily-Web_Search-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Tavily" />
  </a>
</div>

<br />


## Overview

**IntelliSearch** is a dual-engine research tool that solves the "static knowledge" problem of traditional RAG systems. It acts as a unified interface where you can upload your own private documents (PDF/TXT) *and* simultaneously access real-time information from the web.

Powered by **Groq's LPU inference engine** running Llama 3, it delivers near-instant answers. The system uses a **Hybrid Retrieval** strategy: it compresses and re-ranks document chunks for precision while falling back to Tavily's AI-optimized web search for current events.
<br>

## Key Features

* **Blazing Fast Inference:** Utilizes **Groq API** (Llama-3-70b) for sub-second response times.
* **Hybrid RAG Pipeline:**
    * **Vector Search:** FAISS index with HuggingFace embeddings (`all-MiniLM-L6-v2`).
    * **Advanced Retrieval:** Implements `MultiQueryRetriever` to generate different perspectives of your question.
    * **Contextual Compression:** Uses `LLMChainExtractor` to filter out irrelevant text from retrieved documents.
* **Live Web Access:** Integrated **Tavily Search** to fetch real-time data when the document lacks answers.
* **Transparent Reasoning:** The UI displays exactly which document chunks or web sources were used to generate the answer.
<br>

## Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **LLM Inference** | Groq Cloud | Serves Llama-3-70b-Versatile |
| **Orchestration** | LangChain  | Manages the RAG chain and tool usage |
| **Embeddings** | HuggingFace | `sentence-transformers` for local vectorization |
| **Vector DB** | FAISS | Efficient similarity search for document chunks |
| **Web Search** | Tavily AI | Search engine optimized for LLM agents |
| **Frontend** | Streamlit | Modern, responsive chat interface |
<br>

## Prerequisites

Before running the project, you need API keys for the inference and search engines:

1.  **Groq API Key:** Get it free at [console.groq.com](https://console.groq.com/).
2.  **Tavily API Key:** Get it free at [tavily.com](https://tavily.com/).
<br>

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Darahas1/IntelliSearch-RAG.git
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Secrets
```
# .env file content (below)

GROQ_API_KEY=gsk_xxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxx
```

### 5. Run the Application
```bash
streamlit run ui.py
```
<br>

### Project Structure
```
intellisearch-rag/
├── main_logic.py          # Core RAG pipeline (LangChain logic)
├── ui.py                  # Streamlit frontend application
├── .env                   # API Keys (Excluded from Git)
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
```
<br>

### Troubleshooting
<details> <summary><b>❌ ValueError: GROQ_API_KEY not found</b></summary>


Ensure you have created the <code>.env</code> file in the same directory as <code>ui.py</code> and that it contains your valid API key. </details>

<details> <summary><b>❌ ImportError: cannot import name 'ContextualCompressionRetriever'</b></summary>


This project uses <code>langchain-classic</code> for advanced retrieval features. Ensure you have installed it via <code>pip install langchain-classic</code>. </details>

<details> <summary><b>❌ Error processing file (PDF)</b></summary>


Ensure the PDF is not password-protected and contains selectable text (not just scanned images). </details>
<br>

### License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Darahas1/IntelliSearch-RAG/blob/main/LICENSE) file for details.

<div align="center"> <sub>Built with 💙 by Sai Darahas</sub> </div>