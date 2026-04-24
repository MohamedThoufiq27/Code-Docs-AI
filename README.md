# CodeDocsAI — Intelligent Code Documentation Assistant

## Project Overview
A production-grade RAG system that indexes code documentation and Stack Overflow snippets, enabling developers to ask natural language questions and receive context-aware code solutions with cost optimization and monitoring.

## Setup Instructions

### Environment
1. Copy `.env.example` to `.env`
2. Populate the keys:
   - `OPENROUTER_API_KEY`: from https://openrouter.ai
   - `OPENAI_API_KEY`: from https://platform.openai.com
   - `PINECONE_API_KEY`: from https://www.pinecone.io
   - `PINECONE_ENV` and `PINECONE_INDEX_NAME` defaults apply.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Usage
1. Ingest Documents:
```bash
python scripts/ingest_docs.py
```
2. Start the Backend:
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
3. Start the Frontend (Separate terminal):
```bash
streamlit run backend/streamlit_app.py
```

## Features
- **RAG Architecture** with LangChain & Pinecone
- **Multi-step Retrieval** and Context Augmentation
- **Cost Optimization** logging
- **Monitoring & Observability**
- **Streamlit UI**
