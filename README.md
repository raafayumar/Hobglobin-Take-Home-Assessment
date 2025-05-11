# Hobglobin Take-Home Assessment – AI Engineer

**Created by Raafay Umar | May 5, 2025**

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, FastAPI, ChromaDB, and Ollama. It allows users to ask questions about PDF-based project documents and get responses powered by a local LLM.

---

## Problem Statement

Build a system that:
- Extracts fine-print (key proposal-relevant content) from PDF documents.
- Enables users to ask questions about those documents.
- Returns grounded answers using an LLM.
- Exposes two endpoints:
  - `GET /fine-prints`: For retrieving extracted document chunks.
  - `POST /chat`: For answering user questions based on document content.

---

## Features

- Extracts fine-print from project PDFs (text + tables).
- Token-aware chunking using HuggingFace tokenizer.
- Embedding and indexing using `nomic-embed-text` + ChromaDB.
- Local LLM-powered Q&A via `llama3.1` hosted in Ollama.
- FastAPI endpoints to query and interact with the data.
- Evaluation script to test model performance on real questions.

---

## Project Structure

```
.
├── main.py                # FastAPI app with two endpoints
├── pdf_parser.py          # PDF parsing and tokenizer-based chunking
├── ollama_config.py       # Loads LLM and embedding model
├── rag_pipeline.py        # Core LangChain RAG logic
├── test_cli.py            # CLI interface for manual testing
├── test_questions.py      # Evaluates questions and logs to chat_response.txt
├── data/
│   └── pdfs/              # Folder containing project PDF files
├── rag_db/                # ChromaDB persistence directory
├── chat_response.txt      # Output file containing answers for evaluation
├── requirements.txt       # Required dependencies
└── README.md              # You’re reading it
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-private-repo-url>
cd <repo-folder>
```

### 2. Install Dependencies

We recommend using a virtual environment:

```bash
pip install -r requirements.txt
```

Make sure [Ollama](https://ollama.com/) is installed and running with the required models:

```bash
ollama run llama3.1
ollama run nomic-embed-text
```

---

## Running the Application

### Start the FastAPI server

```bash
uvicorn main:app --reload
```

#### Sample `POST /chat` payload:

```json
{
  "query": "List all mandatory documents bidders must submit."
}
```

---

### Run the Evaluation Script

```bash
python test_questions.py
```

This script will send predefined questions to the API and save results to `chat_response.txt`.

---

## File Responsibilities

| File                 | Purpose |
|----------------------|---------|
| `main.py`            | Exposes FastAPI endpoints `/fine-prints` and `/chat` |
| `pdf_parser.py`      | Extracts and chunks text/tables from PDF documents |
| `ollama_config.py`   | Loads LLaMA 3.1 and embedding model from Ollama |
| `rag_pipeline.py`    | Sets up LangChain's ConversationalRetrievalChain |
| `test_cli.py`        | CLI app to manually test the system |
| `test_questions.py`  | Sends questions to API and logs results to file |
| `chat_response.txt`  | Output file for all API-generated answers |
| `requirements.txt`   | List of required Python packages |

---

## Improvements

- Improve PDF parsing by using layout-aware models like `pdfplumber` or `deepdoctection`.
- Add support for detecting and chunking tables, forms, and image captions separately.
- Enhance chunking strategies by using visual layout heuristics and multi-modal embedding models.
- Allow real-time PDF upload via API for dynamic indexing.
- Add UI for chatbot and fine-print visualization.
