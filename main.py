# Created by Raafay Umar for Hobglobin Software - AI Engineer Take-Home Assessment on 5th May 2025

"""
This FastAPI application exposes two endpoints as part of a Retrieval-Augmented Generation (RAG) system:

1. **GET /fine-prints**: Parses and chunks PDF documents from a directory and returns the extracted "fine-print" text segments.
2. **POST /chat**: Accepts a natural language question and returns a document-grounded answer, leveraging LangChain, Chroma vector store, and a local LLM via Ollama.

The core logic relies on:
- `pdf_parser.py` for PDF text and table extraction + chunking using HuggingFace tokenizer.
- `ollama_config.py` to load LLM and embedding models via Ollama.
- `rag_pipeline.py` which defines a RAG pipeline for indexing and querying.
- `test_cli.py` for local CLI testing (not used here).

This file ties all modules into a web API using FastAPI.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pathlib import Path
from pydantic import BaseModel
from typing import List

from pdf_parser import load_all_pdfs
from rag_pipeline import RAGPipeline

app = FastAPI(
    title="Hobglobin RAG API",
    description="FastAPI service for fine-print extraction and chatbot querying",
    version="1.0"
)

# Load PDFs and initialize pipeline on startup
pdf_dir = Path("data/pdfs")
documents = load_all_pdfs(pdf_dir)
rag_pipeline = RAGPipeline()
rag_pipeline.index_documents(documents)

# Pydantic model for input query
class ChatRequest(BaseModel):
    query: str

@app.get("/fine-prints")
def get_fine_prints():
    """
    Returns all extracted fine-print chunks from PDF documents.
    """
    chunks = [doc.page_content for doc in documents]
    return {"fine_prints": chunks, "count": len(chunks)}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Answers a question using the RAG system grounded on PDF content.
    """
    result = rag_pipeline.query(request.query)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "excerpt": doc.page_content[:300]
        } for doc in result["sources"]
    ]

    return JSONResponse({
        "question": request.query,
        "answer": result["answer"],
        "sources": sources
    })

@app.get("/")
def root():
    return {"message": "Hobglobin RAG system is up. Use /fine-prints or /chat."}
