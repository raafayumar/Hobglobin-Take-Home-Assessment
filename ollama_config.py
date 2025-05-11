# Created by Raafay Umar for Hobglobin Software - AI Engineer Take-Home Assessment on 5th May 2025

"""
This module sets up the local LLM and embedding models using Ollama.

- `get_llm_model()`: Returns a ChatOllama instance for LLaMA 3.1 with configurable decoding params.
- `get_embedding_model()`: Returns the `nomic-embed-text` embedding model for vectorization.

Used across the RAG pipeline for both retrieval and response generation.
"""

from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama

def get_llm_model(
    model_name: str = "llama3.1",
    temperature: float = 0.3,
    top_p: float = 0.9,
    top_k: int = 50
    
) -> ChatOllama:
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )


def get_embedding_model(model_name: str = "nomic-embed-text") -> OllamaEmbeddings:
    """Returns the embedding model hosted locally via Ollama."""
    return OllamaEmbeddings(model=model_name)