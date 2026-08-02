"""
vector_store.py
----------------
Builds and persists a FAISS vector index over document chunks using
a Hugging Face sentence-transformer embedding model.
"""

import os
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "faiss_index"


def get_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vector_store(
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    persist: bool = True,
) -> FAISS:
    """Embeds chunks and builds a searchable FAISS index in memory."""
    vectorstore = FAISS.from_documents(chunks, embeddings)
    if persist:
        os.makedirs(INDEX_DIR, exist_ok=True)
        vectorstore.save_local(INDEX_DIR)
    return vectorstore


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Loads a previously saved FAISS index from disk, if one exists."""
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError("No saved FAISS index found. Process PDFs first.")
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )


def index_exists() -> bool:
    return os.path.exists(INDEX_DIR)
