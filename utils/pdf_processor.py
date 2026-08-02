"""
pdf_processor.py
----------------
Handles loading of one or more uploaded PDF files and splitting
their text into overlapping chunks suitable for embedding.
"""

import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


def load_pdfs(uploaded_files) -> List[Document]:
    """
    Accepts a list of Streamlit UploadedFile objects, saves each to a
    temporary file (PyPDFLoader needs a path), loads it, and tags every
    resulting Document with its source filename so answers can cite
    which PDF they came from.
    """
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            all_docs.extend(docs)
        finally:
            os.remove(tmp_path)

    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Splits loaded documents into overlapping chunks. Overlap preserves
    context across chunk boundaries so answers don't lose meaning at
    a split point.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
