"""
app.py
------
Streamlit front-end for the PDF QA Chatbot.

Run locally:
    streamlit run app.py

Features:
  - Multiple PDF upload & processing
  - Semantic search via FAISS
  - LLM-generated answers (Hugging Face hosted model)
  - Persistent chat history within the session
"""

import streamlit as st

from utils.pdf_processor import load_pdfs, split_documents
from utils.vector_store import get_embeddings, build_vector_store
from utils.qa_chain import build_qa_chain

st.set_page_config(page_title="PDF QA Chatbot", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text) tuples
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------------------------------------------------------------
# Sidebar: configuration + PDF upload
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    hf_token = st.text_input(
        "Hugging Face API Token",
        type="password",
        help="Get a free token at https://huggingface.co/settings/tokens",
    )

    llm_repo = st.text_input(
        "LLM repo id",
        value="openai/gpt-oss-120b:cerebras",
        help="Any Hugging Face Inference Endpoint-compatible chat model",
    )

    st.divider()
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    process_clicked = st.button("Process PDFs", type="primary", use_container_width=True)

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.qa_chain is not None:
            st.session_state.qa_chain.memory.clear()
        st.rerun()

# ---------------------------------------------------------------------
# Process PDFs -> chunks -> FAISS index -> QA chain
# ---------------------------------------------------------------------
if process_clicked:
    if not hf_token:
        st.sidebar.error("Please enter your Hugging Face API token.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        with st.spinner("Reading PDFs and building the semantic index..."):
            documents = load_pdfs(uploaded_files)
            chunks = split_documents(documents)
            embeddings = get_embeddings()
            vectorstore = build_vector_store(chunks, embeddings)
            st.session_state.qa_chain = build_qa_chain(vectorstore, hf_token, llm_repo)
            st.session_state.processed = True
            st.session_state.chat_history = []
        st.sidebar.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} PDF(s).")

# ---------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------
st.title("📄 PDF Question Answering Chatbot")
st.caption("LangChain + Hugging Face + FAISS + Streamlit")

if not st.session_state.processed:
    st.info("Upload PDF(s) and click **Process PDFs** in the sidebar to get started.")
else:
    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)

    user_question = st.chat_input("Ask a question about your PDFs...")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain.invoke({"question": user_question})
                answer = result["answer"]
                sources = sorted(
                    {doc.metadata.get("source", "unknown") for doc in result.get("source_documents", [])}
                )
                if sources:
                    answer += "\n\n**Sources:** " + ", ".join(sources)
                st.markdown(answer)

        st.session_state.chat_history.append(("assistant", answer))
