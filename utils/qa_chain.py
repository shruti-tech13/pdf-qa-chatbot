"""
qa_chain.py
-----------
Wires the FAISS retriever to a Hugging Face-hosted chat model (via
Hugging Face's OpenAI-compatible router) through a
ConversationalRetrievalChain, giving the chatbot memory of prior
turns (bonus feature: chat history).
"""

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

DEFAULT_LLM_REPO = "meta-llama/Llama-3.2-3B-Instruct"

QA_PROMPT = PromptTemplate(
    template="""You are a helpful assistant answering questions about the
user's uploaded PDF documents. Use ONLY the context below to answer.
If the answer isn't in the context, say you don't know instead of
guessing. Cite the source filename when relevant.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


def build_llm(hf_api_token: str, repo_id: str = DEFAULT_LLM_REPO, temperature: float = 0.2):
    return ChatOpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_api_token,
        model=repo_id,
        temperature=temperature,
        max_tokens=512,
    )


def build_qa_chain(vectorstore, hf_api_token: str, repo_id: str = DEFAULT_LLM_REPO):
    """
    Returns a ConversationalRetrievalChain backed by:
      - FAISS similarity retriever (semantic search, top-4 chunks)
      - Hugging Face-hosted chat model (via router)
      - ConversationBufferMemory (multi-turn chat history)
    """
    llm = build_llm(hf_api_token, repo_id)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )
    return chain
