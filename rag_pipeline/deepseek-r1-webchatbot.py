"""
Streamlit Website RAG App
Description: Streamlit-based Retrieval-Augmented Generation (RAG) system
using Ollama (deepseek-r1), LangChain, FAISS, and website scraping.
This is a practice implementation for educational purposes. I do not intend to copy
or infringe upon any proprietary code or intellectual property.
"""

import os
import tempfile
import requests
import numpy as np
import streamlit as st
from bs4 import BeautifulSoup

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import BSHTMLLoader
from langchain.memory import ConversationBufferMemory


# -----------------------------
# Configuration
# -----------------------------
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
MODEL_NAME = "deepseek-r1:latest"
TEMPERATURE = 0.4


# -----------------------------
# Streamlit Session State
# -----------------------------
if "qa" not in st.session_state:
    st.session_state.qa = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -----------------------------
# Website Processing
# -----------------------------
def fetch_and_process_website(url: str):
    """Fetch and process website content into text chunks."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        with st.spinner("Fetching website content..."):
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            # Store HTML temporarily
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".html", encoding="utf-8"
            ) as temp_file:
                temp_file.write(response.text)
                temp_file_path = temp_file.name

            try:
                loader = BSHTMLLoader(temp_file_path)
                documents = loader.load()
            except ImportError:
                st.warning(
                    "'lxml' not installed. Falling back to html.parser."
                )
                loader = BSHTMLLoader(
                    temp_file_path, bs_kwargs={"features": "html.parser"}
                )
                documents = loader.load()
            finally:
                os.unlink(temp_file_path)

            text_splitter = CharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            texts = text_splitter.split_documents(documents)

            return texts

    except Exception as e:
        st.error(f"Error processing website: {e}")
        return None


# -----------------------------
# RAG Initialization
# -----------------------------
def initialize_rag_pipeline(texts):
    """Initialize embeddings, vector store, and QA chain."""
    with st.spinner("Initializing RAG pipeline..."):
        llm = ChatOllama(
            model=MODEL_NAME,
            temperature=TEMPERATURE
        )

        embeddings = OllamaEmbeddings(
            model="deepseek-r1:latest"
        )

        vectorstore = FAISS.from_documents(texts, embeddings)

        template = """
Context:
{context}

Question:
{question}

Answer the question concisely using only the given context.
If the context does not contain relevant information, say:
"I don't have enough information to answer that question."

If the question is generic (e.g., "What is an electric vehicle?"),
you may answer it directly.
"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=memory,
            chain_type_kwargs={"prompt": prompt}
        )

        return qa, vectorstore


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.title("🤖 Website RAG Query System")
    st.write(
        "Enter a website URL, process its content, "
        "and ask questions using a RAG-based chatbot."
    )

    # URL Input
    url = st.text_input("Enter website URL")

    if st.button("Process Website") and url:
        texts = fetch_and_process_website(url)

        if texts:
            st.success(
                f"Processed {len(texts)} text chunks successfully."
            )
            st.session_state.qa, st.session_state.vectorstore = (
                initialize_rag_pipeline(texts)
            )
            st.session_state.chat_history = []

    # Query Section
    if st.session_state.qa and st.session_state.vectorstore:
        st.divider()
        st.subheader("Ask Questions")

        query = st.text_input("Enter your question")

        if st.button("Ask") and query:
            with st.spinner("Searching for answer..."):
                relevant_docs = (
                    st.session_state.vectorstore
                    .similarity_search_with_score(query, k=3)
                )

                with st.expander("View relevant chunks"):
                    for idx, (doc, score) in enumerate(
                        relevant_docs, start=1
                    ):
                        st.write(f"Chunk {idx} (Score: {score:.4f})")
                        st.write(doc.page_content)
                        st.divider()

                response = st.session_state.qa.invoke(
                    {"query": query}
                )

                st.session_state.chat_history.append(
                    {
                        "question": query,
                        "answer": response["result"]
                    }
                )

        # Chat History
        if st.session_state.chat_history:
            st.divider()
            st.subheader("Chat History")

            for chat in st.session_state.chat_history:
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer']}")
                st.divider()

    # Sidebar
    with st.sidebar:
        st.subheader("About")
        st.write(
            """
This application is a Retrieval-Augmented Generation (RAG) system that:
1. Scrapes website content
2. Converts it into embeddings
3. Stores them in FAISS
4. Answers user questions using Ollama (deepseek-r1)
"""
        )

        st.subheader("Configuration")
        st.write(f"Model: {MODEL_NAME}")
        st.write(f"Temperature: {TEMPERATURE}")
        st.write(f"Chunk Size: {CHUNK_SIZE}")
        st.write(f"Chunk Overlap: {CHUNK_OVERLAP}")


if __name__ == "__main__":
    main()
