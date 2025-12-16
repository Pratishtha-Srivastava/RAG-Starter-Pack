"""
CSV RAG Pipeline

Description: Retrieval-Augmented Generation (RAG) pipeline for CSV data
using LangChain, OpenAI embeddings, FAISS, and RetrievalQA.
This is a practice implementation for educational purposes. I do not intend to copy
or infringe upon any proprietary code or intellectual property.
"""

import os
import numpy as np

from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# -----------------------------
# Configuration
# -----------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS = 150
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.4


# -----------------------------
# Environment Variable Check
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Please set it as an environment variable."
    )


# -----------------------------
# CSV Loading
# -----------------------------
def load_csv_data(file_path: str):
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from CSV")
    return documents


# -----------------------------
# Embedding Creation
# -----------------------------
def create_embeddings(documents):
    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks")

    embeddings = OpenAIEmbeddings()

    # Debug: Sample embedding
    if texts:
        sample_text = texts[0].page_content
        sample_embedding = embeddings.embed_query(sample_text)

        print("\nSample Text:")
        print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)

        print("\nSample Embedding (first 10 dimensions):")
        print(np.array(sample_embedding[:10]))

        print(f"Embedding vector size: {len(sample_embedding)}")

    return texts, embeddings


# -----------------------------
# Vector Store
# -----------------------------
def create_vectorstore(texts, embeddings):
    return FAISS.from_documents(texts, embeddings)


# -----------------------------
# QA Chain
# -----------------------------
def setup_qa_chain(vectorstore):
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    template = """
Use the following context to answer the question.
If you do not know the answer, say you do not know.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


# -----------------------------
# Query Processing
# -----------------------------
def process_query(query: str, qa_chain):
    result = qa_chain({"query": query})
    return result["result"], result["source_documents"]


# -----------------------------
# Main Application
# -----------------------------
def main():
    print("Welcome to the CSV RAG Pipeline")

    csv_path = input("\nEnter path to CSV file: ").strip()

    print("\nLoading CSV data...")
    documents = load_csv_data(csv_path)

    print("\nCreating embeddings...")
    texts, embeddings = create_embeddings(documents)

    print("\nCreating vector store...")
    vectorstore = create_vectorstore(texts, embeddings)

    print("\nSetting up QA chain...")
    qa_chain = setup_qa_chain(vectorstore)

    print("\nRAG Pipeline ready!")
    print("Type 'quit' to exit.")

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == "quit":
            print("Goodbye!")
            break

        answer, sources = process_query(query, qa_chain)

        print(f"\nAnswer:\n{answer}")
        print("\nSources:")
        for source in sources:
            print(f"- {source.page_content[:100]}...")


if __name__ == "__main__":
    main()
