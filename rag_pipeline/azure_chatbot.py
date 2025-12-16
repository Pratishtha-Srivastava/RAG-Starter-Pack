"""
Azure Chatbot
Description: Retrieval-Augmented Generation (RAG) pipeline using LangChain,
Azure OpenAI, FAISS, and website scraping.
This is a practice implementation for educational purposes. I do not intend to copy
or infringe upon any proprietary code or intellectual property.
"""

import os
import requests
import numpy as np
import tempfile
from bs4 import BeautifulSoup

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import BSHTMLLoader


# -----------------------------
# Configuration
# -----------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MAX_TOKENS = 15000
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.4

# -----------------------------
# Environment Variables
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AZURE_LLM_DEPLOYMENT = os.environ.get("AZURE_LLM_DEPLOYMENT")
AZURE_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Please set it as an environment variable."
    )

if not AZURE_LLM_DEPLOYMENT or not AZURE_EMBEDDING_DEPLOYMENT:
    raise EnvironmentError(
        "Azure deployment names must be set as environment variables."
    )


# -----------------------------
# Website Utilities
# -----------------------------
def fetch_html(url: str) -> str | None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None


def process_website(url: str):
    html_content = fetch_html(url)
    if not html_content:
        raise ValueError("No content could be fetched from the website.")

    # Store HTML temporarily
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".html", encoding="utf-8"
    ) as temp_file:
        temp_file.write(html_content)
        temp_file_path = temp_file.name

    try:
        loader = BSHTMLLoader(temp_file_path)
        documents = loader.load()
    except ImportError:
        print("'lxml' not found. Falling back to html.parser.")
        loader = BSHTMLLoader(
            temp_file_path, bs_kwargs={"features": "html.parser"}
        )
        documents = loader.load()
    finally:
        os.unlink(temp_file_path)

    print(f"\nDocuments loaded: {len(documents)}")

    text_splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)

    print(f"Text chunks created: {len(texts)}")
    return texts


# -----------------------------
# Debug Utilities
# -----------------------------
def print_sample_embeddings(texts, embeddings):
    if not texts:
        print("No text available for embedding sample.")
        return

    sample_text = texts[0].page_content
    sample_embedding = embeddings.embed_query(sample_text)

    print("\nSample Text:")
    print(sample_text[:200] + "..." if len(sample_text) > 200 else sample_text)

    print("\nSample Embedding (first 10 dimensions):")
    print(np.array(sample_embedding[:10]))

    print(f"Embedding vector size: {len(sample_embedding)}")


# -----------------------------
# LLM & Prompt
# -----------------------------
llm = AzureOpenAI(
    deployment_name=AZURE_LLM_DEPLOYMENT,
    model_name=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS
)

PROMPT_TEMPLATE = """
Context:
{context}

Question:
{question}

Answer the question concisely using only the given context.
If the context does not contain enough information, say:
"I don't have enough information to answer that question."

If the question is generic (e.g., "What is an electric vehicle?"),
you may answer it directly.
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


# -----------------------------
# RAG Pipeline
# -----------------------------
def rag_pipeline(query: str, qa_chain, vectorstore):
    relevant_docs = vectorstore.similarity_search_with_score(query, k=3)

    print("\nTop 3 Relevant Chunks:")
    context = ""

    for idx, (doc, score) in enumerate(relevant_docs, start=1):
        print(f"{idx}. Score: {score:.4f}")
        print(doc.page_content[:200] + "...\n")
        context += doc.page_content + "\n\n"

    full_prompt = PROMPT.format(context=context, question=query)

    print("\nPrompt Sent to Model:")
    print(full_prompt)
    print("=" * 60)

    response = qa_chain.invoke({"query": query})
    return response["result"]


# -----------------------------
# Main Application
# -----------------------------
if __name__ == "__main__":
    print("Welcome to the Website RAG Chatbot")

    while True:
        url = input("\nEnter website URL (or 'quit'): ").strip()
        if url.lower() == "quit":
            print("Goodbye!")
            break

        try:
            print("\nProcessing website...")
            texts = process_website(url)

            if not texts:
                print("No usable content found.")
                continue

            embeddings = AzureOpenAIEmbeddings(
                deployment=AZURE_EMBEDDING_DEPLOYMENT,
                model="text-embedding-ada-002"
            )

            print_sample_embeddings(texts, embeddings)

            vectorstore = FAISS.from_documents(texts, embeddings)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                chain_type_kwargs={"prompt": PROMPT}
            )

            print("\nRAG initialized. Ask questions!")
            print("Type 'new' for a new website or 'quit' to exit.")

            while True:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    exit()
                if query.lower() == "new":
                    break

                answer = rag_pipeline(query, qa_chain, vectorstore)
                print(f"\nAnswer:\n{answer}")

        except Exception as e:
            print(f"Error: {e}")
            print("Try a different URL or check configuration.")
