"""
Project Name: Website Conversational RAG Chatbot

Description:
A conversational Retrieval-Augmented Generation (RAG) chatbot that:
- Scrapes website content
- Splits text into chunks
- Stores embeddings in FAISS
- Uses OpenAI LLM with conversational memory
- Answers user queries based on website context

This is a practice implementation for educational purposes. I do not intend to copy
or infringe upon any proprietary code or intellectual property.
"""

import os
import bs4
import requests

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


# -----------------------------
# Configuration
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Please set it as an environment variable."
    )

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
WEBSITE_URL = "https://www.snapy.ai/"


# -----------------------------
# LLM & Embeddings
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4
)

embeddings = OpenAIEmbeddings()


# -----------------------------
# Website Scraping
# -----------------------------
def fetch_website_content(url: str):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching the website: {e}")
        return None


def scrape_website(url: str):
    content = fetch_website_content(url)
    if not content:
        return []

    soup = bs4.BeautifulSoup(content, "html.parser")
    text_content = soup.get_text(separator="\n", strip=True)
    return [text_content]


# -----------------------------
# Website Processing
# -----------------------------
def process_website(url: str):
    docs = scrape_website(url)

    if not docs:
        print("No content could be extracted from the website.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    splits = text_splitter.split_text(docs[0])

    if not splits:
        print("No valid text chunks were created.")
        return None

    vectorstore = FAISS.from_texts(
        splits,
        embedding=embeddings
    )

    return vectorstore


# -----------------------------
# Vector Store Initialization
# -----------------------------
vectorstore = process_website(WEBSITE_URL)

if not vectorstore:
    print("Failed to create vectorstore. Exiting.")
    exit(1)


# -----------------------------
# Prompt Template
# -----------------------------
prompt_template = """
Use the following pieces of context to answer the human's question.
If you don't know the answer, say that you don't know.

Context:
{context}

Human:
{question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# -----------------------------
# Memory & Chain
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)


# -----------------------------
# Chatbot Function
# -----------------------------
def chatbot(query: str):
    response = qa_chain({"question": query})
    return response["answer"]


# -----------------------------
# CLI Interface
# -----------------------------
if __name__ == "__main__":
    print("Welcome to the Website Chatbot!")
    print(f"This chatbot answers questions about: {WEBSITE_URL}")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            print("Thank you for using the Website Chatbot. Goodbye!")
            break

        response = chatbot(user_input)
        print(f"\nChatbot: {response}\n")
