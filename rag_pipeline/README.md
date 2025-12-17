# RAG Pipeline — README

**Overview**

This repository contains small Retrieval-Augmented Generation (RAG) pipelines and example chatbots that:
- scrape website content or load CSVs
# RAG Pipeline — Simple Overview

Welcome — this repo contains a few small demos and utilities that show how to build a Retrieval-Augmented Generation (RAG) pipeline using website scraping, CSV data, embeddings, FAISS, and small LLMs.

Here are the main scripts and what they do:

- `website_chatbot.py` — scrape a website, create embeddings, build an in-memory FAISS index, and ask questions from the command line.
- `memory_based.py` — simple conversational RAG using a fixed website URL and conversational memory.
- `csv_bot.py` — load a CSV file, create embeddings and a FAISS index, then answer questions about the CSV.
- `deepseek-r1-webchatbot.py` — Streamlit-based app using a local model adapter (Ollama) and a web UI.
- `generate_dataset_csv.py` — generate a small synthetic CSV dataset (useful for testing).

---

## Quick start (Windows PowerShell)

1. Create & activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set your OpenAI API key (don't commit it):

Create a `.env` file (or copy the included `.env.example` to `.env`) and fill in your values. Example `.env` (from `.env.example`):

```
OPENAI_API_KEY=sk-<YOUR_KEY_HERE>
AZURE_LLM_DEPLOYMENT=<your_azure_llm_deployment>
AZURE_EMBEDDING_DEPLOYMENT=<your_azure_embedding_deployment>
DEBUG=false
```

> Note: Do not commit your real API keys. This repo already ignores `.env`. Keep `.env.example` in the repo as a safe template (it contains only placeholders).

---

## How to run

- Run a website RAG session (console):

```powershell
python website_chatbot.py
```

- Run the CSV demo:

```powershell
python csv_bot.py
```

- Run the conversational example:

```powershell
python memory_based.py
```

- Run the Streamlit web UI:

```powershell
streamlit run deepseek-r1-webchatbot.py
```

- Generate a sample dataset:

```powershell
python generate_dataset_csv.py
```

---

## Short notes & tips

- If scraping returns no content, try another URL or install `lxml` / use Playwright for JS-heavy sites.
- The scripts expect `OPENAI_API_KEY` to be set. For local development you can use `python-dotenv` and a `.env` file, but never commit real keys.
- The code prints a few debug snippets (examples and embeddings) which are useful during development — be careful if you run this with sensitive data.
