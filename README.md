# Intelligent Document Search and Q&A System (RAG)

[English](README.md) | [Türkçe](README_TR.md)

A lightweight, modular Retrieval-Augmented Generation system that analyzes local PDF, TXT, and Markdown files and answers questions using their contents. Chunking and similarity search run locally with Python, NumPy, TF-IDF, and cosine similarity; Gemini is optional for synthesized answers.

## Components

- `main.py`: interactive command-line interface
- `pdf_parser.py`: extracts text while preserving page references
- `retriever.py`: chunks documents and builds the TF-IDF index
- `llm_client.py`: optionally sends retrieved context to Gemini
- `config.example.json`: safe configuration template
- `docs/`: documents to index

## Setup

```bash
pip install pypdf requests numpy
```

Copy `config.example.json` to `config.json` and add a Gemini API key if you want generated answers. Without a key, the system works offline and returns the most relevant passages with source and page information.

## Run

Place documents in `docs/`, then run:

```bash
python main.py
```

Enter a question, use `list` to display indexed sources, or enter `q` to quit.

