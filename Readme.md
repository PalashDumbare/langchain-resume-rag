# langchain-resume-rag

Small practice project demonstrating retrieval-augmented generation (RAG) using LangChain-style components.

## Project structure

- `index.py` — entry point to build the vector database: loads PDFs from `data/`, splits them into chunks, and creates the Chroma DB.
- `chat.py` — entry point for the chatbot: loads the existing vector database and starts a Q&A loop.
- `src/` — core modules
	- `loader.py` — document loader(s)
	- `splitter.py` — document splitting logic
	- `embeddings.py` — embedding helpers
	- `vectordb.py` — vector database creation and loading (`create_vector_database`, `load_vector_database`)
	- `retriever.py` — retrieval layer
	- `prompt.py` — prompt templates
	- `chain.py` — orchestrating chains/workflows
	- `history.py` — chat history helpers
- `config.py` — configuration (paths, models, chunking, retrieval settings)
- `data/` — sample documents to index
- `resume_db/` — local Chroma vector DB files (created by `index.py`)
- `requirements.txt` — Python dependencies

## Quickstart

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Build the vector database:

```bash
python index.py
```

This loads the PDFs from `data/`, splits them into chunks, and creates the Chroma vector database in `resume_db/`.

4. Start the chatbot:

```bash
python chat.py
```

This loads the existing vector database and starts an interactive session. Type `exit` to quit.

## Notes

- If you want to rebuild the vector store from scratch, remove or back up `resume_db/` before running `index.py` again.
- Inspect `src/loader.py`, `src/splitter.py`, and `config.py` to customize document ingestion, chunking, and retrieval behavior.
- This project requires Ollama with `llama3.2:3b` installed locally (see `LLM_MODEL` in `config.py`).
