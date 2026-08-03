# langchain-resume-rag

Small practice project demonstrating retrieval-augmented generation (RAG) using LangChain-style components.

## Project structure

- `src/` — core modules
	- `loader.py` — document loader(s)
	- `splitter.py` — document splitting logic
	- `embeddings.py` — embedding helpers (if present)
	- `vectordb.py` — vector database wrappers
	- `retriever.py` — retrieval layer
	- `prompt.py` — prompt templates
	- `chain.py` — orchestrating chains/workflows
	- `config.py` — configuration
- `data/` — sample documents to index
- `chroma_db/` — (optional) local vector DB files
- `app.py` — example runner that loads documents and prints chunk stats
- `requirements.txt` — Python dependencies
- `.env` — environment variables (not checked in)

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

3. Create a `.env` file if your project requires API keys or configuration. See `src/config.py` for expected variables.

4. Run the example:

```bash
python app.py
```

This will load documents from the `data/` folder, split them into chunks, and print counts and a sample chunk.

## Notes

- If you plan to rebuild the vector store, remove or back up `chroma_db/` first.
- Inspect `src/loader.py` and `src/splitter.py` to customize document ingestion and chunking behavior.
