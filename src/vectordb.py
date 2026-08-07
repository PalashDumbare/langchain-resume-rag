from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
from pathlib import Path
from config import CHROMA_DB_DIR


def create_vector_database(chunks):
    print(f"Creating a new vector database in {CHROMA_DB_DIR}.")
    embedding_model = get_embedding_model()
    vector_db = Chroma.from_documents(
        chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR)
    return vector_db


def load_vector_database():
    print(f"Loading existing vector database from {CHROMA_DB_DIR}.")
    embedding_model = get_embedding_model()
    vector_db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function = embedding_model
    )
    return vector_db
