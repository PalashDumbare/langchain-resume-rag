from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
from pathlib import Path
from config import CHROMA_DB_DIR


def get_vector_database(chunks):

    # Check if the vector database already exists
    if Path(CHROMA_DB_DIR).exists():
        print(f"Loading existing vector database from {CHROMA_DB_DIR}.")
        embedding_model = get_embedding_model()
        vector_db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function = embedding_model
        )
        return vector_db
    
    # If the vector database does not exist, create a new one
    print(f"Creating a new vector database in {CHROMA_DB_DIR}.")
    embedding_model = get_embedding_model()
    vector_db = Chroma.from_documents(
        chunks, 
        embedding=embedding_model,
        persist_directory=CHROMA_DB_DIR)
    return vector_db

