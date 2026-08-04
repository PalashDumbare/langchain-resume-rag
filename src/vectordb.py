from langchain_chroma import Chroma
from src.embeddings import get_embedding_model
from pathlib import Path

PERSIST_DIRECTORY = "resume_db"


def get_vector_database(chunks):

    # Check if the vector database already exists
    if Path(PERSIST_DIRECTORY).exists():
        print(f"Loading existing vector database from {PERSIST_DIRECTORY}.")
        embedding_model = get_embedding_model()
        vector_db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function = embedding_model
        )
        return vector_db
    
    # If the vector database does not exist, create a new one
    print(f"Creating a new vector database in {PERSIST_DIRECTORY}.")
    embedding_model = get_embedding_model()
    vector_db = Chroma.from_documents(
        chunks, 
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY)
    return vector_db

