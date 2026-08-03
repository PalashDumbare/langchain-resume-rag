from langchain_chroma import Chroma
from src.embeddings import get_embedding_model

PERSIST_DIRECTORY = "resume_db"


def create_vector_database(chunks):
    embedding_model = get_embedding_model()
    vector_db = Chroma.from_documents(
        chunks, 
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY)
    return vector_db

