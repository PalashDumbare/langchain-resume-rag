from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def get_embedding_model(model_name=EMBEDDING_MODEL):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings