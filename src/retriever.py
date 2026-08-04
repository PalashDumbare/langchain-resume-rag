from langchain_core.vectorstores import VectorStoreRetriever
from config import TOP_K


# Given a question, return the most relevant documents.
def get_retriever(vector_db,k=TOP_K)  -> VectorStoreRetriever:
    retriever = vector_db.as_retriever(
        search_kwargs={"k": k},
        search_type="similarity",
        )
    return retriever
