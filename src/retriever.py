from langchain_core.vectorstores import VectorStoreRetriever

# Given a question, return the most relevant documents.
def get_retriever(vector_db,k=3)  -> VectorStoreRetriever:
    retriever = vector_db.as_retriever(
        search_kwargs={"k": k},
        search_type="similarity",
        )
    return retriever
