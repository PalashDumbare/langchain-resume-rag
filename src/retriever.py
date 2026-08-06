from langchain_core.vectorstores import VectorStoreRetriever
from config import TOP_K
from config import (SEARCH_TYPE,TOP_K,SIMILARITY_THRESHOLD)


# Given a question, return the most relevant documents.
def get_retriever(vector_db,k=TOP_K)  -> VectorStoreRetriever:

    if SEARCH_TYPE == "similarity":
        return vector_db.as_retriever(
            search_kwargs={"k": k},
            search_type="similarity",
            )
    elif SEARCH_TYPE == "similarity_score_threshold":
        return vector_db.as_retriever(
            search_kwargs={"k": k, "score_threshold": SIMILARITY_THRESHOLD},
            search_type="similarity_score_threshold",
        )
    elif SEARCH_TYPE == "mmr":
        return vector_db.as_retriever(
            search_kwargs={"k": k},
            search_type="mmr",
        )
    
    raise ValueError(f"Invalid SEARCH_TYPE: {SEARCH_TYPE}. Must be one of 'similarity', 'similarity_score_threshold', or 'mmr'.")
