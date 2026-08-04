
from src.loader import load_documents
from src.splitter import split_documents
from src.vectordb import get_vector_database
from src.retriever import get_retriever
from src.chain import get_rag_chain

def main():
    data_dir = "data"
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages.")
    print(f"Created {len(chunks)} chunks.")

    vector_db = get_vector_database(chunks)
    print("Vector database is ready.")

    chain = get_rag_chain(get_retriever(vector_db))

    query = "List all skills of Palash?"
    response = chain.invoke(query)

    print("\nAnswer\n")
    print(response)


if __name__ == "__main__":
    main()
