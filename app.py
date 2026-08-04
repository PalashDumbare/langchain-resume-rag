from src.loader import load_documents
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vectordb import get_vector_database
from src.retriever import get_retriever


def main():
    data_dir = "data"
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages.")
    print(f"Created {len(chunks)} chunks.")

    vector_db = get_vector_database(chunks)
    print("Vector database created successfully.")

    query = "List all skills of Palash?"
    retriever = get_retriever(vector_db)
    results = retriever.invoke(query)
    print(f"Found {len(results)} relevant documents for the query: '{query}'")
    for i, doc in enumerate(results, start=1):
        print("=" * 80)
        print(f"Result {i}")
        print(doc.metadata)
        print("-" * 80)
        print(doc.page_content)


if __name__ == "__main__":
    main()
