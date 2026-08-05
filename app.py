
from src.loader import load_documents
from src.splitter import split_documents
from src.vectordb import get_vector_database
from src.retriever import get_retriever
from src.chain import get_rag_chain
from config import DATA_DIR

def main():
    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages.")
    print(f"Created {len(chunks)} chunks.")

    vector_db = get_vector_database(chunks)
    print("Vector database is ready.")

    retriever = get_retriever(vector_db)
    chain = get_rag_chain(retriever)

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        response = chain.invoke(
            {"question": query},
            config={
                # Only 1 conversation session is supported.
                "configurable": {
                    "session_id": "default_session"
                }
            },
        )

        print("\nAnswer:")
        print(response)


if __name__ == "__main__":
    main()
