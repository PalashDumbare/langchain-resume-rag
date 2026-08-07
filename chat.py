
from src.vectordb import load_vector_database
from src.retriever import get_retriever
from src.chain import get_rag_chain

def main():
    vector_db = load_vector_database()
    print("Vector database loaded.")

    retriever = get_retriever(vector_db)
    chain = get_rag_chain(retriever)

    while True:
        query = input("\nAsk a question (type 'exit' to quit): ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        for chunk in chain.stream(
            {"question": query},
            config={
                "configurable": {
                    "session_id": "default_session"
                }
            },
        ): print(chunk, end="", flush=True)

        print()


if __name__ == "__main__":
    main()
