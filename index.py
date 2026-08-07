
from src.loader import load_documents
from src.splitter import split_documents
from src.vectordb import create_vector_database
from config import DATA_DIR

def main():
    documents = load_documents(DATA_DIR)
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages.")
    print(f"Created {len(chunks)} chunks.")

    vector_db = create_vector_database(chunks)
    print("Vector database is ready.")


if __name__ == "__main__":
    main()
