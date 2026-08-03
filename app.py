from numpy import rint

from src.loader import load_documents
from src.splitter import split_documents


def main():
    data_dir = "data"
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages.")
    print(f"Created {len(chunks)} chunks.")
    print(f"First chunk: {chunks[0].metadata}")


if __name__ == "__main__":
    main()