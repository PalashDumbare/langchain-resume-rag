from numpy import rint

from src.loader import load_documents
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vectordb import create_vector_database


def main():
    data_dir = "data"
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    print(f"Loaded {len(documents)} pages.")
    print(f"Created {len(chunks)} chunks.")
    embedding_model = get_embedding_model()
    vector = embedding_model.embed_query(chunks[0].page_content)
    print(f"Embedding dimensions: {len(vector)}")
    print(vector[:10])
    vector_db = create_vector_database(chunks)
    print("Vector database created successfully.")


if __name__ == "__main__":
    main()
