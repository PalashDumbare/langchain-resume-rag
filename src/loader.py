from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader


def load_documents(data_dir: str):
    documents = []

    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = (Path(__file__).resolve().parent.parent / data_path).resolve()

    pdf_files = sorted(data_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {data_path}.")
    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    return documents


if __name__ == "__main__":
    data_dir = "data"
    documents = load_documents(data_dir)
    print(f"Loaded {len(documents)} documents from {data_dir}.")