import os

from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- CONFIG ---------------- #

PDF_FOLDER = "data"
CHROMA_PATH = "chroma_db"

# ---------------- LOAD PDFS ---------------- #

def load_documents():

    documents = []

    pdf_files = [
        file for file in os.listdir(PDF_FOLDER)
        if file.endswith(".pdf")
    ]

    print(f"Found {len(pdf_files)} PDFs")

    for file in tqdm(pdf_files, desc="Loading PDFs"):

        pdf_path = os.path.join(PDF_FOLDER, file)

        loader = PyPDFLoader(pdf_path)

        pages = loader.load()

        for page_num, page in enumerate(pages):

            page.metadata["source"] = file
            page.metadata["page"] = page_num + 1

            documents.append(page)

    return documents

# ---------------- SPLIT TEXT ---------------- #

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)

# ---------------- CREATE VECTOR STORE ---------------- #

def create_vector_store(chunks):

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
    )

    print("Creating ChromaDB...")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print("ChromaDB created successfully")

# ---------------- MAIN ---------------- #

if __name__ == "__main__":

    docs = load_documents()

    print(f"Loaded {len(docs)} pages")

    chunks = split_documents(docs)

    print(f"Created {len(chunks)} chunks")

    create_vector_store(chunks)

