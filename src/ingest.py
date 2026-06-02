from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import os

DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

all_docs = []

# ---------------- LOAD PDFs ---------------- #

for file in os.listdir(DATA_PATH):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(
            DATA_PATH,
            file
        )

        loader = PyPDFLoader(pdf_path)

        documents = loader.load()

        # ADD SOURCE + PAGE

        for doc in documents:

            doc.metadata["source"] = file

            # page already exists
            # make it human readable

            if "page" in doc.metadata:

                doc.metadata["page"] = (
                    int(doc.metadata["page"]) + 1
                )

        all_docs.extend(documents)

# ---------------- SPLIT ---------------- #

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(
    all_docs
)

# ---------------- CREATE DB ---------------- #

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH
)

vectorstore.persist()

print("✅ Chroma DB Created Successfully")
