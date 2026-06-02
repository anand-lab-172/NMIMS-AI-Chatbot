import os

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    Chroma
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

# ---------------- PATHS ---------------- #

DATA_PATH = "data"

CHROMA_PATH = "chroma_db"

# ---------------- EMBEDDINGS ---------------- #

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- CREATE VECTOR DB ---------------- #

def create_vector_db():

    # Skip if DB already exists

    if os.path.exists(
        "chroma_db/chroma.sqlite3"
    ):

        print("✅ Chroma DB already exists")

        return

    all_docs = []

    # ---------------- LOAD PDFs ---------------- #

    for file in os.listdir(DATA_PATH):

        if file.endswith(".pdf"):

            pdf_path = os.path.join(
                DATA_PATH,
                file
            )

            print(f"Loading: {file}")

            loader = PyPDFLoader(
                pdf_path
            )

            documents = loader.load()

            for doc in documents:

                doc.metadata["source"] = file

                if "page" in doc.metadata:

                    doc.metadata["page"] = (
                        int(doc.metadata["page"]) + 1
                    )

                else:

                    doc.metadata["page"] = "N/A"

            all_docs.extend(documents)

    # ---------------- SPLIT ---------------- #

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        all_docs
    )

    print(
        f"Total Chunks: {len(chunks)}"
    )

    # ---------------- CREATE VECTORSTORE ---------------- #

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )

    vectorstore.persist()

    print(
        "✅ Chroma DB Created Successfully"
    )
