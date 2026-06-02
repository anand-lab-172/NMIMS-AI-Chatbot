from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vectorstore():

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    return vectorstore

def retrieve_and_rerank(query):

    vectorstore = get_vectorstore()

    docs = vectorstore.similarity_search(
        query,
        k=5
    )

    return docs
