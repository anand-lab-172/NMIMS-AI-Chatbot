from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = "chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_retriever():

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever

def retrieve_and_rerank(query):

    retriever = get_retriever()

    docs = retriever.get_relevant_documents(query)

    return docs
