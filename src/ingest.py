from langchain_community.vectorstores import (
    Chroma
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

CHROMA_PATH = "chroma_db"

# ---------------- EMBEDDINGS ---------------- #

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------- VECTORSTORE ---------------- #

def get_vectorstore():

    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model
    )

    return vectorstore

# ---------------- RETRIEVAL ---------------- #

def retrieve_and_rerank(query):

    try:

        vectorstore = get_vectorstore()

        docs = vectorstore.similarity_search_with_score(
            query,
            k=5
        )

        formatted_results = []

        for doc, score in docs:

            similarity_score = 1 / (
                1 + float(score)
            )

            similarity_score = round(
                similarity_score,
                3
            )

            formatted_results.append(
                (
                    similarity_score,
                    similarity_score,
                    doc
                )
            )

        return formatted_results

    except Exception as e:

        print(
            f"Retriever Error: {e}"
        )

        return []
