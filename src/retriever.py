import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder

# ---------------- CONFIG ---------------- #

CHROMA_PATH = "chroma_db"

# ---------------- EMBEDDINGS ---------------- #

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# ---------------- VECTOR DB ---------------- #

vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

# ---------------- RERANKER ---------------- #

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ---------------- RETRIEVE + RERANK ---------------- #

def retrieve_and_rerank(query, top_k=6):

    # ---------------- VECTOR SEARCH ---------------- #

    retrieved_docs = vectordb.similarity_search_with_score(
        query,
        k=top_k
    )

    # retrieved_docs = [(doc, vector_score)]

    docs = [doc for doc, score in retrieved_docs]

    # ---------------- RERANK ---------------- #

    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    # ---------------- COMBINE ---------------- #

    scored_docs = []

    for rerank_score, (doc, vector_score) in zip(
        scores,
        retrieved_docs
    ):

        if rerank_score < 2:
            continue

        scored_docs.append(
            (
                rerank_score,
                vector_score,
                doc
            )
        )

    # ---------------- SORT ---------------- #

    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scored_docs[:3]