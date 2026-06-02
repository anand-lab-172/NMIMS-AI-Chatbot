```python id="jlwm26"
import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from sentence_transformers import CrossEncoder

# ---------------- CONFIG ---------------- #

CHROMA_PATH = "chroma_db"

RERANK_THRESHOLD = 0.5

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

    try:

        # ---------------- VECTOR SEARCH ---------------- #

        retrieved_docs = vectordb.similarity_search_with_score(
            query,
            k=top_k
        )

        # ---------------- EMPTY CHECK ---------------- #

        if not retrieved_docs:

            return []

        # ---------------- EXTRACT DOCS ---------------- #

        docs = []

        for item in retrieved_docs:

            try:

                doc, vector_score = item

                if hasattr(doc, "page_content"):

                    content = doc.page_content.strip()

                    if content:

                        docs.append(
                            (doc, vector_score)
                        )

            except:

                continue

        # ---------------- EMPTY DOC CHECK ---------------- #

        if len(docs) == 0:

            return []

        # ---------------- BUILD PAIRS ---------------- #

        pairs = [

            (query, doc.page_content)

            for doc, vector_score in docs
        ]

        # ---------------- EMPTY PAIRS CHECK ---------------- #

        if len(pairs) == 0:

            return []

        # ---------------- RERANK ---------------- #

        scores = reranker.predict(pairs)

        # ---------------- COMBINE ---------------- #

        scored_docs = []

        for rerank_score, (
            doc,
            vector_score
        ) in zip(scores, docs):

            if rerank_score < RERANK_THRESHOLD:
                continue

            scored_docs.append(
                (
                    float(rerank_score),
                    float(vector_score),
                    doc
                )
            )

        # ---------------- SORT ---------------- #

        scored_docs.sort(
            key=lambda x: x[0],
            reverse=True
        )

        # ---------------- FALLBACK ---------------- #

        if len(scored_docs) == 0:

            scored_docs = [

                (
                    0,
                    vector_score,
                    doc
                )

                for doc, vector_score
                in docs[:3]
            ]

        return scored_docs[:3]

    except Exception as e:

        print("Retriever Error:", str(e))

        return []
```
