import streamlit as st
import time

from src.retriever import retrieve_and_rerank

from src.llm import (
    ask_gemini,
    ask_chatgpt,
    ask_groq
)

st.set_page_config(
    page_title="NMIMS AI Chatbot",
    page_icon="🎓",
    layout="wide"
)

# ---------------- CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #AAAAAA;
    margin-bottom: 30px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
    '<div class="title">🎓 NMIMS AI Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Multi-LLM RAG-based NMIMS PDF Chatbot</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.title("⚡ AI Engine")

    engine = st.selectbox(
        "Choose AI Engine",
        [
            "Groq",
            "Gemini",
            "ChatGPT"
        ],
        index=0
    )

    # ---------------- GEMINI ---------------- #

    if engine == "Gemini":

        gemini_model = st.selectbox(
            "Choose Gemini Model",
            [
                "gemini-1.5-flash",
                "gemini-1.5-pro"
            ]
        )

    # ---------------- CHATGPT ---------------- #

    elif engine == "ChatGPT":

        chatgpt_model = st.selectbox(
            "Choose ChatGPT Model",
            [
                "gpt-4o-mini",
                "gpt-4o"
            ]
        )

    # ---------------- GROQ ---------------- #

    else:

        groq_model = st.selectbox(
            "Choose Groq Model",
            [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile"
            ],
            index=0
        )

    st.markdown("---")

    # ---------------- AI MODES ---------------- #

    mode = st.selectbox(
        "🧠 Choose AI Mode",
        [
            "MBA Tutor",
            "Case Study Solver",
            "Interview Prep",
            "Research Analyst",
            "Exam Mode",
            "Corporate Consultant"
        ]
    )

    st.markdown("---")

    # ---------------- NOTES ---------------- #

    st.subheader("📝 AI Notes")

    generate_notes = st.button(
        "Generate MBA Notes"
    )

# ---------------- CHAT HISTORY ---------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHATS ---------------- #

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- CONFIDENCE SCORE ---------------- #

def calculate_confidence(results, llm_score=None):

    try:

        if not results:
            return 60

        rerank_scores = [
            float(r[0])
            for r in results
        ]

        avg_score = (
            sum(rerank_scores)
            / len(rerank_scores)
        )

        retrieval_confidence = (
            avg_score * 100
        )

        retrieval_confidence = max(
            60,
            min(retrieval_confidence, 95)
        )

        if llm_score is not None:

            final_confidence = (
                retrieval_confidence * 0.4 +
                llm_score * 0.6
            )

            return round(
                min(final_confidence, 95),
                2
            )

        return round(
            retrieval_confidence,
            2
        )

    except:

        return 80

# ---------------- LLM EVALUATOR ---------------- #

def evaluate_answer(
    query,
    response,
    context,
    engine,
    model_name
):

    evaluation_prompt = f"""
You are an MBA professor evaluating an AI-generated answer.

Question:
{query}

Context:
{context}

Answer:
{response}

Evaluate:
- relevance
- analytical depth
- clarity
- completeness

Return ONLY a number between 1 and 100.
"""

    try:

        if engine == "Gemini":

            score = ask_gemini(
                evaluation_prompt,
                model_name
            )

        elif engine == "ChatGPT":

            score = ask_chatgpt(
                evaluation_prompt,
                model_name
            )

        else:

            score = ask_groq(
                evaluation_prompt,
                model_name
            )

        score = float(
            "".join(
                c for c in score
                if c.isdigit() or c == "."
            )
        )

        return max(1, min(score, 100))

    except:

        return 85

# ---------------- NOTES ---------------- #

def generate_mba_notes(context):

    notes_prompt = f"""
Generate MBA revision notes.

Format:
- Title
- Concepts
- Definitions
- Bullet Points
- Examples
- Summary

Context:
{context}
"""

    return notes_prompt

# ---------------- NOTES BUTTON ---------------- #

if generate_notes:

    with st.spinner("📝 Generating MBA Notes..."):

        try:

            results = retrieve_and_rerank(
                "Generate MBA notes"
            )

            docs = [
                doc
                for rerank_score,
                vector_score,
                doc in results
            ]

            context = "\n\n".join([
                doc.page_content
                for doc in docs
            ])

            notes_prompt = generate_mba_notes(
                context
            )

            if engine == "Gemini":

                notes = ask_gemini(
                    notes_prompt,
                    gemini_model
                )

            elif engine == "ChatGPT":

                notes = ask_chatgpt(
                    notes_prompt,
                    chatgpt_model
                )

            else:

                notes = ask_groq(
                    notes_prompt,
                    groq_model
                )

            st.success(
                "✅ MBA Notes Generated"
            )

            st.markdown(notes)

            st.download_button(
                label="📥 Download Notes",
                data=notes,
                file_name="mba_notes.txt",
                mime="text/plain"
            )

        except Exception as e:

            st.error(str(e))

# ---------------- USER INPUT ---------------- #

query = st.chat_input(
    "Ask your MBA question..."
)

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    status = st.empty()

    status.info(
        "🔍 Retrieving MBA knowledge..."
    )

    results = retrieve_and_rerank(query)

    docs = [
        doc
        for rerank_score,
        vector_score,
        doc in results
    ]

    unique_chunks = []

    seen = set()

    for doc in docs:

        content = doc.page_content.strip()

        if content not in seen:

            seen.add(content)

            unique_chunks.append(content)

    context = "\n\n".join(unique_chunks)

    chat_history = ""

    for msg in st.session_state.messages[-6:]:

        role = msg["role"]
        content = msg["content"]

        chat_history += (
            f"{role}: {content}\n"
        )

    # ---------------- AI MODES ---------------- #

    if mode == "MBA Tutor":

        system_prompt = """
You are an MBA professor.
Explain concepts clearly.
Use examples and structure.
"""

    elif mode == "Case Study Solver":

        system_prompt = """
You are a business consultant.
Provide SWOT and recommendations.
"""

    elif mode == "Interview Prep":

        system_prompt = """
You are an MBA interviewer.
Provide concise professional answers.
"""

    elif mode == "Research Analyst":

        system_prompt = """
You are a research analyst.
Provide detailed insights.
"""

    elif mode == "Exam Mode":

        system_prompt = """
Provide concise exam-ready answers.
"""

    else:

        system_prompt = """
You are a strategy consultant.
Provide executive insights.
"""

    final_prompt = f"""
Previous Conversation:
{chat_history}

{system_prompt}

Context:
{context}

Question:
{query}
"""

    active_model = "Unknown"

    with st.chat_message("assistant"):

        with st.spinner("🧠 Thinking..."):

            start_time = time.time()

            try:

                if engine == "Gemini":

                    response = ask_gemini(
                        final_prompt,
                        gemini_model
                    )

                    active_model = (
                        gemini_model
                    )

                elif engine == "ChatGPT":

                    response = ask_chatgpt(
                        final_prompt,
                        chatgpt_model
                    )

                    active_model = (
                        chatgpt_model
                    )

                else:

                    response = ask_groq(
                        final_prompt,
                        groq_model
                    )

                    active_model = (
                        groq_model
                    )

            except Exception as e:

                response = (
                    f"❌ Error:\n\n{str(e)}"
                )

            end_time = time.time()

            response_time = round(
                end_time - start_time,
                2
            )

            st.markdown(response)

            st.caption(
                f"⏱️ Response Time: "
                f"{response_time} sec"
            )

            llm_score = evaluate_answer(
                query,
                response,
                context,
                engine,
                active_model
            )

            confidence = calculate_confidence(
                results,
                llm_score
            )

            if confidence >= 80:

                st.success(
                    f"✅ AI Confidence Score: "
                    f"{confidence}%"
                )

            elif confidence >= 60:

                st.warning(
                    f"⚠️ AI Confidence Score: "
                    f"{confidence}%"
                )

            else:

                st.error(
                    f"❌ AI Confidence Score: "
                    f"{confidence}%"
                )

            st.progress(confidence / 100)

            with st.expander(
                "📚 Retrieved Context"
            ):

                for i, (
                    rerank_score,
                    vector_score,
                    doc
                ) in enumerate(results):

                    source = doc.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = doc.metadata.get(
                        "page",
                        "N/A"
                    )

                    st.markdown(
                        f"### Chunk {i+1}"
                    )

                    st.caption(
                        f"📄 Source: {source} | "
                        f"📑 Page: {page}"
                    )

                    st.caption(
                        f"🎯 Reranker: "
                        f"{round(float(rerank_score), 3)} | "
                        f"📏 Vector: "
                        f"{round(float(vector_score), 3)}"
                    )

                    st.write(
                        doc.page_content[:1000]
                    )

                    st.markdown("---")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

# ---------------- FOOTER ---------------- #

st.markdown(
    '<div class="footer">'
    'Built using LangChain • '
    'ChromaDB • Streamlit'
    '</div>',
    unsafe_allow_html=True
)
