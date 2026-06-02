
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
    '<div class="subtitle">RAG-based MBA Academic Assistant</div>',
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

    if not results:
        return 0

    try:

        rerank_scores = [
            float(rerank_score)
            for rerank_score, vector_score, doc
            in results
        ]

        avg_score = (
            sum(rerank_scores)
            / len(rerank_scores)
        )

        normalized = (
            (avg_score + 10) / 20
        ) * 100

        retrieval_confidence = max(
            45,
            min(normalized, 95)
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

        return 75

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

Evaluate based on:
- relevance
- analytical depth
- business understanding
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

        return None

# ---------------- NOTES GENERATOR ---------------- #

def generate_mba_notes(context):

    notes_prompt = f"""
You are an MBA professor.

Generate MBA revision notes.

Format:
- Title
- Key Concepts
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
                "Generate comprehensive MBA notes"
            )

            context = "\n\n".join([

                doc.page_content

                for rerank_score,
                vector_score,
                doc in results
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

    # ---------------- RETRIEVE ---------------- #

    results = retrieve_and_rerank(query)

    # ---------------- CONTEXT ---------------- #

    unique_chunks = []

    seen = set()

    for rerank_score, vector_score, doc in results:

        content = doc.page_content.strip()

        if content not in seen:

            seen.add(content)

            unique_chunks.append(content)

    context = "\n\n".join(unique_chunks)

    # ---------------- MEMORY ---------------- #

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
You are an expert MBA professor.

Explain concepts clearly.

Use:
- examples
- business context
- structured explanations
"""

    elif mode == "Case Study Solver":

        system_prompt = """
You are a top-tier business consultant.

Provide:
- SWOT analysis
- recommendations
- business impact
"""

    elif mode == "Interview Prep":

        system_prompt = """
You are an MBA placement interviewer.

Provide concise professional responses.
"""

    elif mode == "Research Analyst":

        system_prompt = """
You are a business research analyst.

Provide detailed insights and analysis.
"""

    elif mode == "Exam Mode":

        system_prompt = """
You are an MBA exam assistant.

Provide short and direct answers.
"""

    else:

        system_prompt = """
You are a corporate strategy consultant.

Provide executive-level strategic insights.
"""

    # ---------------- FINAL PROMPT ---------------- #

    final_prompt = f"""
Previous Conversation:
{chat_history}

{system_prompt}

IMPORTANT:

1. Use uploaded documents first
2. If context is insufficient, say so
3. Avoid hallucinations

Context:
{context}

Question:
{query}
"""

    active_model = "Unknown"

    # ---------------- RESPONSE ---------------- #

    with st.chat_message("assistant"):

        with st.spinner("🧠 Thinking..."):

            start_time = time.time()

            try:

                status.info(
                    "🧠 Generating MBA insights..."
                )

                if engine == "Gemini":

                    response = ask_gemini(
                        final_prompt,
                        gemini_model
                    )

                    active_model = gemini_model

                elif engine == "ChatGPT":

                    response = ask_chatgpt(
                        final_prompt,
                        chatgpt_model
                    )

                    active_model = chatgpt_model

                else:

                    response = ask_groq(
                        final_prompt,
                        groq_model
                    )

                    active_model = groq_model

            except Exception as e:

                response = (
                    f"❌ Error:\n\n{str(e)}"
                )

            end_time = time.time()

            response_time = round(
                end_time - start_time,
                2
            )

            # ---------------- SHOW RESPONSE ---------------- #

            st.markdown(response)

            st.caption(
                f"⏱️ Response Time: "
                f"{response_time} sec"
            )

            # ---------------- EVALUATION ---------------- #

            status.info(
                "📊 Evaluating response..."
            )

            llm_score = evaluate_answer(
                query,
                response,
                context,
                engine,
                active_model
            )

            # ---------------- CONFIDENCE ---------------- #

            confidence = calculate_confidence(
                results,
                llm_score
            )

            status.success(
                "✅ Response Ready"
            )

            # ---------------- SHOW CONFIDENCE ---------------- #

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

            # ---------------- CONTEXT ---------------- #

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
                        f"📄 Source: {source}"
                    )

                    st.caption(
                        f"📑 Page: {page}"
                    )

                    st.caption(
                        f"🎯 Reranker: "
                        f"{round(float(rerank_score), 3)}"
                    )

                    st.caption(
                        f"📏 Vector Score: "
                        f"{round(float(vector_score), 3)}"
                    )

                    st.write(
                        doc.page_content[:1000]
                    )

                    st.markdown("---")

    # ---------------- SAVE CHAT ---------------- #

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
