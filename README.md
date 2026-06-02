# 🎓 NMIMS AI Chatbot

A Multi-LLM RAG (Retrieval-Augmented Generation) AI Assistant built specifically for NMIMS MBA students.

This chatbot helps students with:

* 📚 MBA concepts and theory
* 📝 Assignment assistance
* 📊 Case study analysis
* 💼 Interview preparation
* 📖 AI-generated notes
* 🔍 PDF-based question answering
* 📄 Source & page-level retrieval

---

# 🚀 Live Demo

https://nmims-ai-chatbot.streamlit.app/

---

# 🏗️ Architecture

The project follows a Retrieval-Augmented Generation (RAG) architecture:

1. MBA PDFs are loaded
2. Text is chunked into smaller sections
3. Embeddings are generated using Sentence Transformers
4. Chunks are stored in ChromaDB
5. User query is embedded
6. Similar chunks are retrieved
7. Context is sent to selected LLM
8. AI generates contextual response

---

# ⚡ Features

* Multi-LLM Support

  * Groq
  * Gemini
  * OpenAI

* AI Modes

  * MBA Tutor
  * Case Study Solver
  * Interview Prep
  * Research Analyst
  * Exam Mode
  * Corporate Consultant

* Streamlit Chat UI

* PDF-based RAG pipeline

* Source & Page Retrieval

* MBA Notes Generator

* Auto Vector DB Generation

* Cloud Deployment

---

# 🛠️ Tech Stack

| Component       | Technology           |
| --------------- | -------------------- |
| Frontend        | Streamlit            |
| Framework       | LangChain            |
| Vector Database | ChromaDB             |
| Embeddings      | all-MiniLM-L6-v2     |
| LLM Providers   | Groq, Gemini, OpenAI |
| PDF Loader      | PyPDFLoader          |
| Deployment      | Streamlit Cloud      |
| Language        | Python               |

---

# 📂 Project Structure

```bash
NMIMS-AI-Chatbot/
│
├── app.py
├── ingest.py
├── requirements.txt
├── runtime.txt
├── .gitignore
│
├── src/
│   ├── llm.py
│   └── retriever.py
│
├── data/
│   └── *.pdf
│
└── chroma_db/
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/anand-lab-172/NMIMS-AI-Chatbot.git

cd NMIMS-AI-Chatbot
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Add Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_key
OPENAI_API_KEY=your_key
GROQ_API_KEY=your_key
```

---

## 4. Run Ingestion

```bash
python ingest.py
```

---

## 5. Run Application

```bash
streamlit run app.py
```

---

# ☁️ Streamlit Deployment

The project is deployed using Streamlit Cloud.

Runtime:

```txt
python-3.11
```

---

# 📸 Screenshots

<img width="2851" height="1393" alt="image" src="https://github.com/user-attachments/assets/e5c8be4c-b5b0-4311-a659-b6342ec87eef" />


---

# 📌 Future Improvements

* Conversational memory
* Hybrid retrieval
* Cross-encoder reranking
* Authentication system
* PDF preview support
* User analytics dashboard

---

# 👨‍💻 Author

Anandaram Ganapathi

MBA – Operations & Data Science Management
NMIMS

---

# ⭐ If you found this useful

Please consider starring the repository.
