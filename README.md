# 🚀 AI Interview Evaluator (RAG + Resume + JD + Local LLM)

An AI-powered Interview Evaluation System built using Python, Streamlit, ChromaDB, SentenceTransformers, and Ollama (Mistral).

The application generates personalized interview questions using a candidate's Resume, Job Description (JD), and a custom Knowledge Base (`notes.txt`). It then evaluates answers and provides structured feedback in real time.

---

## 📌 Features

### 🎯 Personalized Interview Generation
* **Resume Upload:** Supports PDF format.
* **Job Description:** Easily upload or paste the JD.
* **Smart Generation:** Generates role-specific, tailored interview questions.

### 📚 Knowledge Base Driven
* Uses a custom `notes.txt` file as the source of truth.
* Supports **SQL, Python, AWS, Power BI, Pandas, ETL, Data Engineering, Analytics**, etc.
* Questions are dynamically generated based on the uploaded knowledge base.

### 🧠 Retrieval Augmented Generation (RAG)
* Semantic search using vector embeddings.
* Retrieves only the most relevant content from your notes.
* Significantly reduces AI hallucinations.

### 🤖 AI Answer Evaluation
* Evaluates candidate responses instantly.
* **Provides structured feedback:**
  * 🎯 Score
  * 💬 Detailed Feedback
  * 🔍 Missing Points
  * 💡 Improved Answer Example

### 🔒 Fully Local & Privacy Friendly
* Powered by **Ollama + Mistral** running locally on your hardware.
* No OpenAI or third-party API keys required.
* Complete data privacy—no data ever leaves your machine.

### 💰 Zero Cost
* 100% free open-source stack.
* Zero API charges or subscription fees.

---

## 📐 Architecture & Workflow

```text
       User
        │
        ▼
   Streamlit UI
        │
        ├── Resume Upload
        └── Job Description
        │
        ▼
 Text Extraction
        │
        ▼
 Knowledge Base (notes.txt)
        │
        ▼
    Chunking
        │
        ▼
Sentence Transformers (all-MiniLM-L6-v2)
        │
        ▼
    ChromaDB (Vector Database)
        │
        ▼
Semantic Retrieval (RAG)
        │
        ▼
   Ollama (Mistral)
        │
        ├── Question Generation
        └── Answer Evaluation
        │
        ▼
Feedback & Scoring
