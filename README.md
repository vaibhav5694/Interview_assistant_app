🚀 AI Interview Evaluator (RAG + Resume + JD + Local LLM)

An AI-powered Interview Evaluation System built using Python, Streamlit, ChromaDB, SentenceTransformers, and Ollama (Mistral).

The application generates personalized interview questions using a candidate's Resume, Job Description (JD), and a custom Knowledge Base (notes.txt). It then evaluates answers and provides structured feedback in real time.

📌 Features
🎯 Personalized Interview Generation
Upload Resume (PDF)
Upload/Paste Job Description
Generate role-specific interview questions
📚 Knowledge Base Driven
Uses a custom notes.txt
Supports SQL, Python, AWS, Power BI, Pandas, ETL, Data Engineering, Analytics, etc.
Questions are generated based on uploaded knowledge
🧠 Retrieval Augmented Generation (RAG)
Semantic search using embeddings
Retrieves only relevant content from notes
Reduces hallucinations
🤖 AI Answer Evaluation
Evaluates candidate responses
Provides:
Score
Feedback
Missing Points
Improved Answer
🔒 Fully Local & Privacy Friendly
Uses Ollama + Mistral locally
No OpenAI API required
No data leaves your machine
💰 Zero Cost
100% free stack
No API charges





User
 │
 ▼
Streamlit UI
 │
 ├── Resume Upload
 ├── Job Description
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
Sentence Transformers
(all-MiniLM-L6-v2)
 │
 ▼
ChromaDB
(Vector Database)
 │
 ▼
Semantic Retrieval (RAG)
 │
 ▼
Ollama (Mistral)
 │
 ├── Question Generation
 ├── Answer Evaluation
 │
 ▼
Feedback & Scoring

