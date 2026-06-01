import streamlit as st
import os
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="AI Interview Assistant", layout="wide")

# ── LOAD NOTES ──────────────────────────────────
with open("notes.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ── SPLITTER ────────────────────────────────────
def split_text(text, chunk_size=400, overlap=40):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = split_text(text)

# ── MODEL ───────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ── CHROMADB ────────────────────────────────────
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="interview_notes")

# ── INDEXING ────────────────────────────────────
if collection.count() < len(chunks):
    with st.spinner("Indexing notes..."):
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[str(i)],
                metadatas=[{"source": "notes.txt"}]
            )

# ── GREETING CHECK ──────────────────────────────
GREETINGS = [
    "hello", "hi", "hey", "how are you",
    "good morning", "good evening", "what's up",
    "howdy", "namaste", "hii", "helo"
]

def is_greeting(query):
    return query.strip().lower() in GREETINGS

# ── CHAT MEMORY ─────────────────────────────────
if "chat" not in st.session_state:
    st.session_state.chat = []

# ── UI ──────────────────────────────────────────
st.title("💡 AI Interview Assistant")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask your interview question...")

# ── MAIN LOGIC ──────────────────────────────────
if query:
    st.chat_message("user").write(query)
    st.session_state.chat.append({"role": "user", "content": query})

    # ── GREETING HANDLE ─────────────────────────
    if is_greeting(query):
        answer = "Hello! 👋 I'm your AI Interview Assistant. Ask me any interview question — ML, Python, SQL, GenAI, System Design, and more!"

    else:
        # ── EMBEDDING ───────────────────────────
        query_embedding = model.encode(query).tolist()

        # ── RETRIEVAL ───────────────────────────
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )

        retrieved_chunks = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if "distances" in results else []

        # ── RELEVANCE CHECK ─────────────────────
        DISTANCE_THRESHOLD = 1.0

        if not retrieved_chunks or (
            len(distances) > 0 and min(distances) > DISTANCE_THRESHOLD
        ):
            context = ""
        else:
            context = "\n\n".join(retrieved_chunks)

        # ── PROMPT ──────────────────────────────
        if context.strip():
            prompt = f"""
You are an expert interview coach.
Answer the question using ONLY the context below.
Be concise and structured.

FORMAT:
1. Explanation
2. Key Points
3. Example (if available)

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
        else:
            prompt = f"""
You are an expert interview coach.
Answer this interview question from your knowledge.
Be concise and structured.

FORMAT:
1. Explanation
2. Key Points
3. Example (if available)

QUESTION:
{query}

ANSWER:
"""

        # ── LLM CALL ────────────────────────────
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        answer = response["message"]["content"]

    # ── DISPLAY ─────────────────────────────────
    st.chat_message("assistant").write(answer)
    st.session_state.chat.append({"role": "assistant", "content": answer})


    