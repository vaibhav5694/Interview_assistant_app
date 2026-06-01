import streamlit as st
import os
import chromadb
import PyPDF2
import ollama
import re
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Interview System", layout="wide")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# SIDEBAR / SETTINGS
# -----------------------------
st.title("💡 AI Interview System (Custom Topics + RAG)")

st.subheader("🎯 Interview Settings")

topics = st.multiselect(
    "Select Topics",
    ["SQL", "Python", "Data Engineering", "Machine Learning", "DSA", "System Design"]
)

num_questions = st.slider("Number of Questions", 1, 15, 10)

# -----------------------------
# LOAD NOTES
# -----------------------------
with open("notes.txt", "r", encoding="utf-8") as f:
    notes = f.read()

def split_text(text, size=400):
    return [text[i:i+size] for i in range(0, len(text), size)]

chunks = split_text(notes)

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# VECTOR DB
# -----------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("notes")

# -----------------------------
# INDEXING
# -----------------------------
if collection.count() == 0:
    with st.spinner("Indexing notes..."):
        for i, chunk in enumerate(chunks):
            emb = model.encode(chunk).tolist()
            collection.add(
                documents=[chunk],
                embeddings=[emb],
                ids=[str(i)]
            )

# -----------------------------
# RETRIEVAL
# -----------------------------
def search(query):
    q_emb = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=5
    )
    return " ".join(results["documents"][0])

# -----------------------------
# RESUME PARSER
# -----------------------------
def extract_resume(file):
    text = ""
    if file is not None:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# -----------------------------
# SCORE EXTRACTION
# -----------------------------
def extract_score(text):
    match = re.search(r"Score\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

# -----------------------------
# QUESTION GENERATION
# -----------------------------
def generate_questions(context, topics, num_questions):

    prompt = f"""
You are an expert technical interviewer.

Generate EXACTLY {num_questions} interview questions.

Focus ONLY on these topics:
{topics}

Rules:
- Mix difficulty (easy, medium, hard)
- Each question on new line
- No numbering
- No extra text

CONTEXT:
{context}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# -----------------------------
# EVALUATION
# -----------------------------
def evaluate_answer(q, a):

    prompt = f"""
You are an interview evaluator.

Question: {q}

Answer: {a}

Return strictly in this format:
Score: (0-10)
Correct/Incorrect
Feedback
Improved Answer
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# -----------------------------
# UI INPUTS
# -----------------------------
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd = st.text_area("Paste Job Description (optional)")

start = st.button("Start Interview")

# -----------------------------
# SESSION STATE
# -----------------------------
if "questions" not in st.session_state:
    st.session_state.questions = []

if "index" not in st.session_state:
    st.session_state.index = 0

if "answers" not in st.session_state:
    st.session_state.answers = []

if "show_feedback" not in st.session_state:
    st.session_state.show_feedback = False

# -----------------------------
# START INTERVIEW
# -----------------------------
if start:

    if not topics:
        st.error("Please select at least one topic!")
        st.stop()

    resume_text = extract_resume(resume_file)

    rag_context = search(jd + " " + resume_text + " interview")

    context = f"""
RESUME:
{resume_text}

JOB DESCRIPTION:
{jd}

SELECTED TOPICS:
{topics}

NOTES:
{rag_context}
"""

    with st.spinner("Generating Questions..."):
        q_text = generate_questions(context, topics, num_questions)

    questions = [q.strip() for q in q_text.split("\n") if q.strip() != ""]

    st.session_state.questions = questions[:num_questions]
    st.session_state.index = 0
    st.session_state.answers = []
    st.session_state.show_feedback = False

    st.rerun()

# -----------------------------
# INTERVIEW FLOW
# -----------------------------
if st.session_state.questions:

    idx = st.session_state.index
    total_q = len(st.session_state.questions)

    st.info(f"📊 Question {idx+1} of {total_q}")

    if idx < total_q:

        q = st.session_state.questions[idx]

        st.subheader(f"Q{idx+1}: {q}")

        ans = st.text_area("Your Answer", key=f"ans_{idx}")

        if st.button("Submit Answer", key=f"submit_{idx}"):

            if ans.strip() == "":
                st.warning("Please write an answer first!")

            else:
                with st.spinner("Evaluating..."):
                    result = evaluate_answer(q, ans)
                    score = extract_score(result)

                st.session_state.answers.append({
                    "question": q,
                    "answer": ans,
                    "feedback": result,
                    "score": score
                })

                st.session_state.last_result = result
                st.session_state.last_score = score
                st.session_state.show_feedback = True

        if st.session_state.show_feedback:

            st.success(f"🎯 Score: {st.session_state.last_score}/10")
            st.write("### AI Feedback")
            st.write(st.session_state.last_result)

            if st.button("➡ Next Question", key=f"next_{idx}"):

                st.session_state.index += 1
                st.session_state.show_feedback = False
                st.rerun()

    else:

        st.success("🎉 Interview Completed!")

        total_score = sum([a["score"] for a in st.session_state.answers])
        max_score = len(st.session_state.answers) * 10
        avg_score = total_score / len(st.session_state.answers)

        st.subheader(f"📊 Final Score: {total_score}/{max_score}")
        st.subheader(f"📈 Average Score: {avg_score:.2f}/10")

        st.write("## Detailed Report")

        for i, item in enumerate(st.session_state.answers):

            st.markdown(f"### Q{i+1}")
            st.markdown(f"**Question:** {item['question']}")
            st.markdown(f"**Score:** {item['score']}/10")
            st.markdown(f"**Feedback:** {item['feedback']}")
            st.markdown("---")



