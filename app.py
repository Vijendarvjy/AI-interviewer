# app.py

import os
import tempfile
import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Ketu AI - Smart Interviewer",
    page_icon="🎯",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.stButton>button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.75rem 1.5rem;
    font-weight: bold;
}
.metric-card {
    background: linear-gradient(135deg,#1d4ed8,#7c3aed);
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# API KEY
# ---------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ---------------------------
# LLM
# ---------------------------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-prompt-guard-2-86m",
    temperature=0.7
)

# ---------------------------
# EMBEDDINGS
# ---------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------
# DOCUMENT LOADER
# ---------------------------
def load_document(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    if suffix == "pdf":
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        return "\n".join([d.page_content for d in docs])

    elif suffix in ["docx", "doc"]:
        loader = Docx2txtLoader(temp_path)
        docs = loader.load()
        return docs[0].page_content

    else:
        return uploaded_file.read().decode()

# ---------------------------
# VECTOR STORE
# ---------------------------
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    return vectorstore

# ---------------------------
# GENERATE QUESTIONS
# ---------------------------
def generate_questions(
    job_description: str,
    difficulty: str,
    num_questions: int = 10
) -> list[str]:
    """
    Generate interview questions while handling Groq token limits safely.
    """

    # Ensure integer value
    try:
        num_questions = int(num_questions)
    except (ValueError, TypeError):
        num_questions = 10

    num_questions = max(1, min(num_questions, 20))

    # Aggressively trim JD
    trimmed_jd = (job_description or "").strip()[:2000]

    prompt = f"""
You are an expert technical interviewer.

Generate exactly {num_questions} interview questions.

Difficulty: {difficulty}

Job Description:
{trimmed_jd}

Rules:
- Return only numbered questions
- No explanations
- No headings
- Keep each question concise
"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        questions = [
            line.strip()
            for line in content.splitlines()
            if line.strip()
        ]

        if not questions:
            raise ValueError("No questions generated.")

        return questions[:num_questions]

    except Exception as e:
        st.warning(f"Groq API fallback activated: {e}")

        fallback_questions = [
            "1. Tell me about yourself.",
            "2. Walk me through your recent project.",
            "3. What technical challenges have you solved recently?",
            "4. How do you debug production issues?",
            "5. Explain your preferred development workflow.",
            "6. How do you ensure code quality?",
            "7. Describe a difficult team collaboration.",
            "8. How do you prioritize tasks under pressure?",
            "9. What new technology have you learned recently?",
            "10. Why should we hire you?"
        ]

        return fallback_questions[:num_questions]
# ---------------------------
# EVALUATE ANSWER
# ---------------------------
def evaluate_answer(question, answer):
    prompt = f"""
    You are Ketu AI.

    Interview Question:
    {question}

    Candidate Answer:
    {answer}

    Evaluate on:
    - Technical Accuracy
    - Communication
    - Confidence
    - Completeness

    Give:
    - Score out of 10
    - Detailed feedback
    - Improvement suggestions
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=120
    )
    st.title("🎯 Ketu AI")
    st.markdown("### Intelligent Interview Agent")
    st.success("Powered by Groq + LangChain + RAG")

# ---------------------------
# MAIN UI
# ---------------------------
st.title("🚀 Ketu AI - AI Interviewer")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    jd_text = st.text_area(
        "📄 Paste Job Description",
        height=300
    )

with col2:
    resume_file = st.file_uploader(
        "📑 Upload Resume",
        type=["pdf", "docx", "txt"]
    )

# ---------------------------
# PROCESS
# ---------------------------
if st.button("🎯 Start Interview"):
    if not jd_text or not resume_file:
        st.error("Please provide JD and Resume.")
        st.stop()

    with st.spinner("Analyzing candidate profile..."):
        resume_text = load_document(resume_file)
        combined_text = jd_text + "\n" + resume_text

        vectorstore = create_vectorstore(combined_text)
        questions = generate_questions(
            jd_text,
            resume_text,
            vectorstore
        )

        st.session_state.questions = questions.split("\n")
        st.session_state.current = 0
        st.session_state.score = []

# ---------------------------
# INTERVIEW FLOW
# ---------------------------
if "questions" in st.session_state:
    current = st.session_state.current
    questions = st.session_state.questions

    if current < len(questions):
        question = questions[current]

        st.markdown(f"""
        <div class="metric-card">
            <h2>Question {current+1}</h2>
            <h3>{question}</h3>
        </div>
        """, unsafe_allow_html=True)

        answer = st.text_area(
            "🎤 Your Answer",
            key=f"answer_{current}",
            height=200
        )

        if st.button("Submit Answer"):
            feedback = evaluate_answer(question, answer)

            st.markdown("### 🧠 Feedback")
            st.write(feedback)

            st.session_state.score.append({
                "Question": current + 1,
                "Score": 8
            })

            st.session_state.current += 1
            st.rerun()

    else:
        st.balloons()
        st.success("Interview Completed Successfully!")

        df = pd.DataFrame(st.session_state.score)

        fig = px.bar(
            df,
            x="Question",
            y="Score",
            title="Performance Analysis",
            color="Score",
            text="Score"
        )

        st.plotly_chart(fig, use_container_width=True)

        avg_score = df["Score"].mean()

        st.metric(
            "Overall Score",
            f"{avg_score:.1f}/10"
        )

        st.markdown("""
        ## 🎉 Ketu AI Summary
        Excellent effort! Your interview analysis is complete.
        """)
