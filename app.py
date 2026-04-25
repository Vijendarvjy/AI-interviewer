import os
import re
import time
import queue
import tempfile
import threading
from io import BytesIO

import streamlit as st
import pandas as pd
import plotly.express as px

from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
from audio_recorder_streamlit import audio_recorder

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ============================================================
# IMPORTANT: ADD THIS AT THE VERY TOP OF YOUR APP
# (before importing sentence_transformers, torch, or langchain)
# ============================================================

# Disable Streamlit file watcher (fixes torch.classes error)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Optional: suppress torch dynamo issues
os.environ["TORCHDYNAMO_DISABLE"] = "1"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Ketu AI Voice Interviewer",
    page_icon="🎙️",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}
.stButton>button {
    width: 100%;
    border-radius: 15px;
    border: none;
    background: linear-gradient(90deg,#7c3aed,#2563eb);
    color: white;
    padding: 0.8rem;
    font-weight: 700;
}
.question-card {
    background: linear-gradient(135deg,#1d4ed8,#7c3aed);
    padding: 2rem;
    border-radius: 25px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.feedback-card {
    background: #111827;
    padding: 1.5rem;
    border-radius: 20px;
    border-left: 6px solid #8b5cf6;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# API KEY
# ============================================================
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ============================================================
# LLM
# ============================================================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=4096,
)

# ============================================================
# EMBEDDINGS (STREAMLIT CLOUD SAFE)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_embeddings():
    """
    Uses CPU-only SentenceTransformers.
    Avoids PyTorch meta-tensor errors on Streamlit Cloud.
    """
    from sentence_transformers import SentenceTransformer

    class LocalEmbeddings:
        def __init__(self):
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )

        def embed_documents(self, texts):
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embeddings.tolist()

        def embed_query(self, text):
            embedding = self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            return embedding.tolist()

    return LocalEmbeddings()

embeddings = load_embeddings()

# ============================================================
# TEXT TO SPEECH
# ============================================================
def speak_text(text: str):
    tts = gTTS(text=text, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    st.audio(fp.read(), format='audio/mp3', autoplay=True)

# ============================================================
# DOCUMENT LOADER
# ============================================================
def load_document(uploaded_file):
    suffix = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{suffix}') as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    try:
        if suffix == 'pdf':
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            return '\n'.join([d.page_content for d in docs])

        elif suffix in ['docx', 'doc']:
            loader = Docx2txtLoader(temp_path)
            docs = loader.load()
            return docs[0].page_content

        else:
            return uploaded_file.getvalue().decode('utf-8', errors='ignore')
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ============================================================
# VECTOR STORE
# ============================================================
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    docs = splitter.create_documents([text])
    return FAISS.from_documents(docs, embeddings)

# ============================================================
# GENERATE QUESTIONS
# ============================================================
def generate_questions(job_description, resume_text, num_questions=10):
    prompt = f"""
    You are an expert senior interviewer.

    Based on the job description and candidate resume,
    generate exactly {num_questions} highly relevant interview questions.

    JOB DESCRIPTION:
    {job_description[:2500]}

    RESUME:
    {resume_text[:2500]}

    Rules:
    - Ask one question at a time.
    - Questions should feel natural and conversational.
    - Start easy, then gradually increase difficulty.
    - Return only numbered questions.
    """

    response = llm.invoke(prompt)
    content = response.content

    questions = []
    for line in content.splitlines():
        line = line.strip()
        if line and any(c.isdigit() for c in line[:3]):
            cleaned = re.sub(r'^\d+[.)]\s*', '', line)
            questions.append(cleaned)

    return questions[:num_questions]

# ============================================================
# EVALUATE ANSWER
# ============================================================
def evaluate_answer(question, answer):
    prompt = f"""
    You are an elite technical interviewer.

    QUESTION:
    {question}

    ANSWER:
    {answer}

    Evaluate professionally.

    Return strictly in this format:

    SCORE: <number>/10
    FEEDBACK: <detailed feedback>
    IMPROVEMENT: <suggestions>
    """

    response = llm.invoke(prompt)
    return response.content

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🎯 Ketu AI")
    st.subheader("Human-Like AI Interviewer")
    st.success("Voice + Groq + RAG")

# ============================================================
# MAIN UI
# ============================================================
st.title("🎙️ Ketu AI Voice Interviewer")
st.markdown("Conduct realistic AI-powered interviews with voice interaction.")

col1, col2 = st.columns(2)

with col1:
    jd_text = st.text_area(
        "Paste Job Description",
        height=350
    )

with col2:
    resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx", "txt"]
    )

# ============================================================
# START INTERVIEW
# ============================================================
if st.button("🚀 Start Voice Interview"):
    if not jd_text or not resume_file:
        st.error("Please provide both Job Description and Resume.")
        st.stop()

    with st.spinner("Preparing interview..."):
        resume_text = load_document(resume_file)
        questions = generate_questions(jd_text, resume_text)

        st.session_state.questions = questions
        st.session_state.current = 0
        st.session_state.scores = []
        st.session_state.started = True

# ============================================================
# INTERVIEW FLOW
# ============================================================
if st.session_state.get("started", False):
    questions = st.session_state.questions
    current = st.session_state.current

    if current < len(questions):
        question = questions[current]

        st.markdown(f"""
        <div class="question-card">
            <h2>Question {current + 1}</h2>
            <h3>{question}</h3>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔊 Read Question"):
            speak_text(question)

        st.markdown("### 🎤 Record Your Answer")
        audio_bytes = audio_recorder(
            text="Click to Record",
            recording_color="#e11d48",
            neutral_color="#2563eb",
            icon_name="microphone",
            icon_size="2x"
        )

        answer = st.text_area(
            "Or type your answer",
            key=f"answer_{current}",
            height=200
        )

        if st.button("Submit Answer"):
            if not answer.strip():
                st.warning("Please provide your answer.")
                st.stop()

            with st.spinner("AI evaluating your response..."):
                feedback = evaluate_answer(question, answer)

                st.markdown("""
                <div class="feedback-card">
                """, unsafe_allow_html=True)

                st.markdown("### 🧠 Interview Feedback")
                st.write(feedback)

                st.markdown("</div>", unsafe_allow_html=True)

                score_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*10', feedback)
                score = float(score_match.group(1)) if score_match else 8.0

                st.session_state.scores.append({
                    "Question": current + 1,
                    "Score": score
                })

                st.session_state.current += 1
                st.rerun()

    else:
        st.balloons()
        st.success("🎉 Interview Completed!")

        df = pd.DataFrame(st.session_state.scores)

        fig = px.line(
            df,
            x="Question",
            y="Score",
            markers=True,
            title="Performance Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

        avg = df["Score"].mean()

        st.metric(
            "Overall Score",
            f"{avg:.1f}/10"
        )

        if avg >= 8:
            verdict = "Excellent performance!"
        elif avg >= 6:
            verdict = "Good performance with room for improvement."
        else:
            verdict = "Needs more preparation."

        st.markdown(f"## 🏆 Final Verdict\n{verdict}")
