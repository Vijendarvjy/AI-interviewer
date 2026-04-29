# ============================================================
# CRITICAL ENV VARS — must be first, before any other imports
# ============================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ── Torch patch (must be before transformers) ──────────────
import torch

class _TorchClassesPatch:
    def __getattr__(self, name):
        if name in ["__path__", "_path"]:
            return []
        raise AttributeError(name)

torch.classes = _TorchClassesPatch()

# ── Standard library ───────────────────────────────────────
import re
import io
from openai import OpenAI
import time
import base64
import tempfile
import threading
from io import BytesIO
from datetime import datetime

# ── Third-party ────────────────────────────────────────────
import streamlit as st

try:
    st.set_option("server.fileWatcherType", "none")
except Exception:
    pass

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from gtts import gTTS
from sentence_transformers import SentenceTransformer

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ── Optional audio recorder ────────────────────────────────
try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False

try:
    from streamlit_mic_recorder import mic_recorder
    HAS_MIC_RECORDER = True
except ImportError:
    HAS_MIC_RECORDER = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Ketu AI · Voice Interviewer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DESIGN SYSTEM & CSS
# ============================================================
DESIGN = """
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:         #03060f;
    --surface:    #080d1a;
    --surface2:   #0d1526;
    --border:     #1a2540;
    --accent:     #00d4ff;
    --accent2:    #7b2fff;
    --accent3:    #ff4d6d;
    --success:    #00e5a0;
    --warning:    #ffb700;
    --text:       #e2eaf8;
    --muted:      #4a5a7a;
    --radius:     18px;
    --glow-blue:  0 0 50px rgba(0,212,255,0.12);
    --glow-purple:0 0 50px rgba(123,47,255,0.12);
}

/* ── Reset ─────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 100% 50% at 50% -5%, rgba(0,212,255,0.07), transparent),
        radial-gradient(ellipse 60% 40% at 100% 80%, rgba(123,47,255,0.07), transparent),
        radial-gradient(ellipse 50% 40% at 0% 60%,   rgba(255,77,109,0.04), transparent),
        var(--bg) !important;
}
[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Scanline texture overlay ───────────────────────── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,212,255,0.012) 2px,
        rgba(0,212,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* ── Typography ─────────────────────────────────────── */
h1 { font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 0.06em !important; }
h2, h3 { font-family: 'Outfit', sans-serif !important; font-weight: 600 !important; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.25s cubic-bezier(0.34,1.56,0.64,1) !important;
    letter-spacing: 0.03em !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 25px rgba(0,212,255,0.2), inset 0 0 25px rgba(0,212,255,0.04) !important;
    transform: translateY(-2px) scale(1.01) !important;
}
.stButton > button:active { transform: translateY(0) scale(0.99) !important; }

/* ── Text area ───────────────────────────────────────── */
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.1) !important;
}

/* ── File uploader ───────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Progress ────────────────────────────────────────── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--accent2), var(--accent)) !important;
    border-radius: 99px !important;
    box-shadow: 0 0 12px rgba(0,212,255,0.4) !important;
}
.stProgress > div > div {
    background: var(--border) !important;
    border-radius: 99px !important;
    height: 6px !important;
}

/* ── Selectbox ───────────────────────────────────────── */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}

/* ── Slider ──────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }

/* ── Metrics ─────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.4rem !important;
    transition: box-shadow 0.2s ease !important;
}
[data-testid="stMetric"]:hover { box-shadow: var(--glow-blue) !important; }
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2.4rem !important;
    letter-spacing: 0.06em !important;
    color: var(--accent) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; }

/* ── Spinner ─────────────────────────────────────────── */
.stSpinner > div { border-color: var(--accent) transparent transparent transparent !important; }

/* ── Alerts ──────────────────────────────────────────── */
.stSuccess { background: rgba(0,229,160,0.08) !important; border-color: var(--success) !important; border-radius: 12px !important; }
.stError   { background: rgba(255,77,109,0.08) !important; border-color: var(--accent3) !important; border-radius: 12px !important; }
.stWarning { background: rgba(255,183,0,0.08) !important; border-color: var(--warning) !important; border-radius: 12px !important; }
.stInfo    { background: rgba(0,212,255,0.08) !important; border-color: var(--accent) !important; border-radius: 12px !important; }

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Custom components ───────────────────────────────── */
.ketu-hero {
    text-align: center;
    padding: 4rem 0 2.5rem;
    position: relative;
}
.ketu-logo {
    font-family: 'Bebas Neue', sans-serif;
    font-size: clamp(3.5rem, 8vw, 7rem);
    letter-spacing: 0.12em;
    background: linear-gradient(135deg, #fff 0%, var(--accent) 50%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    filter: drop-shadow(0 0 30px rgba(0,212,255,0.3));
    animation: heroIn 0.8s cubic-bezier(0.34,1.56,0.64,1) both;
}
.ketu-sub {
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    font-size: 0.9rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.5rem;
    animation: heroIn 0.8s 0.1s cubic-bezier(0.34,1.56,0.64,1) both;
}
.ketu-tagline {
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    color: rgba(226,234,248,0.6);
    margin-top: 0.8rem;
    animation: heroIn 0.8s 0.2s ease both;
}
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 1.5rem;
    animation: fadeUp 0.5s ease both;
    position: relative;
    overflow: hidden;
}
.panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0.4;
}
.panel-accent { border-color: rgba(0,212,255,0.3); box-shadow: var(--glow-blue); }
.panel-purple { border-color: rgba(123,47,255,0.3); box-shadow: var(--glow-purple); }

.question-card {
    background: linear-gradient(135deg, rgba(0,212,255,0.06) 0%, rgba(123,47,255,0.06) 100%);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: var(--radius);
    padding: 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--glow-blue);
    animation: fadeUp 0.4s ease both;
    position: relative;
}
.question-card::after {
    content: '';
    position: absolute;
    top: -1px; left: 10%; right: 10%;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    border-radius: 99px;
}
.q-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.q-text {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(1.1rem, 2.5vw, 1.5rem);
    font-weight: 600;
    line-height: 1.45;
    color: var(--text);
    margin: 0;
}
.feedback-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem;
    margin-top: 1.5rem;
    border-left: 3px solid var(--accent2);
    animation: slideIn 0.4s ease both;
}
.score-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 1rem;
    border-radius: 99px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}
.score-high   { background: rgba(0,229,160,0.12); color: var(--success); border: 1px solid rgba(0,229,160,0.3); }
.score-mid    { background: rgba(255,183,0,0.12);  color: var(--warning); border: 1px solid rgba(255,183,0,0.3); }
.score-low    { background: rgba(255,77,109,0.12); color: var(--accent3); border: 1px solid rgba(255,77,109,0.3); }

.progress-bar-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
}
.verdict-banner {
    text-align: center;
    padding: 3rem 2rem;
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    position: relative;
    overflow: hidden;
    animation: fadeUp 0.6s ease both;
}
.verdict-grade {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 7rem;
    line-height: 1;
    letter-spacing: 0.1em;
}
.timeline-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 12px var(--accent);
    display: inline-block;
    margin-right: 0.5rem;
    animation: pulse 2s ease infinite;
}
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.8rem;
}
.recording-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255,77,109,0.1);
    border: 1px solid rgba(255,77,109,0.3);
    border-radius: 99px;
    padding: 0.4rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent3);
}
.live-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent3);
    animation: pulse 1s ease infinite;
}
.tip-box {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: rgba(0,212,255,0.7);
    margin-top: 0.8rem;
}

@keyframes heroIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.85); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
"""

st.markdown(f"<style>{DESIGN}</style>", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "screen":        "setup",   # setup | interview | results
        "questions":     [],
        "current":       0,
        "scores":        [],
        "feedback_list": [],
        "started":       False,
        "resume_text":   "",
        "jd_text":       "",
        "candidate_name":"",
        "role_title":    "",
        "num_questions": 8,
        "session_start": None,
        "time_per_q":    [],
        "q_start":       None,
        "tts_enabled":   True,
        "show_feedback": False,
        "current_feedback": "",
        "current_score":   0.0,
        "submitted":       False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ============================================================
# API & MODEL SETUP
# ============================================================
@st.cache_resource(show_spinner=False)
# Initialize the OpenAI client (ensure your API key is in st.secrets)
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def transcribe_voice(audio_bytes):
    """Sends audio bytes to OpenAI Whisper and returns the text."""
    if not audio_bytes:
        return None
        
    try:
        # Whisper requires a file object with a proper file name extension
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "interview_response.wav" 
        
        # Call the Whisper API
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return response.text
        
    except Exception as e:
        st.error(f"Failed to transcribe audio: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_embeddings():
    class LocalEmbeddings:
        def __init__(self):
            self.model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            )
        def embed_documents(self, texts):
            return self.model.encode(
                texts, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False, batch_size=32,
            ).tolist()
        def embed_query(self, text):
            return self.model.encode(
                [text], normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False,
            )[0].tolist()
    return LocalEmbeddings()


# ============================================================
# TTS
# ============================================================
def tts_autoplay(text: str):
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        st.markdown(
            f'<audio autoplay style="display:none">'
            f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.caption(f"⚠️ TTS unavailable: {e}")


# ============================================================
# DOCUMENT LOADER
# ============================================================
def load_document(uploaded_file) -> str:
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name
    try:
        if suffix == "pdf":
            docs = PyPDFLoader(temp_path).load()
            return "\n".join(d.page_content for d in docs)
        elif suffix in ("docx", "doc"):
            docs = Docx2txtLoader(temp_path).load()
            return docs[0].page_content if docs else ""
        else:
            return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# QUESTION GENERATION
# ============================================================
def generate_questions(jd: str, resume: str, role: str, n: int, llm) -> list[str]:
    prompt = f"""You are a senior technical interviewer preparing for a {role} interview.

JOB DESCRIPTION:
{jd[:2500]}

CANDIDATE RESUME:
{resume[:2500]}

Generate exactly {n} insightful interview questions tailored to this candidate and role.

Guidelines:
- Start with 2 rapport-building / background questions
- Follow with 3-4 core technical/skill questions based on the JD
- Include 1-2 behavioral (STAR) questions
- End with 1 forward-looking / ambition question
- Questions should feel human, specific, and conversational — NOT generic
- Vary sentence structure and question type
- Return ONLY a numbered list, one question per line, no preamble

Format:
1. Question text here
2. Question text here
...
"""
    response = llm.invoke(prompt)
    questions = []
    for line in response.content.splitlines():
        line = line.strip()
        if line and re.match(r'^\d+[.)\-]', line):
            cleaned = re.sub(r'^\d+[.)\-]\s*', '', line).strip()
            if cleaned:
                questions.append(cleaned)
    return questions[:n]


# ============================================================
# ANSWER EVALUATION
# ============================================================
def evaluate_answer(question: str, answer: str, role: str, llm) -> dict:
    prompt = f"""You are an elite interviewer evaluating a {role} candidate.

QUESTION: {question}

CANDIDATE ANSWER: {answer}

Evaluate the response professionally. Return your evaluation in EXACTLY this format (no extra text):

SCORE: <number from 0 to 10>
STRENGTH: <one sentence on what was good>
WEAKNESS: <one sentence on what was lacking or could improve>
SUGGESTION: <one concrete, actionable improvement tip>
VERDICT: <one-word verdict: Excellent | Good | Average | Needs Work>
"""
    response = llm.invoke(prompt)
    content  = response.content

    result = {
        "score":      0.0,
        "strength":   "",
        "weakness":   "",
        "suggestion": "",
        "verdict":    "Average",
        "raw":        content,
    }
    patterns = {
        "score":      r"SCORE:\s*([\d.]+)",
        "strength":   r"STRENGTH:\s*(.+)",
        "weakness":   r"WEAKNESS:\s*(.+)",
        "suggestion": r"SUGGESTION:\s*(.+)",
        "verdict":    r"VERDICT:\s*(\w[\w\s]*)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            result[key] = float(val) if key == "score" else val

    result["score"] = min(10.0, max(0.0, float(result["score"])))
    return result


# ============================================================
# PLOTLY THEME HELPER
# ============================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit, sans-serif", color="#e2eaf8"),
    xaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
    yaxis=dict(gridcolor="#1a2540", zerolinecolor="#1a2540"),
    margin=dict(t=40, b=40, l=20, r=20),
)


# ============================================================
# SCREEN — SETUP
# ============================================================
def screen_setup():
    # Hero
    st.markdown("""
    <div class="ketu-hero">
        <div class="ketu-logo">KETU AI</div>
        <div class="ketu-sub">Voice · Intelligence · Interview</div>
        <p class="ketu-tagline">Conduct realistic AI-powered interviews with adaptive questioning & instant feedback</p>
    </div>
    """, unsafe_allow_html=True)

    # Check API
    llm = get_llm()
    if llm is None:
        st.error("⚠️ `GROQ_API_KEY` not found in `st.secrets`. Add it to `.streamlit/secrets.toml` to continue.")

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="panel panel-accent">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📋 Job Context</div>', unsafe_allow_html=True)

        st.session_state.candidate_name = st.text_input(
            "Candidate Name (optional)",
            placeholder="e.g. Arjun Mehta",
            value=st.session_state.candidate_name,
        )
        st.session_state.role_title = st.text_input(
            "Role / Job Title *",
            placeholder="e.g. Senior Backend Engineer",
            value=st.session_state.role_title,
        )
        st.session_state.jd_text = st.text_area(
            "Job Description *",
            height=260,
            placeholder="Paste the full job description here…",
            value=st.session_state.jd_text,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel panel-purple">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📄 Candidate Resume</div>', unsafe_allow_html=True)

        resume_file = st.file_uploader(
            "Upload Resume (PDF, DOCX, TXT)",
            type=["pdf", "docx", "doc", "txt"],
            label_visibility="collapsed",
        )
        if resume_file:
            with st.spinner("Reading resume…"):
                st.session_state.resume_text = load_document(resume_file)
            words = len(st.session_state.resume_text.split())
            st.success(f"✅ Loaded — {words:,} words extracted")
            with st.expander("Preview extracted text"):
                st.text(st.session_state.resume_text[:800] + "…")

        st.markdown("---")
        st.markdown('<div class="section-label">⚙️ Interview Settings</div>', unsafe_allow_html=True)

        st.session_state.num_questions = st.slider(
            "Number of Questions", min_value=3, max_value=15,
            value=st.session_state.num_questions,
        )
        st.session_state.tts_enabled = st.toggle(
            "🔊 Read questions aloud (TTS)", value=st.session_state.tts_enabled,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Launch button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀  Launch Interview Session", use_container_width=True):
            if not st.session_state.jd_text.strip():
                st.error("Please paste a job description.")
            elif not st.session_state.resume_text.strip():
                st.error("Please upload a resume.")
            elif not st.session_state.role_title.strip():
                st.error("Please enter the role / job title.")
            elif llm is None:
                st.error("GROQ_API_KEY missing.")
            else:
                with st.spinner("🤖 Generating tailored interview questions…"):
                    qs = generate_questions(
                        st.session_state.jd_text,
                        st.session_state.resume_text,
                        st.session_state.role_title,
                        st.session_state.num_questions,
                        llm,
                    )
                if not qs:
                    st.error("Could not generate questions. Check your API key.")
                else:
                    st.session_state.questions     = qs
                    st.session_state.current       = 0
                    st.session_state.scores        = []
                    st.session_state.feedback_list = []
                    st.session_state.time_per_q    = []
                    st.session_state.started       = True
                    st.session_state.session_start = time.time()
                    st.session_state.q_start       = time.time()
                    st.session_state.submitted     = False
                    st.session_state.show_feedback = False
                    st.session_state.screen        = "interview"
                    st.rerun()


# ============================================================
# SCREEN — INTERVIEW
# ============================================================
def screen_interview():
    llm       = get_llm()
    questions = st.session_state.questions
    idx       = st.session_state.current
    n         = len(questions)

    # Finished?
    if idx >= n:
        st.session_state.screen = "results"
        st.rerun()
        return

    q       = questions[idx]
    elapsed = int(time.time() - st.session_state.q_start) if st.session_state.q_start else 0

    # ── Top Status Bar ─────────────────────────────────────
    sb1, sb2, sb3, sb4 = st.columns([3, 1, 1, 1])
    with sb1:
        pct = idx / n
        st.progress(pct)
        st.caption(f"Question {idx + 1} of {n}  ·  {int(pct*100)}% complete")
    with sb2:
        avg = (sum(s.get("score", 0) for s in st.session_state.scores) / len(st.session_state.scores)) if st.session_state.scores else 0
        st.metric("Avg Score", f"{avg:.1f}/10" if st.session_state.scores else "—")
    with sb3:
        total_elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
        mins, secs = divmod(total_elapsed, 60)
        st.metric("Session Time", f"{mins:02d}:{secs:02d}")
    with sb4:
        colour = "#00e5a0" if elapsed < 60 else "#ffb700" if elapsed < 120 else "#ff4d6d"
        em, es = divmod(elapsed, 60)
        st.markdown(
            f'<div style="text-align:center;font-family:\'Bebas Neue\',sans-serif;font-size:2rem;'
            f'color:{colour};letter-spacing:0.06em">{em:02d}:{es:02d}</div>'
            f'<div style="text-align:center;font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
            f'color:var(--muted)">THIS Q</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Question Card ───────────────────────────────────────
    name_prefix = f"{st.session_state.candidate_name}, " if st.session_state.candidate_name else ""
    st.markdown(f"""
    <div class="question-card">
        <div class="q-number">
            ◆ {st.session_state.role_title} Interview &nbsp;·&nbsp; Q{idx + 1}/{n}
        </div>
        <p class="q-text">{q}</p>
    </div>
    """, unsafe_allow_html=True)

    # Auto-read on first render
    tts_key = f"_tts_{idx}"
    if st.session_state.tts_enabled and tts_key not in st.session_state:
        tts_autoplay(q)
        st.session_state[tts_key] = True

    # Not yet submitted
    if not st.session_state.submitted:
        c_btn1, c_btn2, _ = st.columns([1, 1, 3])
        with c_btn1:
            if st.button("🔊 Re-read Question", key=f"reread_{idx}"):
                tts_autoplay(q)
        with c_btn2:
            if st.session_state.scores:
                last_score = st.session_state.scores[-1].get("score", 0)
            else:
                last_score = 0   
                color = "#00e5a0" if last_score >= 7 else "#ffb700" if last_score >= 5 else "#ff4d6d"
                st.markdown(
                    f'<div class="stat-chip" style="color:{color}">Last: {last_score}/10</div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-label" style="margin-top:1.2rem">✍️ Your Answer</div>', unsafe_allow_html=True)

        # Audio input (optional)
        if HAS_AUDIO_RECORDER:
            st.markdown(
                '<div class="recording-indicator"><div class="live-dot"></div> Audio recording available</div>',
                unsafe_allow_html=True,
            )
            audio_bytes = audio_recorder(
                text="", recording_color="#ff4d6d", neutral_color="#1a2540",
                icon_name="microphone", icon_size="2x", key=f"audio_{idx}",
            )
            if audio_bytes:
                st.info("🎤 Audio captured — transcription requires Whisper integration. Type your answer below.")

        answer = st.text_area(
            "Type your answer here…",
            key=f"answer_{idx}",
            height=200,
            placeholder=f"Answer question {idx + 1} as you would in a real interview. Be specific and use examples where possible.",
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="tip-box">💡 Tip: Use the STAR method (Situation · Task · Action · Result) '
            'for behavioral questions.</div>',
            unsafe_allow_html=True,
        )

        col_submit, col_skip = st.columns([3, 1])
        with col_submit:
            if st.button("✅  Submit Answer & Get Feedback", key=f"submit_{idx}", use_container_width=True):
                if not answer.strip():
                    st.warning("Please write your answer before submitting.")
                else:
                    with st.spinner("🤖 Analysing your response…"):
                        result = evaluate_answer(q, answer, st.session_state.role_title, llm)
                    st.session_state.current_feedback = result
                    st.session_state.current_score    = result["score"]
                    st.session_state.submitted        = True
                    elapsed_q = time.time() - st.session_state.q_start
                    st.session_state.time_per_q.append(round(elapsed_q, 1))
                    st.session_state.scores.append({
                        "Question": idx + 1,
                        "Score":    result["score"],
                        "Verdict":  result["verdict"],
                        "Q_text":   q[:60] + ("…" if len(q) > 60 else ""),
                    })
                    st.session_state.feedback_list.append(result)
                    st.rerun()
        with col_skip:
            if st.button("Skip →", key=f"skip_{idx}"):
                st.session_state.time_per_q.append(0)
                st.session_state.scores.append({
                    "Question": idx + 1, "Score": 0.0,
                    "Verdict": "Skipped", "Q_text": q[:60],
                })
                st.session_state.feedback_list.append({
                    "score": 0.0, "strength": "—", "weakness": "Skipped",
                    "suggestion": "Attempt every question in a real interview.",
                    "verdict": "Skipped", "raw": "",
                })
                st.session_state.current  += 1
                st.session_state.q_start   = time.time()
                st.session_state.submitted = False
                st.rerun()

    # Submitted — show feedback
    else:
        result = st.session_state.current_feedback
        score  = result["score"]
        verdict = result.get("verdict", "Average")
        score_class = "score-high" if score >= 7 else "score-mid" if score >= 5 else "score-low"
        score_icon  = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"

        st.markdown(f"""
        <div class="feedback-card">
            <span class="score-pill {score_class}">{score_icon} {score}/10 · {verdict}</span>
            <div style="display:grid;gap:0.8rem">
                <div>
                    <div class="section-label">✅ Strength</div>
                    <div style="font-size:0.95rem">{result.get("strength","")}</div>
                </div>
                <div>
                    <div class="section-label">⚠️ Weakness</div>
                    <div style="font-size:0.95rem">{result.get("weakness","")}</div>
                </div>
                <div>
                    <div class="section-label">💡 Suggestion</div>
                    <div style="font-size:0.95rem;color:rgba(0,212,255,0.8)">{result.get("suggestion","")}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # TTS for feedback
        if st.session_state.tts_enabled:
            fb_summary = f"Score {score} out of 10. {result.get('strength','')} {result.get('suggestion','')}"
            tts_autoplay(fb_summary)

        st.markdown("<br>", unsafe_allow_html=True)
        col_next, col_review = st.columns([3, 1])
        with col_next:
            label = "Next Question ➜" if idx + 1 < n else "🏁 View Results"
            if st.button(label, key=f"next_{idx}", use_container_width=True):
                st.session_state.current  += 1
                st.session_state.q_start   = time.time()
                st.session_state.submitted = False
                st.session_state.show_feedback = False
                st.rerun()
        with col_review:
            if st.button("📋 Raw", key=f"raw_{idx}"):
                st.session_state.show_feedback = not st.session_state.show_feedback

        if st.session_state.show_feedback:
            with st.expander("Full AI Evaluation", expanded=True):
                st.code(result["raw"], language="markdown")


# ============================================================
# SCREEN — RESULTS
# ============================================================
def screen_results():
    scores  = st.session_state.scores
    feedbacks = st.session_state.feedback_list
    n       = len(scores)

    if not scores:
        st.warning("No data to show.")
        if st.button("← Back"):
            st.session_state.screen = "setup"
            st.rerun()
        return

    avg   = sum(s["Score"] for s in scores) / n
    best  = max(s["Score"] for s in scores)
    worst = min(s["Score"] for s in scores)
    total_time = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
    mins, secs = divmod(total_time, 60)
    avg_time = round(sum(st.session_state.time_per_q) / len(st.session_state.time_per_q), 1) if st.session_state.time_per_q else 0

    # Grade
    if avg >= 8.5: grade, gcol, gverdict = "S", "#ffd700", "Outstanding"
    elif avg >= 7:  grade, gcol, gverdict = "A", "#00e5a0", "Excellent"
    elif avg >= 5.5:grade, gcol, gverdict = "B", "#00d4ff", "Good"
    elif avg >= 4:  grade, gcol, gverdict = "C", "#ffb700", "Average"
    else:            grade, gcol, gverdict = "D", "#ff4d6d", "Needs Improvement"

    name = st.session_state.candidate_name or "Candidate"
    role = st.session_state.role_title

    # Verdict banner
    st.markdown(f"""
    <div class="verdict-banner">
        <div class="verdict-grade" style="color:{gcol};text-shadow:0 0 60px {gcol}66">{grade}</div>
        <div style="font-family:'Outfit',sans-serif;font-size:1.4rem;font-weight:600;margin-top:0.3rem">
            {gverdict} Performance
        </div>
        <div style="font-family:'JetBrains Mono',monospace;color:var(--muted);font-size:0.82rem;margin-top:0.5rem">
            {name} · {role} · {datetime.now().strftime('%d %b %Y, %H:%M')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🏆 Avg Score",   f"{avg:.1f}/10")
    m2.metric("⬆️ Best",        f"{best:.0f}/10")
    m3.metric("⬇️ Lowest",      f"{worst:.0f}/10")
    m4.metric("⏱ Session",      f"{mins}m {secs}s")
    m5.metric("📊 Avg/Q",       f"{avg_time}s")

    st.markdown("---")

    # Charts
    c_left, c_right = st.columns(2, gap="large")
    with c_left:
        df = pd.DataFrame(scores)
        fig = go.Figure()
        # Area fill
        fig.add_trace(go.Scatter(
            x=df["Question"], y=df["Score"],
            fill="tozeroy",
            fillcolor="rgba(0,212,255,0.06)",
            line=dict(color="#00d4ff", width=2.5),
            mode="lines+markers",
            marker=dict(size=9, color="#00d4ff", line=dict(width=2, color="#fff")),
            name="Score",
        ))
        # 7-point benchmark
        fig.add_hline(y=7, line=dict(color="#7b2fff", dash="dot", width=1.5),
                      annotation_text="Target (7)", annotation_font_color="#7b2fff")
        fig.update_layout(
            title="Performance Trend", yaxis_range=[0, 10],
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        verdict_counts = {}
        for s in scores:
            v = s["Verdict"]
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        vc_df = pd.DataFrame(list(verdict_counts.items()), columns=["Verdict", "Count"])
        color_map = {
            "Excellent":    "#00e5a0",
            "Good":         "#00d4ff",
            "Average":      "#ffb700",
            "Needs Work":   "#ff4d6d",
            "Skipped":      "#4a5a7a",
        }
        fig2 = px.pie(
            vc_df, names="Verdict", values="Count",
            color="Verdict",
            color_discrete_map=color_map,
            hole=0.55,
            title="Answer Quality Distribution",
        )
        fig2.update_traces(textfont_color="#e2eaf8")
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # Radar chart — scoring by question
    st.markdown('<div class="section-label">📈 Score Radar</div>', unsafe_allow_html=True)
    theta = [f"Q{s['Question']}" for s in scores]
    r_vals = [s["Score"] for s in scores]
    fig3 = go.Figure(go.Scatterpolar(
        r=r_vals + [r_vals[0]],
        theta=theta + [theta[0]],
        fill="toself",
        fillcolor="rgba(123,47,255,0.12)",
        line=dict(color="#7b2fff", width=2),
        marker=dict(color="#00d4ff", size=7),
    ))
    fig3.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 10], color="#4a5a7a",
                            gridcolor="#1a2540", tickfont=dict(color="#4a5a7a")),
            angularaxis=dict(color="#4a5a7a", gridcolor="#1a2540"),
        ),
        showlegend=False,
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Per-question breakdown
    st.markdown('<div class="section-label">📋 Full Breakdown</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    for i, (s, fb) in enumerate(zip(scores, feedbacks)):
        score_val   = s["Score"]
        score_class = "score-high" if score_val >= 7 else "score-mid" if score_val >= 5 else "score-low"
        icon        = "✅" if score_val >= 7 else "⚠️" if score_val >= 5 else "❌"
        t           = st.session_state.time_per_q[i] if i < len(st.session_state.time_per_q) else "—"
        st.markdown(f"""
        <div style="border-bottom:1px solid var(--border);padding:0.9rem 0;display:flex;
                    justify-content:space-between;align-items:flex-start;gap:1rem">
            <div>
                <div style="font-weight:600;font-size:0.93rem">{icon} Q{s['Question']}: {s['Q_text']}</div>
                <div style="font-size:0.82rem;color:var(--muted);margin-top:0.3rem">{fb.get('suggestion','')}</div>
            </div>
            <div style="white-space:nowrap;text-align:right">
                <span class="score-pill {score_class}" style="font-size:0.78rem">{score_val}/10</span>
                <div style="font-size:0.72rem;color:var(--muted);margin-top:0.3rem">{t}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # AI Overall Summary
    llm = get_llm()
    if llm and st.button("🤖 Generate AI Interview Report", use_container_width=True):
        with st.spinner("Writing full report…"):
            qa_pairs = "\n".join(
                f"Q{s['Question']}: {s['Q_text']} → Score {s['Score']}/10, Verdict: {s['Verdict']}"
                for s in scores
            )
            summary_prompt = f"""
You are a senior hiring manager writing a post-interview assessment for {name} applying for {role}.

Session Results:
{qa_pairs}
Average Score: {avg:.1f}/10
Grade: {grade} ({gverdict})

Write a professional 3-paragraph interview assessment:
1. Overall impression and key strengths
2. Areas that need development
3. Hiring recommendation with reasoning

Be specific, professional, and constructive.
"""
            resp = llm.invoke(summary_prompt)
            st.markdown('<div class="feedback-card">', unsafe_allow_html=True)
            st.markdown("### 📝 AI Interview Report")
            st.write(resp.content)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄  New Interview", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    with c2:
        if st.button("⚙️  Change Settings", use_container_width=True):
            st.session_state.screen  = "setup"
            st.session_state.started = False
            st.rerun()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;
                letter-spacing:0.1em;color:#00d4ff;margin-bottom:0.5rem">KETU AI</div>
    """, unsafe_allow_html=True)
    st.caption("Voice · Intelligence · Interview")
    st.markdown("---")

    screen = st.session_state.get("screen", "setup")
    if screen == "interview":
        q_idx = st.session_state.current
        n_q   = len(st.session_state.questions)
        st.metric("Progress", f"{q_idx}/{n_q}")
        st.progress(q_idx / n_q if n_q else 0)
        if st.session_state.scores:
            avg = sum(s["Score"] for s in st.session_state.scores) / len(st.session_state.scores)
            st.metric("Running Avg", f"{avg:.1f}/10")
        st.markdown("---")

    st.markdown("##### ⚡ Features")
    st.markdown("""
    - 🤖 Llama 3.3 70B via Groq
    - 🎙️ Voice input + TTS output
    - 📄 PDF / DOCX resume parsing
    - 🧠 RAG-powered question gen
    - 📊 Real-time scoring & radar
    - 📝 AI post-interview report
    """)
    st.markdown("---")
    st.caption(f"Session: {datetime.now().strftime('%H:%M · %d %b %Y')}")


# ============================================================
# ROUTER
# ============================================================
screen = st.session_state.screen
if screen == "setup":
    screen_setup()
elif screen == "interview":
    screen_interview()
elif screen == "results":
    screen_results()
