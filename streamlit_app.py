# ============================================================
# CRITICAL ENV VARS — must be first, before any other imports
# ============================================================
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
class _TorchClassesPatch:
    def __getattr__(self, name):
        if name in ["__path__", "_path"]:
            return []
        raise AttributeError(name)
torch.classes = _TorchClassesPatch()

# ── Standard library ───────────────────────────────────────
import re, io, time, base64, tempfile, json, random
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

try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="KETU AI · Elite Interviewer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DESIGN SYSTEM
# ============================================================
DESIGN = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=Cabinet+Grotesk:wght@400;500;700;800;900&display=swap');

:root {
    --bg:          #020408;
    --surface:     #060c14;
    --surface2:    #0a1220;
    --surface3:    #0f1a2e;
    --border:      #162035;
    --border2:     #1e2d47;
    --cyan:        #00e5ff;
    --violet:      #8b5cf6;
    --rose:        #f43f5e;
    --emerald:     #10b981;
    --amber:       #f59e0b;
    --text:        #e8f0fe;
    --text2:       #8899bb;
    --text3:       #3d5278;
    --radius:      16px;
    --radius-lg:   24px;
    --glow-cyan:   0 0 60px rgba(0,229,255,0.1);
    --glow-violet: 0 0 60px rgba(139,92,246,0.1);
    --glow-rose:   0 0 60px rgba(244,63,94,0.1);
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 20% -10%, rgba(0,229,255,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 110%, rgba(139,92,246,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 50% 50%, rgba(15,26,46,0.8) 0%, transparent 100%),
        var(--bg) !important;
}

[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Grid overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(0,229,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,229,255,0.015) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none; z-index: 0;
}

/* ── Typography ── */
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
code,pre  { font-family: 'DM Mono', monospace !important; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 1.8rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 30px rgba(0,229,255,0.15), inset 0 0 30px rgba(0,229,255,0.03) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(1px) !important; }

/* ── Text areas ── */
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.875rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 25px rgba(0,229,255,0.08) !important;
    outline: none !important;
}

/* ── Inputs ── */
.stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
}
.stTextInput input:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.08) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--violet) !important; }

/* ── Progress ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
    border-radius: 99px !important;
    box-shadow: 0 0 15px rgba(0,229,255,0.3) !important;
}
.stProgress > div > div {
    background: var(--border) !important;
    border-radius: 99px !important;
    height: 4px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.5rem !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--cyan) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    color: var(--text3) !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Alerts ── */
.stSuccess { background: rgba(16,185,129,0.07) !important; border-color: var(--emerald) !important; border-radius: 12px !important; }
.stError   { background: rgba(244,63,94,0.07) !important;  border-color: var(--rose) !important;    border-radius: 12px !important; }
.stWarning { background: rgba(245,158,11,0.07) !important; border-color: var(--amber) !important;   border-radius: 12px !important; }
.stInfo    { background: rgba(0,229,255,0.05) !important;  border-color: var(--cyan) !important;    border-radius: 12px !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ════════════════════════════════════════
   CUSTOM COMPONENTS
════════════════════════════════════════ */

/* Hero */
.hero-wrap {
    text-align: center;
    padding: 5rem 0 3rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--cyan);
    margin-bottom: 1.2rem;
    opacity: 0.8;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(4rem, 10vw, 8rem);
    font-weight: 800;
    line-height: 0.92;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #fff 0%, var(--cyan) 45%, var(--violet) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    filter: drop-shadow(0 0 40px rgba(0,229,255,0.2));
    animation: heroReveal 1s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.hero-sub {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 1.15rem;
    color: var(--text2);
    margin-top: 1.2rem;
    max-width: 560px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
    animation: heroReveal 1s 0.15s cubic-bezier(0.16, 1, 0.3, 1) both;
}

/* Panels */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    animation: panelIn 0.5s ease both;
}
.panel-glow-cyan  { border-color: rgba(0,229,255,0.2);   box-shadow: var(--glow-cyan); }
.panel-glow-violet{ border-color: rgba(139,92,246,0.2);  box-shadow: var(--glow-violet); }
.panel-glow-rose  { border-color: rgba(244,63,94,0.2);   box-shadow: var(--glow-rose); }
.panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(0,229,255,0.5) 50%, transparent 100%);
    opacity: 0.5;
}

/* Section label */
.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* Interviewer avatar */
.interviewer-wrap {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.5rem 2rem;
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: var(--radius-lg);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.interviewer-wrap::before {
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--cyan), var(--violet));
}
.avatar-ring {
    width: 64px; height: 64px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--surface3), var(--surface2));
    border: 2px solid var(--cyan);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.8rem;
    box-shadow: 0 0 25px rgba(0,229,255,0.2);
    flex-shrink: 0;
    position: relative;
}
.avatar-ring.speaking::after {
    content: '';
    position: absolute; inset: -6px;
    border-radius: 50%;
    border: 2px solid var(--cyan);
    animation: speakPulse 1.5s ease infinite;
    opacity: 0.5;
}
.interviewer-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: var(--text);
}
.interviewer-status {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: var(--cyan);
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--emerald);
    animation: blink 2s ease infinite;
}
.interviewer-speech {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 0.95rem;
    color: var(--text2);
    line-height: 1.55;
    flex: 1;
    font-style: italic;
}

/* Question card */
.q-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.04) 0%, rgba(139,92,246,0.04) 100%);
    border: 1px solid rgba(0,229,255,0.18);
    border-radius: var(--radius-lg);
    padding: 2.5rem;
    margin: 1.5rem 0;
    position: relative;
    animation: panelIn 0.4s ease both;
}
.q-card::after {
    content: '';
    position: absolute; top: -1px; left: 8%; right: 8%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    border-radius: 99px;
}
.q-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--cyan);
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    opacity: 0.8;
}
.q-text {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.15rem, 2.5vw, 1.55rem);
    font-weight: 600;
    line-height: 1.4;
    color: var(--text);
    margin: 0;
}
.q-type-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.28rem 0.8rem;
    border-radius: 99px;
    font-family: 'DM Mono', monospace;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    margin-top: 1rem;
}
.badge-technical  { background: rgba(139,92,246,0.12); color: var(--violet); border: 1px solid rgba(139,92,246,0.3); }
.badge-behavioral { background: rgba(0,229,255,0.08);  color: var(--cyan);   border: 1px solid rgba(0,229,255,0.25); }
.badge-situational{ background: rgba(245,158,11,0.1);  color: var(--amber);  border: 1px solid rgba(245,158,11,0.3); }
.badge-rapport    { background: rgba(16,185,129,0.1);  color: var(--emerald);border: 1px solid rgba(16,185,129,0.3); }

/* Waveform */
.waveform-wrap {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 4px;
    height: 48px;
    margin: 1rem 0;
}
.wave-bar {
    width: 4px;
    border-radius: 99px;
    background: var(--cyan);
    opacity: 0.8;
    animation: waveDance var(--speed) ease-in-out infinite alternate;
}

/* Feedback card */
.feedback-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: var(--radius-lg);
    padding: 2rem;
    margin-top: 1.5rem;
    border-left: 3px solid var(--violet);
    animation: slideRight 0.4s ease both;
}
.feedback-section {
    margin-bottom: 1.2rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}
.feedback-section:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.feedback-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.label-strength   { color: var(--emerald); }
.label-weakness   { color: var(--rose); }
.label-suggestion { color: var(--amber); }
.label-tone       { color: var(--violet); }
.feedback-text {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 0.95rem;
    color: var(--text2);
    line-height: 1.55;
}

/* Score display */
.score-display {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.score-circle {
    width: 80px; height: 80px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    flex-shrink: 0;
    position: relative;
}
.score-high { background: rgba(16,185,129,0.1); color: var(--emerald); border: 2px solid var(--emerald); box-shadow: 0 0 25px rgba(16,185,129,0.2); }
.score-mid  { background: rgba(245,158,11,0.1); color: var(--amber);   border: 2px solid var(--amber);   box-shadow: 0 0 25px rgba(245,158,11,0.2); }
.score-low  { background: rgba(244,63,94,0.1);  color: var(--rose);    border: 2px solid var(--rose);    box-shadow: 0 0 25px rgba(244,63,94,0.2); }
.verdict-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
}
.verdict-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--text3);
    margin-top: 0.2rem;
}

/* Tone chips */
.tone-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
.tone-chip {
    padding: 0.25rem 0.75rem;
    border-radius: 99px;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    background: rgba(139,92,246,0.1);
    color: var(--violet);
    border: 1px solid rgba(139,92,246,0.25);
}

/* Recording strip */
.rec-strip {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.8rem 1.2rem;
    background: rgba(244,63,94,0.06);
    border: 1px solid rgba(244,63,94,0.2);
    border-radius: 12px;
    margin-bottom: 1rem;
}
.rec-label {
    display: flex; align-items: center; gap: 0.5rem;
    font-family: 'DM Mono', monospace; font-size: 0.78rem; color: var(--rose);
}
.rec-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--rose); animation: blink 1s ease infinite; }

/* Results */
.result-hero {
    text-align: center;
    padding: 4rem 2rem;
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
}
.result-grade {
    font-family: 'Syne', sans-serif;
    font-size: clamp(5rem, 12vw, 9rem);
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
}
.result-grade-A { background: linear-gradient(135deg, var(--emerald), var(--cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-grade-B { background: linear-gradient(135deg, var(--cyan), var(--violet)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-grade-C { background: linear-gradient(135deg, var(--amber), #f97316); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-grade-D { background: linear-gradient(135deg, var(--rose), #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--text3);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 1rem;
}
.result-tagline {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-size: 1.2rem;
    color: var(--text2);
    margin-top: 0.5rem;
}

/* Q timeline in results */
.q-timeline-item {
    display: flex; gap: 1.2rem; align-items: flex-start;
    padding: 1.2rem 0;
    border-bottom: 1px solid var(--border);
}
.q-timeline-item:last-child { border-bottom: none; }
.q-timeline-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text3);
    min-width: 28px;
    padding-top: 0.15rem;
}
.q-timeline-content { flex: 1; }
.q-timeline-q {
    font-family: 'Cabinet Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text);
    margin-bottom: 0.4rem;
}
.q-timeline-score {
    display: inline-flex; align-items: center; gap: 0.35rem;
    font-family: 'DM Mono', monospace; font-size: 0.72rem;
    padding: 0.2rem 0.6rem; border-radius: 99px;
}

/* Tip box */
.tip-box {
    background: rgba(0,229,255,0.03);
    border: 1px solid rgba(0,229,255,0.12);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: rgba(0,229,255,0.6);
    margin-top: 1rem;
    line-height: 1.6;
}

/* Follow-up indicator */
.followup-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    background: rgba(244,63,94,0.08);
    border: 1px solid rgba(244,63,94,0.2);
    border-radius: 99px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--rose);
    letter-spacing: 0.06em;
    margin-bottom: 0.8rem;
}

/* ── Animations ── */
@keyframes heroReveal {
    from { opacity: 0; transform: translateY(30px); filter: blur(8px); }
    to   { opacity: 1; transform: translateY(0);    filter: blur(0); }
}
@keyframes panelIn {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideRight {
    from { opacity: 0; transform: translateX(-12px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}
@keyframes speakPulse {
    0%   { transform: scale(1);   opacity: 0.5; }
    100% { transform: scale(1.3); opacity: 0; }
}
@keyframes waveDance {
    from { height: 6px; }
    to   { height: var(--max-h); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
"""
st.markdown(f"<style>{DESIGN}</style>", unsafe_allow_html=True)

# Waveform animation injection
WAVEFORM_HTML = """
<div class="waveform-wrap" id="waveform">
""" + "".join([
    f'<div class="wave-bar" style="--speed:{random.uniform(0.4,0.9):.2f}s;--max-h:{random.randint(16,44)}px;height:{random.randint(6,20)}px;opacity:{random.uniform(0.5,0.9):.2f};background:{"var(--cyan)" if i % 3 != 2 else "var(--violet)"};"></div>'
    for i in range(32)
]) + "</div>"

# ============================================================
# INTERVIEWER PERSONA
# ============================================================
INTERVIEWER = {
    "name": "Ketu",
    "title": "Senior AI Interviewer · KETU",
    "avatar": "🤖",
    "greetings": [
        "Great to meet you! I've reviewed your profile carefully — I'm genuinely excited to learn more about your journey.",
        "Welcome! I've gone through your background and the role requirements. Let's have an honest conversation.",
        "Hello! I've prepared some targeted questions for you. Take your time — there are no trick questions here.",
        "Hi there! I want this to feel like a real conversation, not an interrogation. Ready when you are.",
    ],
    "transitions": [
        "Interesting — thanks for sharing that. Let me move to the next area.",
        "Got it, I appreciate your openness. Moving on…",
        "That's helpful context. Let's continue.",
        "Thank you. Here's my next question for you.",
        "I see — noted. Let's keep going.",
    ],
    "encouragements": [
        "You're doing well — keep that level of detail.",
        "Good answer. I appreciate the specificity.",
        "That's exactly the kind of thinking we look for.",
    ],
}

QUESTION_TYPES = {
    "rapport":     ("💬", "badge-rapport",     "Rapport"),
    "technical":   ("⚙️", "badge-technical",   "Technical"),
    "behavioral":  ("🧠", "badge-behavioral",  "Behavioral"),
    "situational": ("🎯", "badge-situational", "Situational"),
    "ambition":    ("🚀", "badge-rapport",     "Forward-looking"),
}

# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "screen":           "setup",
        "questions":        [],
        "q_types":          [],
        "current":          0,
        "scores":           [],
        "feedback_list":    [],
        "resume_text":      "",
        "jd_text":          "",
        "candidate_name":   "",
        "role_title":       "",
        "num_questions":    8,
        "session_start":    None,
        "q_start":          None,
        "time_per_q":       [],
        "tts_enabled":      True,
        "submitted":        False,
        "current_feedback": None,
        "nova_message":     "",
        "is_followup":      False,
        "followup_count":   0,
        "transcript":       [],   # full conversation log
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============================================================
# LLM SETUP
# ============================================================
@st.cache_resource(show_spinner=False)
def get_llm():
    try:
        return ChatGroq(
            temperature=0.4,
            model_name="llama-3.3-70b-versatile",
            api_key=st.secrets["GROQ_API_KEY"],
        )
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_embeddings():
    class LocalEmbeddings:
        def __init__(self):
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        def embed_documents(self, texts):
            return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()
        def embed_query(self, text):
            return self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].tolist()
    return LocalEmbeddings()

# ============================================================
# TTS
# ============================================================
def tts_autoplay(text: str):
    if not st.session_state.get("tts_enabled", True):
        return
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        st.markdown(
            f'<audio autoplay style="display:none"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass

# ============================================================
# TRANSCRIPTION (Groq Whisper)
# ============================================================
def transcribe_voice(audio_bytes: bytes) -> str:
    try:
        from groq import Groq
        gc = Groq(api_key=st.secrets["GROQ_API_KEY"])
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.wav"
        result = gc.audio.transcriptions.create(model="whisper-large-v3-turbo", file=buf)
        return result.text.strip()
    except Exception as e:
        st.warning(f"⚠️ Transcription failed: {e}")
        return ""

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
# QUESTION GENERATION (with types)
# ============================================================
def generate_questions(jd: str, resume: str, role: str, n: int, llm) -> tuple[list[str], list[str]]:
    prompt = f"""You are Nova, a senior AI interviewer at KETU. You are preparing for a {role} interview.

JOB DESCRIPTION:
{jd[:2500]}

CANDIDATE RESUME:
{resume[:2500]}

Generate exactly {n} insightful, human, conversational interview questions tailored specifically to this candidate and role.

Structure:
- Q1-2: rapport (warm, personal, background-focused)
- Q3-{max(4, n-3)}: technical (deep skill/knowledge based on JD & resume)
- Q{max(4, n-3)+1}-{n-1}: behavioral (STAR-method, real scenarios)
- Q{n}: ambition (forward-looking, growth-oriented)

Rules:
- Sound like a real interviewer, not a template
- Reference specific skills or experiences from the resume
- Vary sentence structure — some short, some multi-part
- NO preamble, NO explanations, just the questions

Return ONLY this JSON (no markdown, no extra text):
{{
  "questions": ["question 1", "question 2", ...],
  "types": ["rapport", "rapport", "technical", ...]
}}

Types must be one of: rapport, technical, behavioral, situational, ambition
"""
    response = llm.invoke(prompt)
    try:
        raw = response.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)
        questions = data.get("questions", [])[:n]
        types = data.get("types", ["technical"] * n)[:n]
        # Pad types if needed
        while len(types) < len(questions):
            types.append("technical")
        return questions, types
    except Exception:
        # Fallback: parse numbered list
        questions, types = [], []
        for line in response.content.splitlines():
            line = line.strip()
            if line and re.match(r'^\d+[.)\-]', line):
                cleaned = re.sub(r'^\d+[.)\-]\s*', '', line).strip()
                if cleaned:
                    questions.append(cleaned)
                    types.append("technical")
        return questions[:n], types[:n]

# ============================================================
# ADVANCED ANSWER EVALUATION
# ============================================================
def evaluate_answer(question: str, answer: str, role: str, q_type: str, llm) -> dict:
    prompt = f"""You are Nova, an elite AI interviewer evaluating a {role} candidate.
Question type: {q_type}

QUESTION: {question}

CANDIDATE ANSWER: {answer}

Evaluate with nuance and precision. Return ONLY this JSON (no markdown):
{{
  "score": <0-10 float>,
  "verdict": "<Exceptional|Strong|Solid|Average|Weak>",
  "strength": "<specific 1-sentence strength>",
  "weakness": "<specific 1-sentence gap or missing element>",
  "suggestion": "<concrete, actionable 1-sentence tip>",
  "tone_signals": ["<signal1>", "<signal2>", "<signal3>"],
  "needs_followup": <true|false>,
  "followup_question": "<a natural follow-up question if score < 6 or answer was vague, else empty string>",
  "nova_reaction": "<1 short sentence Nova would naturally say after this answer, in first person, conversational>"
}}

tone_signals: pick 3 from [Confident, Structured, Vague, Concise, Detailed, Nervous, Passionate, Hesitant, Analytical, Creative, Experienced, Rambling, Thoughtful, Unprepared]
"""
    response = llm.invoke(prompt)
    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        result = json.loads(raw)
        result["score"] = min(10.0, max(0.0, float(result.get("score", 5))))
        return result
    except Exception:
        # Fallback parse
        content = response.content
        result = {
            "score": 5.0, "verdict": "Average",
            "strength": "Answer was provided.",
            "weakness": "Could not fully evaluate.",
            "suggestion": "Try to be more specific and structured.",
            "tone_signals": ["Thoughtful"],
            "needs_followup": False,
            "followup_question": "",
            "nova_reaction": "Thanks for sharing that.",
        }
        sm = re.search(r'"score"\s*:\s*([\d.]+)', content)
        if sm:
            result["score"] = min(10.0, max(0.0, float(sm.group(1))))
        return result

# ============================================================
# GENERATE POST-INTERVIEW SUMMARY
# ============================================================
def generate_summary(feedback_list: list, role: str, candidate_name: str, avg_score: float, llm) -> str:
    qa_pairs = "\n\n".join([
        f"Q{i+1}: {item['q']}\nAnswer: {item['a'][:300]}...\nScore: {item['eval']['score']}/10 — {item['eval']['verdict']}"
        for i, item in enumerate(feedback_list)
    ])
    prompt = f"""You are Nova, a senior AI interviewer. Write a professional post-interview assessment for {candidate_name or 'the candidate'} applying for {role}.

Interview data:
{qa_pairs}

Overall score: {avg_score:.1f}/10

Write a 3-paragraph assessment:
1. Overall impression and performance summary
2. Key strengths demonstrated across the interview
3. Areas for development and specific recommendations

Be honest, specific, and constructive. Professional but human tone. No bullet points — flowing prose only.
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# ============================================================
# HELPERS
# ============================================================
def score_class(score: float) -> str:
    return "score-high" if score >= 7 else "score-mid" if score >= 5 else "score-low"

def grade_letter(avg: float) -> str:
    if avg >= 8.5: return "A+"
    if avg >= 7.5: return "A"
    if avg >= 6.5: return "B+"
    if avg >= 5.5: return "B"
    if avg >= 4.5: return "C"
    return "D"

def grade_class(g: str) -> str:
    if g.startswith("A"): return "result-grade-A"
    if g.startswith("B"): return "result-grade-B"
    if g.startswith("C"): return "result-grade-C"
    return "result-grade-D"

def grade_tagline(g: str) -> str:
    taglines = {
        "A+": "Outstanding — a rare calibre of candidate.",
        "A":  "Excellent performance — strong hire signal.",
        "B+": "Very good — above expectations in most areas.",
        "B":  "Solid candidate with clear strengths.",
        "C":  "Adequate but notable gaps remain.",
        "D":  "Significant development needed.",
    }
    return taglines.get(g, "Interview complete.")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Cabinet Grotesk, sans-serif", color="#e8f0fe"),
    xaxis=dict(gridcolor="#162035", zerolinecolor="#162035"),
    yaxis=dict(gridcolor="#162035", zerolinecolor="#162035"),
    margin=dict(t=40, b=40, l=20, r=20),
)

# ============================================================
# SCREEN — SETUP
# ============================================================
def screen_setup():
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">⚡ AI-Powered · Adaptive · Real-Time</div>
        <div class="hero-title">KETU AI</div>
        <p class="hero-sub">Meet Nova — your elite AI interviewer. She adapts in real-time, reads between the lines, and gives you feedback that actually matters.</p>
    </div>
    """, unsafe_allow_html=True)

    llm = get_llm()
    if llm is None:
        st.error("⚠️ `GROQ_API_KEY` not found. Add it to `.streamlit/secrets.toml`.")
        return

    # Nova intro card
    st.markdown(f"""
    <div class="interviewer-wrap">
        <div class="avatar-ring">🤖</div>
        <div>
            <div class="interviewer-name">Nova · AI Interviewer</div>
            <div class="interviewer-status"><span class="status-dot"></span> Ready to interview</div>
        </div>
        <div class="interviewer-speech">
            "Hello! I'm Nova, your AI interviewer. I'll personalise every question to your background and the role you're applying for. Fill in the details below and we'll begin — I promise to make this worth your time."
        </div>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="panel panel-glow-cyan">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">📋 Job Context</div>', unsafe_allow_html=True)
        st.session_state.candidate_name = st.text_input("Your Name (optional)", placeholder="e.g. Arjun Mehta", value=st.session_state.candidate_name)
        st.session_state.role_title = st.text_input("Role / Job Title *", placeholder="e.g. Senior Backend Engineer", value=st.session_state.role_title)
        st.session_state.jd_text = st.text_area("Job Description *", height=280, placeholder="Paste the full job description here…", value=st.session_state.jd_text)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel panel-glow-violet">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">📄 Your Resume</div>', unsafe_allow_html=True)
        resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf","docx","doc","txt"], label_visibility="collapsed")
        if resume_file:
            with st.spinner("Reading resume…"):
                st.session_state.resume_text = load_document(resume_file)
            words = len(st.session_state.resume_text.split())
            st.success(f"✅ Resume loaded — {words:,} words")
            with st.expander("Preview extracted text"):
                st.text(st.session_state.resume_text[:600] + "…")

        st.markdown("---")
        st.markdown('<div class="sec-label">⚙️ Session Settings</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.num_questions = st.slider("Questions", 4, 15, st.session_state.num_questions)
        with col2:
            st.session_state.tts_enabled = st.toggle("🔊 Voice (TTS)", value=st.session_state.tts_enabled)

        st.markdown("""
        <div class="tip-box">
        💡 Nova will ask adaptive follow-up questions if your answer needs more depth.
        Voice recording uses Groq's Whisper — speak clearly for best results.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀  Begin Interview with Nova", use_container_width=True):
            if not st.session_state.jd_text.strip():
                st.error("Please paste a job description.")
            elif not st.session_state.resume_text.strip():
                st.error("Please upload a resume.")
            elif not st.session_state.role_title.strip():
                st.error("Please enter the role / job title.")
            else:
                with st.spinner("🤖 Nova is reviewing your profile and crafting questions…"):
                    qs, types = generate_questions(
                        st.session_state.jd_text,
                        st.session_state.resume_text,
                        st.session_state.role_title,
                        st.session_state.num_questions,
                        llm,
                    )
                if not qs:
                    st.error("Could not generate questions. Check your API key.")
                else:
                    greeting = random.choice(INTERVIEWER["greetings"])
                    st.session_state.questions     = qs
                    st.session_state.q_types       = types
                    st.session_state.current       = 0
                    st.session_state.scores        = []
                    st.session_state.feedback_list = []
                    st.session_state.time_per_q    = []
                    st.session_state.session_start = time.time()
                    st.session_state.q_start       = time.time()
                    st.session_state.submitted     = False
                    st.session_state.nova_message  = greeting
                    st.session_state.is_followup   = False
                    st.session_state.followup_count= 0
                    st.session_state.transcript    = []
                    st.session_state.screen        = "interview"
                    st.rerun()

# ============================================================
# SCREEN — INTERVIEW
# ============================================================
def screen_interview():
    llm = get_llm()
    idx = st.session_state.current
    questions = st.session_state.questions
    q_types   = st.session_state.q_types
    n = len(questions)

    if idx >= n:
        st.session_state.screen = "results"
        st.rerun()

    q      = questions[idx]
    q_type = q_types[idx] if idx < len(q_types) else "technical"
    q_info = QUESTION_TYPES.get(q_type, ("❓", "badge-technical", q_type.title()))

    # ── Top bar ──────────────────────────────────────────────
    tb1, tb2, tb3, tb4 = st.columns([4, 1, 1, 1])
    with tb1:
        st.progress(idx / n)
        elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
        mins, secs = divmod(elapsed, 60)
        st.caption(f"Question {idx+1} of {n}  ·  {mins:02d}:{secs:02d} elapsed")
    with tb2:
        avg = (sum(s.get("score",0) for s in st.session_state.scores) / len(st.session_state.scores)) if st.session_state.scores else 0.0
        st.metric("Avg Score", f"{avg:.1f}")
    with tb3:
        st.metric("Answered", f"{len(st.session_state.scores)}/{n}")
    with tb4:
        if st.button("⏹ End", help="End interview and see results"):
            st.session_state.screen = "results"
            st.rerun()

    st.markdown("---")

    # ── Nova avatar + speech ─────────────────────────────────
    nova_msg = st.session_state.get("nova_message", "")
    speaking_class = "speaking" if nova_msg else ""
    is_followup = st.session_state.get("is_followup", False)

    st.markdown(f"""
    <div class="interviewer-wrap">
        <div class="avatar-ring {speaking_class}">{INTERVIEWER['avatar']}</div>
        <div>
            <div class="interviewer-name">{INTERVIEWER['name']}</div>
            <div class="interviewer-status"><span class="status-dot"></span> {INTERVIEWER['title']}</div>
        </div>
        <div class="interviewer-speech">"{nova_msg or 'Ready for your answer…'}"</div>
    </div>
    """, unsafe_allow_html=True)

    # TTS for nova message
    if nova_msg and f"tts_done_{idx}_{nova_msg[:20]}" not in st.session_state:
        tts_autoplay(nova_msg)
        st.session_state[f"tts_done_{idx}_{nova_msg[:20]}"] = True

    # ── Question card ─────────────────────────────────────────
    if is_followup:
        st.markdown('<div class="followup-badge">🔄 Follow-up Question</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="q-card">
        <div class="q-num">Question {idx+1} of {n}</div>
        <p class="q-text">{q}</p>
        <div><span class="q-type-badge {q_info[1]}">{q_info[0]} {q_info[2]}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Answer area ───────────────────────────────────────────
    if not st.session_state.submitted:

        # Voice recording
        if HAS_AUDIO_RECORDER:
            st.markdown('<div class="sec-label">🎙️ Voice Answer</div>', unsafe_allow_html=True)
            st.markdown('<div class="rec-strip"><span class="rec-label"><span class="rec-dot"></span> Click microphone to record</span><span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:var(--text3)">Powered by Groq Whisper</span></div>', unsafe_allow_html=True)
            audio_bytes = audio_recorder(text="", icon_size="2x", key=f"rec_{idx}")
            if audio_bytes and f"transcribed_{idx}" not in st.session_state:
                st.markdown(WAVEFORM_HTML, unsafe_allow_html=True)
                with st.spinner("Transcribing your voice…"):
                    text = transcribe_voice(audio_bytes)
                    if text:
                        st.session_state[f"answer_{idx}"] = text
                        st.session_state[f"transcribed_{idx}"] = True
                        st.rerun()

        st.markdown('<div class="sec-label">✍️ Text Answer</div>', unsafe_allow_html=True)

        if f"transcribed_{idx}" in st.session_state:
            st.info(f"🎙️ Transcribed: *{st.session_state.get(f'answer_{idx}', '')}*")

        ans = st.text_area(
            "Your response",
            value=st.session_state.get(f"answer_{idx}", ""),
            key=f"input_{idx}",
            height=180,
            placeholder="Type your answer here, or use voice recording above…",
            label_visibility="collapsed",
        )

        tip_map = {
            "technical":   "💡 Tip: Be specific — mention tools, approaches, and outcomes.",
            "behavioral":  "💡 Tip: Use the STAR method — Situation, Task, Action, Result.",
            "rapport":     "💡 Tip: Be authentic and conversational.",
            "situational": "💡 Tip: Walk through your thought process step by step.",
            "ambition":    "💡 Tip: Connect your goals to the role and company.",
        }
        st.markdown(f'<div class="tip-box">{tip_map.get(q_type, "💡 Take your time and be specific.")}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            submit = st.button("✓  Submit Answer", use_container_width=True)
        with col2:
            skip = st.button("Skip →", use_container_width=True, help="Skip this question")

        if skip:
            st.session_state.transcript.append({"role": "user", "content": "[Skipped]", "q": q})
            transition = random.choice(INTERVIEWER["transitions"])
            st.session_state.current   += 1
            st.session_state.submitted  = False
            st.session_state.is_followup = False
            st.session_state.nova_message = transition
            st.session_state.q_start = time.time()
            st.rerun()

        if submit:
            if not ans.strip():
                st.warning("Please provide an answer before submitting.")
            else:
                with st.spinner("Nova is analysing your response…"):
                    eval_res = evaluate_answer(q, ans, st.session_state.role_title, q_type, llm)

                # Track transcript
                st.session_state.transcript.append({"role": "user", "content": ans, "q": q})
                st.session_state.transcript.append({"role": "nova", "content": eval_res.get("nova_reaction", "")})

                # Only count non-followup to main scores
                if not is_followup:
                    st.session_state.scores.append(eval_res)
                    st.session_state.feedback_list.append({
                        "q": q, "a": ans, "eval": eval_res, "type": q_type,
                        "time": int(time.time() - (st.session_state.q_start or time.time()))
                    })
                else:
                    # Update last score with follow-up bonus
                    if st.session_state.scores:
                        prev = st.session_state.scores[-1]
                        new_score = min(10.0, (prev["score"] + eval_res["score"]) / 2 + 0.5)
                        st.session_state.scores[-1]["score"] = new_score

                st.session_state.current_feedback = eval_res
                st.session_state.submitted = True

                # Decide follow-up
                needs_followup = (
                    eval_res.get("needs_followup", False)
                    and eval_res.get("followup_question", "")
                    and st.session_state.followup_count < 2
                    and not is_followup  # only 1 level of follow-up
                )
                st.session_state._pending_followup = needs_followup
                st.rerun()

    # ── Feedback view ─────────────────────────────────────────
    else:
        f = st.session_state.current_feedback
        sc = f.get("score", 5.0)
        sc_class = score_class(sc)
        tones = f.get("tone_signals", [])
        nova_react = f.get("nova_reaction", "")

        # Nova reaction
        if nova_react:
            st.markdown(f"""
            <div class="interviewer-wrap">
                <div class="avatar-ring">{INTERVIEWER['avatar']}</div>
                <div>
                    <div class="interviewer-name">{INTERVIEWER['name']}</div>
                    <div class="interviewer-status"><span class="status-dot"></span> Feedback</div>
                </div>
                <div class="interviewer-speech">"{nova_react}"</div>
            </div>
            """, unsafe_allow_html=True)
            if f"tts_react_{idx}" not in st.session_state:
                tts_autoplay(nova_react)
                st.session_state[f"tts_react_{idx}"] = True

        # Score + verdict
        tone_chips_html = "".join([f'<span class="tone-chip">{t}</span>' for t in tones])
        st.markdown(f"""
        <div class="feedback-card">
            <div class="score-display">
                <div class="score-circle {sc_class}">{sc:.1f}</div>
                <div>
                    <div class="verdict-text">{f.get('verdict','Average')}</div>
                    <div class="verdict-sub">out of 10</div>
                    <div class="tone-chips">{tone_chips_html}</div>
                </div>
            </div>
            <div class="feedback-section">
                <div class="feedback-label label-strength">✓ Strength</div>
                <div class="feedback-text">{f.get('strength','—')}</div>
            </div>
            <div class="feedback-section">
                <div class="feedback-label label-weakness">✗ Gap</div>
                <div class="feedback-text">{f.get('weakness','—')}</div>
            </div>
            <div class="feedback-section">
                <div class="feedback-label label-suggestion">→ Suggestion</div>
                <div class="feedback-text">{f.get('suggestion','—')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Follow-up or next
        pending_followup = st.session_state.get("_pending_followup", False)
        fq = f.get("followup_question", "")

        if pending_followup and fq:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄  Answer Follow-up", use_container_width=True):
                    # Inject follow-up as next question (temporary)
                    st.session_state.questions.insert(idx + 1, fq)
                    st.session_state.q_types.insert(idx + 1, q_type)
                    st.session_state.current     += 1
                    st.session_state.submitted    = False
                    st.session_state.is_followup  = True
                    st.session_state.followup_count += 1
                    st.session_state._pending_followup = False
                    st.session_state.nova_message = f"I'd like to dig deeper. {fq}"
                    st.session_state.q_start = time.time()
                    st.rerun()
            with col2:
                if st.button("Skip Follow-up →", use_container_width=True):
                    transition = random.choice(INTERVIEWER["transitions"])
                    st.session_state.current    += 1
                    st.session_state.submitted   = False
                    st.session_state.is_followup = False
                    st.session_state._pending_followup = False
                    st.session_state.nova_message = transition
                    st.session_state.q_start = time.time()
                    st.rerun()
        else:
            next_label = "Finish Interview →" if idx + 1 >= n else "Next Question →"
            if st.button(next_label, use_container_width=True):
                transition = random.choice(INTERVIEWER["transitions"])
                st.session_state.current     += 1
                st.session_state.submitted    = False
                st.session_state.is_followup  = False
                st.session_state._pending_followup = False
                st.session_state.nova_message = transition
                st.session_state.q_start = time.time()
                st.rerun()

# ============================================================
# SCREEN — RESULTS
# ============================================================
def screen_results():
    llm = get_llm()
    scores = st.session_state.scores
    feedback_list = st.session_state.feedback_list

    if not scores:
        st.warning("No answers were recorded.")
        if st.button("Start Over"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        return

    avg_score  = sum(s.get("score",0) for s in scores) / len(scores)
    grade      = grade_letter(avg_score)
    g_class    = grade_class(grade)
    tagline    = grade_tagline(grade)
    name       = st.session_state.candidate_name or "Candidate"
    role       = st.session_state.role_title
    elapsed    = int(time.time() - (st.session_state.session_start or time.time()))
    mins       = elapsed // 60
    n_total    = len(st.session_state.questions)

    # ── Hero banner ───────────────────────────────────────────
    st.markdown(f"""
    <div class="result-hero">
        <div class="hero-eyebrow">Interview Complete · {name} · {role}</div>
        <div class="result-grade {g_class}">{grade}</div>
        <div class="result-label">Final Grade · {avg_score:.1f} / 10</div>
        <div class="result-tagline">{tagline}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final Score", f"{avg_score:.1f}/10")
    m2.metric("Questions Answered", f"{len(scores)}/{n_total}")
    m3.metric("Interview Duration", f"{mins}m")
    top = max(scores, key=lambda s: s.get("score",0)) if scores else {}
    m4.metric("Best Answer", f"{top.get('score',0):.1f}/10")

    st.markdown("---")

    # ── Two-column layout ─────────────────────────────────────
    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        # Score timeline chart
        st.markdown('<div class="sec-label">📈 Score Timeline</div>', unsafe_allow_html=True)
        if len(scores) >= 2:
            fig_line = go.Figure()
            q_labels = [f"Q{i+1}" for i in range(len(scores))]
            vals     = [s.get("score",0) for s in scores]
            fig_line.add_trace(go.Scatter(
                x=q_labels, y=vals, mode="lines+markers",
                line=dict(color="#00e5ff", width=2.5),
                marker=dict(size=8, color=vals, colorscale=[[0,"#f43f5e"],[0.5,"#f59e0b"],[1,"#10b981"]], line=dict(color="#020408",width=2)),
                fill="tozeroy",
                fillcolor="rgba(0,229,255,0.06)",
                name="Score",
            ))
            fig_line.add_hline(y=avg_score, line_dash="dot", line_color="rgba(0,229,255,0.4)", annotation_text=f"avg {avg_score:.1f}")
            fig_line.update_layout(**PLOTLY_LAYOUT, height=250, showlegend=False, yaxis=dict(range=[0,10.5], gridcolor="#162035", zerolinecolor="#162035"), xaxis=dict(gridcolor="#162035"))
            st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

        # Q-by-Q breakdown
        st.markdown('<div class="sec-label">📋 Question Breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        for i, item in enumerate(feedback_list):
            sc = item["eval"].get("score", 0)
            sc_cls = score_class(sc)
            verdict = item["eval"].get("verdict", "—")
            q_t = item.get("type","technical")
            q_info = QUESTION_TYPES.get(q_t, ("❓","badge-technical",q_t.title()))
            time_taken = item.get("time", 0)
            score_color = "#10b981" if sc >= 7 else "#f59e0b" if sc >= 5 else "#f43f5e"
            st.markdown(f"""
            <div class="q-timeline-item">
                <div class="q-timeline-num">Q{i+1}</div>
                <div class="q-timeline-content">
                    <div class="q-timeline-q">{item['q'][:80]}{'…' if len(item['q'])>80 else ''}</div>
                    <span class="q-timeline-score" style="background:rgba(0,0,0,0.2);color:{score_color};border:1px solid {score_color}40">
                        {sc:.1f}/10 · {verdict}
                    </span>
                    <span class="q-type-badge {q_info[1]}" style="margin-left:0.4rem">{q_info[0]} {q_info[2]}</span>
                    {f'<span style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:var(--text3);margin-left:0.4rem">⏱ {time_taken}s</span>' if time_taken else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander(f"Full feedback — Q{i+1}"):
                st.write(f"**Answer:** {item['a']}")
                tones = item['eval'].get('tone_signals', [])
                if tones:
                    st.write(f"**Tone:** {' · '.join(tones)}")
                st.success(f"**Strength:** {item['eval'].get('strength','—')}")
                st.error(f"**Gap:** {item['eval'].get('weakness','—')}")
                st.info(f"**Suggestion:** {item['eval'].get('suggestion','—')}")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # Radar chart
        if len(feedback_list) >= 3:
            st.markdown('<div class="sec-label">🕸️ Skill Radar</div>', unsafe_allow_html=True)
            type_scores: dict[str, list] = {}
            for item in feedback_list:
                t = item.get("type", "technical")
                type_scores.setdefault(t, []).append(item["eval"].get("score", 0))
            categories = list(type_scores.keys())
            values     = [sum(v)/len(v) for v in type_scores.values()]
            if len(categories) >= 3:
                categories_closed = categories + [categories[0]]
                values_closed     = values + [values[0]]
                fig_radar = go.Figure(go.Scatterpolar(
                    r=values_closed, theta=categories_closed,
                    fill="toself", fillcolor="rgba(0,229,255,0.08)",
                    line=dict(color="#00e5ff", width=2),
                    marker=dict(color="#00e5ff", size=6),
                ))
                fig_radar.update_layout(**{**PLOTLY_LAYOUT, "polar": dict(
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(color="#3d5278", gridcolor="#162035"),
                    radialaxis=dict(range=[0,10], color="#3d5278", gridcolor="#162035"),
                )}, height=320)
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        # Score distribution bar
        st.markdown('<div class="sec-label">📊 Score Distribution</div>', unsafe_allow_html=True)
        if scores:
            vals = [s.get("score",0) for s in scores]
            bins = {"0-4": 0, "5-6": 0, "7-8": 0, "9-10": 0}
            for v in vals:
                if v <= 4: bins["0-4"] += 1
                elif v <= 6: bins["5-6"] += 1
                elif v <= 8: bins["7-8"] += 1
                else: bins["9-10"] += 1
            fig_bar = go.Figure(go.Bar(
                x=list(bins.keys()), y=list(bins.values()),
                marker_color=["#f43f5e","#f59e0b","#00e5ff","#10b981"],
                marker_line_width=0,
            ))
            fig_bar.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        # AI summary
        st.markdown('<div class="sec-label">🤖 Nova\'s Assessment</div>', unsafe_allow_html=True)
        if "ai_summary" not in st.session_state:
            with st.spinner("Nova is writing your assessment…"):
                st.session_state.ai_summary = generate_summary(feedback_list, role, name, avg_score, llm)
        st.markdown(f"""
        <div class="panel panel-glow-violet">
            <div style="font-family:'Cabinet Grotesk',sans-serif;font-size:0.92rem;color:var(--text2);line-height:1.7;">
                {st.session_state.ai_summary.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄  New Interview", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    with col2:
        if st.button("📋  Practice Again (Same Role)", use_container_width=True):
            resume = st.session_state.resume_text
            jd     = st.session_state.jd_text
            role_t = st.session_state.role_title
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.resume_text = resume
            st.session_state.jd_text     = jd
            st.session_state.role_title  = role_t
            st.rerun()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.8rem;color:#00e5ff;margin-bottom:0.3rem">KETU AI</div>', unsafe_allow_html=True)
    st.caption("Elite · Adaptive · Intelligent")
    st.markdown("---")

    screen = st.session_state.get("screen", "setup")
    if screen == "interview":
        idx = st.session_state.current
        n   = len(st.session_state.questions)
        st.progress(idx / n if n else 0)
        st.metric("Progress", f"{idx}/{n}")
        if st.session_state.scores:
            avg = sum(s.get("score",0) for s in st.session_state.scores) / len(st.session_state.scores)
            g   = grade_letter(avg)
            st.metric("Current Grade", g)
        st.markdown("---")
        if st.button("End Interview Early"):
            st.session_state.screen = "results"
            st.rerun()
    elif screen == "results":
        st.success("Interview complete!")
        if st.session_state.scores:
            avg = sum(s.get("score",0) for s in st.session_state.scores) / len(st.session_state.scores)
            st.metric("Final Score", f"{avg:.1f}/10")
            st.metric("Grade", grade_letter(avg))

    st.markdown("---")
    st.markdown("##### ⚡ Nova's Capabilities")
    st.markdown("""
- 🧠 Adaptive follow-up questions
- 🎙️ Voice input via Groq Whisper
- 📊 Tone & confidence analysis
- 🕸️ Multi-dimension skill radar
- 📝 AI-written post-interview report
- 🔊 Text-to-speech question delivery
- 📄 PDF / DOCX resume parsing
- ⚡ Llama 3.3-70B via Groq
""")
    st.markdown("---")
    st.caption(f"Session · {datetime.now().strftime('%H:%M · %d %b %Y')}")

# ============================================================
# ROUTER
# ============================================================
if st.session_state.screen == "setup":
    screen_setup()
elif st.session_state.screen == "interview":
    screen_interview()
elif st.session_state.screen == "results":
    screen_results()
