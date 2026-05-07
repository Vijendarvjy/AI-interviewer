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
import re, io, time, base64, tempfile, json, random, hashlib
from io import BytesIO
from datetime import datetime
from collections import Counter

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
# ADVANCED DESIGN SYSTEM
# ============================================================
DESIGN = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=Cabinet+Grotesk:wght@400;500;700;800;900&family=Fraunces:ital,opsz,wght@0,9..144,300;1,9..144,400&display=swap');

:root {
    --bg:           #010306;
    --surface:      #050b12;
    --surface2:     #08111d;
    --surface3:     #0c1828;
    --border:       #111e30;
    --border2:      #182840;
    --border3:      #1f3555;
    --cyan:         #00e5ff;
    --cyan-dim:     rgba(0,229,255,0.6);
    --violet:       #8b5cf6;
    --violet-dim:   rgba(139,92,246,0.6);
    --rose:         #f43f5e;
    --emerald:      #10b981;
    --amber:        #f59e0b;
    --gold:         #fbbf24;
    --ice:          #e0f2fe;
    --text:         #e8f0fe;
    --text2:        #7a93bb;
    --text3:        #2d4464;
    --radius:       16px;
    --radius-lg:    24px;
    --radius-xl:    32px;
    --glow-cyan:    0 0 80px rgba(0,229,255,0.08), 0 0 160px rgba(0,229,255,0.04);
    --glow-violet:  0 0 80px rgba(139,92,246,0.08), 0 0 160px rgba(139,92,246,0.04);
    --glow-rose:    0 0 80px rgba(244,63,94,0.08);
    --glow-emerald: 0 0 60px rgba(16,185,129,0.12);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 100% 60% at 15% -5%,  rgba(0,229,255,0.04)  0%, transparent 55%),
        radial-gradient(ellipse 70%  55% at 85% 105%, rgba(139,92,246,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 50%  50% at 50% 50%,  rgba(8,17,29,0.9)    0%, transparent 100%),
        var(--bg) !important;
}

[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Dot-grid background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0;
    background-image: radial-gradient(circle, rgba(0,229,255,0.08) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0; opacity: 0.4;
}

/* Scanline overlay */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.04) 2px, rgba(0,0,0,0.04) 4px);
    pointer-events: none; z-index: 0;
}

/* ── Typography ── */
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }
code,pre  { font-family: 'DM Mono', monospace !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border3); border-radius: 99px; }

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.25s cubic-bezier(0.16,1,0.3,1) !important;
    letter-spacing: 0.02em !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button::before {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(0,229,255,0.04), transparent);
    opacity: 0; transition: opacity 0.25s ease !important;
}
.stButton > button:hover {
    border-color: var(--cyan) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 25px rgba(0,229,255,0.12), inset 0 0 20px rgba(0,229,255,0.03) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:hover::before { opacity: 1 !important; }
.stButton > button:active { transform: translateY(0px) !important; }

/* Primary button */
button[kind="primary"], .stButton > button[data-testid*="primary"] {
    background: linear-gradient(135deg, rgba(0,229,255,0.12), rgba(139,92,246,0.08)) !important;
    border-color: rgba(0,229,255,0.4) !important;
    color: var(--cyan) !important;
}

/* ── Text areas ── */
.stTextArea textarea {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    color: var(--text) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.875rem !important;
    line-height: 1.65 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: rgba(0,229,255,0.4) !important;
    box-shadow: 0 0 30px rgba(0,229,255,0.06), inset 0 1px 0 rgba(0,229,255,0.05) !important;
    outline: none !important;
}

/* ── Inputs ── */
.stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Cabinet Grotesk', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus {
    border-color: rgba(0,229,255,0.4) !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.06) !important;
}

/* ── Select ── */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: var(--radius) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--violet) !important; }

/* ── Progress ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--violet), var(--cyan)) !important;
    border-radius: 99px !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.25) !important;
}
.stProgress > div > div { background: var(--border) !important; border-radius: 99px !important; height: 3px !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem 1.4rem !important;
    position: relative !important; overflow: hidden !important;
}
[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.3), transparent);
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important; font-size: 1.9rem !important;
    font-weight: 800 !important; color: var(--cyan) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important;
    color: var(--text3) !important; letter-spacing: 0.12em !important; text-transform: uppercase !important;
}

/* ── Alerts ── */
.stSuccess { background: rgba(16,185,129,0.06) !important; border-color: rgba(16,185,129,0.3) !important; border-radius: 12px !important; }
.stError   { background: rgba(244,63,94,0.06) !important;  border-color: rgba(244,63,94,0.3) !important;    border-radius: 12px !important; }
.stWarning { background: rgba(245,158,11,0.06) !important; border-color: rgba(245,158,11,0.3) !important;   border-radius: 12px !important; }
.stInfo    { background: rgba(0,229,255,0.04) !important;  border-color: rgba(0,229,255,0.2) !important;    border-radius: 12px !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Toggle ── */
.stToggle > label { color: var(--text2) !important; font-family: 'DM Mono', monospace !important; font-size: 0.8rem !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--text2) !important;
}

/* ════════════════════════════════════════════
   CUSTOM COMPONENTS
════════════════════════════════════════════ */

/* ── Hero ── */
.hero-wrap { text-align: center; padding: 5rem 0 2.5rem; position: relative; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.28em;
    text-transform: uppercase; color: var(--cyan); margin-bottom: 1.5rem; opacity: 0.7;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: clamp(4.5rem, 12vw, 9rem);
    font-weight: 800; line-height: 0.88; letter-spacing: -0.03em;
    background: linear-gradient(130deg, #ffffff 0%, var(--cyan) 40%, var(--violet) 80%, #f43f5e 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    filter: drop-shadow(0 0 60px rgba(0,229,255,0.15));
    animation: heroReveal 1s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.hero-title-accent {
    font-family: 'Fraunces', serif; font-style: italic; font-weight: 300;
    font-size: clamp(1.2rem, 3vw, 2rem);
    background: linear-gradient(90deg, var(--text2), var(--cyan-dim));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    display: block; margin-top: 0.8rem;
    animation: heroReveal 1s 0.1s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.hero-sub {
    font-family: 'Cabinet Grotesk', sans-serif; font-size: 1.1rem; color: var(--text2);
    margin-top: 1.5rem; max-width: 520px; margin-left: auto; margin-right: auto; line-height: 1.65;
    animation: heroReveal 1s 0.2s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.hero-stats {
    display: flex; justify-content: center; gap: 3rem; margin-top: 2.5rem;
    animation: heroReveal 1s 0.3s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.hero-stat-num {
    font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: var(--text);
}
.hero-stat-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text3); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.2rem; }

/* ── Panels ── */
.panel {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 2rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
    animation: panelIn 0.5s ease both;
}
.panel-glow-cyan   { border-color: rgba(0,229,255,0.15);  box-shadow: var(--glow-cyan); }
.panel-glow-violet { border-color: rgba(139,92,246,0.15); box-shadow: var(--glow-violet); }
.panel-glow-rose   { border-color: rgba(244,63,94,0.15);  box-shadow: var(--glow-rose); }
.panel-glow-gold   { border-color: rgba(251,191,36,0.2);  box-shadow: 0 0 60px rgba(251,191,36,0.06); }
.panel::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(0,229,255,0.4) 50%, transparent 100%);
    opacity: 0.4;
}

/* ── Section label ── */
.sec-label {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: var(--text3); margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Interviewer avatar ── */
.interviewer-wrap {
    display: flex; align-items: flex-start; gap: 1.5rem; padding: 1.5rem 1.8rem;
    background: var(--surface2); border: 1px solid var(--border2); border-radius: var(--radius-lg);
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
    animation: slideDown 0.4s ease both;
}
.interviewer-wrap::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--cyan), var(--violet), rgba(244,63,94,0.5));
}
.avatar-ring {
    width: 60px; height: 60px; border-radius: 50%;
    background: linear-gradient(135deg, var(--surface3), var(--surface2));
    border: 2px solid rgba(0,229,255,0.4); display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem; box-shadow: 0 0 20px rgba(0,229,255,0.15), inset 0 0 20px rgba(0,229,255,0.05);
    flex-shrink: 0; position: relative;
}
.avatar-ring.speaking::after {
    content: ''; position: absolute; inset: -7px; border-radius: 50%;
    border: 2px solid var(--cyan); animation: speakPulse 1.5s ease infinite; opacity: 0.4;
}
.avatar-ring.thinking { border-color: rgba(245,158,11,0.5) !important; animation: thinkPulse 2s ease infinite; }
.interviewer-meta { flex-shrink: 0; }
.interviewer-name { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; color: var(--text); }
.interviewer-status {
    font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--cyan);
    letter-spacing: 0.08em; margin-top: 0.25rem; display: flex; align-items: center; gap: 0.4rem;
}
.status-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--emerald); animation: blink 2s ease infinite; }
.status-dot.thinking { background: var(--amber); animation: blink 0.8s ease infinite; }
.interviewer-speech {
    font-family: 'Fraunces', serif; font-style: italic; font-size: 1rem;
    color: var(--text2); line-height: 1.6; flex: 1; letter-spacing: 0.01em;
}
.speech-quote { color: rgba(0,229,255,0.3); font-size: 1.4rem; line-height: 0; vertical-align: -0.3em; }

/* ── Question card ── */
.q-card {
    background: linear-gradient(135deg, rgba(0,229,255,0.03) 0%, rgba(139,92,246,0.03) 100%);
    border: 1px solid rgba(0,229,255,0.14); border-radius: var(--radius-lg);
    padding: 2.5rem 2.8rem; margin: 1.5rem 0; position: relative;
    animation: panelIn 0.4s ease both;
}
.q-card::after {
    content: ''; position: absolute; top: -1px; left: 8%; right: 8%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
    border-radius: 99px; opacity: 0.7;
}
.q-num { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--cyan); letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 1rem; opacity: 0.7; }
.q-text { font-family: 'Syne', sans-serif; font-size: clamp(1.1rem, 2.2vw, 1.45rem); font-weight: 600; line-height: 1.42; color: var(--text); margin: 0; }
.q-meta { display: flex; align-items: center; gap: 0.6rem; margin-top: 1.2rem; flex-wrap: wrap; }
.q-competency {
    font-family: 'DM Mono', monospace; font-size: 0.62rem; letter-spacing: 0.08em;
    color: var(--text3); background: var(--surface3); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.2rem 0.6rem;
}
.q-type-badge {
    display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.25rem 0.7rem;
    border-radius: 99px; font-family: 'DM Mono', monospace; font-size: 0.62rem; letter-spacing: 0.06em;
}
.badge-technical   { background: rgba(139,92,246,0.1); color: var(--violet); border: 1px solid rgba(139,92,246,0.25); }
.badge-behavioral  { background: rgba(0,229,255,0.07); color: var(--cyan);   border: 1px solid rgba(0,229,255,0.2); }
.badge-situational { background: rgba(245,158,11,0.08); color: var(--amber); border: 1px solid rgba(245,158,11,0.25); }
.badge-rapport     { background: rgba(16,185,129,0.08); color: var(--emerald); border: 1px solid rgba(16,185,129,0.25); }
.badge-ambition    { background: rgba(251,191,36,0.08); color: var(--gold); border: 1px solid rgba(251,191,36,0.25); }

/* ── Waveform ── */
.waveform-wrap { display: flex; align-items: center; justify-content: center; gap: 3px; height: 44px; margin: 0.8rem 0; }
.wave-bar { width: 3px; border-radius: 99px; background: var(--cyan); animation: waveDance var(--speed) ease-in-out infinite alternate; }

/* ── Live coaching bar ── */
.coaching-bar {
    background: rgba(0,229,255,0.03); border: 1px solid rgba(0,229,255,0.1);
    border-radius: 10px; padding: 0.75rem 1.1rem; margin-top: 0.6rem;
    font-family: 'DM Mono', monospace; font-size: 0.72rem; color: rgba(0,229,255,0.55);
    display: flex; align-items: center; gap: 0.6rem; line-height: 1.5;
    transition: all 0.3s ease;
}
.coaching-bar.warn { color: rgba(245,158,11,0.7); border-color: rgba(245,158,11,0.15); background: rgba(245,158,11,0.03); }
.coaching-bar.good { color: rgba(16,185,129,0.7); border-color: rgba(16,185,129,0.15); background: rgba(16,185,129,0.03); }
.coaching-icon { font-size: 0.9rem; }

/* ── Word count indicator ── */
.word-meter { display: flex; align-items: center; gap: 0.8rem; margin-top: 0.5rem; }
.word-meter-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text3); letter-spacing: 0.08em; }
.word-meter-bar { flex: 1; height: 2px; background: var(--border); border-radius: 99px; overflow: hidden; }
.word-meter-fill { height: 100%; border-radius: 99px; transition: width 0.3s ease, background 0.3s ease; }

/* ── Feedback card ── */
.feedback-card {
    background: var(--surface2); border: 1px solid var(--border2); border-radius: var(--radius-lg);
    padding: 2rem; margin-top: 1.5rem; position: relative; overflow: hidden;
    animation: slideRight 0.4s ease both;
}
.feedback-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--violet), var(--cyan));
}
.feedback-section { margin-bottom: 1.1rem; padding-bottom: 1.1rem; border-bottom: 1px solid var(--border); }
.feedback-section:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.feedback-label { font-family: 'DM Mono', monospace; font-size: 0.63rem; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.45rem; }
.label-strength   { color: var(--emerald); }
.label-weakness   { color: var(--rose); }
.label-suggestion { color: var(--amber); }
.label-star       { color: var(--violet); }
.feedback-text    { font-family: 'Cabinet Grotesk', sans-serif; font-size: 0.93rem; color: var(--text2); line-height: 1.6; }

/* ── Score display ── */
.score-display { display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.score-ring-wrap { position: relative; width: 88px; height: 88px; flex-shrink: 0; }
.score-ring-num {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    font-family: 'Syne', sans-serif; font-size: 1.7rem; font-weight: 800;
}
.score-high { color: var(--emerald); }
.score-mid  { color: var(--amber); }
.score-low  { color: var(--rose); }
.verdict-text { font-family: 'Syne', sans-serif; font-size: 1.25rem; font-weight: 700; color: var(--text); }
.verdict-sub  { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: var(--text3); margin-top: 0.2rem; }
.tone-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.tone-chip {
    padding: 0.2rem 0.65rem; border-radius: 99px;
    font-family: 'DM Mono', monospace; font-size: 0.65rem;
    background: rgba(139,92,246,0.08); color: var(--violet-dim); border: 1px solid rgba(139,92,246,0.2);
}
.tone-chip.positive { background: rgba(16,185,129,0.08); color: rgba(16,185,129,0.7); border-color: rgba(16,185,129,0.2); }
.tone-chip.negative { background: rgba(244,63,94,0.07); color: rgba(244,63,94,0.65); border-color: rgba(244,63,94,0.2); }

/* ── STAR analysis ── */
.star-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.5rem; margin-top: 0.8rem; }
.star-cell {
    background: var(--surface3); border: 1px solid var(--border); border-radius: 8px;
    padding: 0.6rem 0.7rem; text-align: center;
}
.star-cell-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--text3); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.25rem; }
.star-cell-value { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; }
.star-found { color: var(--emerald); }
.star-missing { color: var(--rose); opacity: 0.5; }

/* ── Recording strip ── */
.rec-strip {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.75rem 1.1rem; background: rgba(244,63,94,0.05);
    border: 1px solid rgba(244,63,94,0.18); border-radius: 10px; margin-bottom: 0.8rem;
}
.rec-label { display: flex; align-items: center; gap: 0.5rem; font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--rose); }
.rec-dot   { width: 7px; height: 7px; border-radius: 50%; background: var(--rose); animation: blink 1s ease infinite; }

/* ── Mode selector cards ── */
.mode-cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem; margin-top: 0.5rem; }
.mode-card {
    background: var(--surface2); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.1rem; cursor: pointer; transition: all 0.2s ease; position: relative;
}
.mode-card:hover { border-color: var(--border3); transform: translateY(-2px); }
.mode-card.active-casual  { border-color: rgba(16,185,129,0.5);  background: rgba(16,185,129,0.05); }
.mode-card.active-standard{ border-color: rgba(0,229,255,0.4);   background: rgba(0,229,255,0.04); }
.mode-card.active-intense { border-color: rgba(244,63,94,0.4);   background: rgba(244,63,94,0.05); }
.mode-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.9rem; color: var(--text); margin-bottom: 0.3rem; }
.mode-desc  { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text3); line-height: 1.5; }

/* ── Progress ring SVG ── */
.progress-ring-svg { transform: rotate(-90deg); }

/* ── Results ── */
.result-hero {
    text-align: center; padding: 4rem 2rem;
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border); border-radius: var(--radius-xl);
    position: relative; overflow: hidden; margin-bottom: 2rem;
}
.result-hero::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 60% at 50% 0%, rgba(0,229,255,0.05), transparent);
}
.result-grade { font-family: 'Syne', sans-serif; font-size: clamp(5rem, 14vw, 10rem); font-weight: 800; line-height: 0.9; letter-spacing: -0.03em; }
.result-grade-A { background: linear-gradient(135deg, #10b981, #00e5ff, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; filter: drop-shadow(0 0 40px rgba(16,185,129,0.3)); }
.result-grade-B { background: linear-gradient(135deg, #00e5ff, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; filter: drop-shadow(0 0 40px rgba(0,229,255,0.25)); }
.result-grade-C { background: linear-gradient(135deg, #f59e0b, #f97316); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-grade-D { background: linear-gradient(135deg, #f43f5e, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-label   { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: var(--text3); letter-spacing: 0.18em; text-transform: uppercase; margin-top: 1.2rem; }
.result-tagline { font-family: 'Fraunces', serif; font-style: italic; font-size: 1.3rem; color: var(--text2); margin-top: 0.5rem; }

/* ── Q timeline ── */
.q-timeline-item {
    display: flex; gap: 1.1rem; align-items: flex-start;
    padding: 1.1rem 0; border-bottom: 1px solid var(--border);
}
.q-timeline-item:last-child { border-bottom: none; }
.q-timeline-num { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text3); min-width: 26px; padding-top: 0.15rem; }
.q-timeline-content { flex: 1; }
.q-timeline-q { font-family: 'Cabinet Grotesk', sans-serif; font-weight: 600; font-size: 0.92rem; color: var(--text); margin-bottom: 0.4rem; }

/* ── Tip box ── */
.tip-box {
    background: rgba(0,229,255,0.02); border: 1px solid rgba(0,229,255,0.1);
    border-radius: 10px; padding: 0.8rem 1.1rem;
    font-family: 'DM Mono', monospace; font-size: 0.75rem;
    color: rgba(0,229,255,0.5); margin-top: 0.8rem; line-height: 1.6;
}

/* ── Follow-up badge ── */
.followup-badge {
    display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.28rem 0.75rem;
    background: rgba(244,63,94,0.07); border: 1px solid rgba(244,63,94,0.18); border-radius: 99px;
    font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--rose); letter-spacing: 0.06em; margin-bottom: 0.8rem;
}

/* ── Competency pill ── */
.competency-pill {
    display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.4rem 0.9rem;
    border-radius: 8px; font-family: 'DM Mono', monospace; font-size: 0.68rem;
    background: var(--surface3); border: 1px solid var(--border2); color: var(--text2); margin: 0.2rem;
}
.competency-score { font-weight: 700; margin-left: 0.3rem; }

/* ── Export block ── */
.export-strip {
    display: flex; align-items: center; gap: 1rem; padding: 1rem 1.4rem;
    background: rgba(251,191,36,0.04); border: 1px solid rgba(251,191,36,0.15);
    border-radius: 12px; margin-top: 1rem;
}
.export-icon { font-size: 1.4rem; }
.export-info { flex: 1; }
.export-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.9rem; color: var(--text); }
.export-desc  { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: var(--text3); margin-top: 0.15rem; }

/* ── Difficulty badges ── */
.diff-easy   { color: var(--emerald); }
.diff-medium { color: var(--amber); }
.diff-hard   { color: var(--rose); }

/* ── Animations ── */
@keyframes heroReveal {
    from { opacity: 0; transform: translateY(24px); filter: blur(6px); }
    to   { opacity: 1; transform: translateY(0); filter: blur(0); }
}
@keyframes panelIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideRight {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
@keyframes speakPulse {
    0%   { transform: scale(1);   opacity: 0.4; }
    100% { transform: scale(1.35); opacity: 0; }
}
@keyframes thinkPulse {
    0%, 100% { box-shadow: 0 0 12px rgba(245,158,11,0.2); }
    50%       { box-shadow: 0 0 30px rgba(245,158,11,0.4); }
}
@keyframes waveDance {
    from { height: 4px; }
    to   { height: var(--max-h); }
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-6px); }
}
@keyframes shimmer {
    from { background-position: -200% center; }
    to   { background-position: 200% center; }
}
"""

st.markdown(f"<style>{DESIGN}</style>", unsafe_allow_html=True)

# Waveform HTML builder
def build_waveform():
    bars = "".join([
        f'<div class="wave-bar" style="--speed:{random.uniform(0.35,0.85):.2f}s;--max-h:{random.randint(14,42)}px;height:{random.randint(4,16)}px;opacity:{random.uniform(0.45,0.9):.2f};background:{"var(--cyan)" if i % 3 != 2 else "var(--violet)"};"></div>'
        for i in range(32)
    ])
    return f'<div class="waveform-wrap">{bars}</div>'

# Progress ring SVG builder
def score_ring_svg(score: float, size=88, stroke=5):
    r = (size / 2) - stroke
    circ = 2 * 3.14159 * r
    pct = min(score / 10.0, 1.0)
    dash = pct * circ
    color = "#10b981" if score >= 7 else "#f59e0b" if score >= 5 else "#f43f5e"
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" class="progress-ring-svg">
        <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#162035" stroke-width="{stroke}"/>
        <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
            stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"
            style="filter: drop-shadow(0 0 6px {color}66)"/>
    </svg>
    """

# ============================================================
# INTERVIEWER PERSONAS
# ============================================================
PERSONAS = {
    "Ketu": {
        "name": "Ketu",
        "title": "Senior AI Interviewer",
        "avatar": "🤖",
        "style": "balanced, thoughtful, and encouraging",
        "greetings": [
            "Great to meet you! I've reviewed your profile carefully — I'm genuinely excited to learn more about your journey.",
            "Welcome! I've gone through your background and the role requirements. Let's have an honest conversation.",
            "Hello! I've prepared some targeted questions for you. Take your time — there are no trick questions here.",
        ],
        "transitions": [
            "Interesting — thanks for sharing that. Let me move to the next area.",
            "Got it, I appreciate your openness. Moving on…",
            "That's helpful context. Let's continue.",
            "Thank you. Here's my next question for you.",
        ],
    },
    "Aria": {
        "name": "Aria",
        "title": "Technical Director · KETU",
        "avatar": "🧬",
        "style": "technical, precise, and deeply analytical",
        "greetings": [
            "I'll be direct: I want to understand how you think, not just what you know. Let's begin.",
            "I've reviewed your technical background. I'll be asking you to go deep — that's how we find out what you're really capable of.",
        ],
        "transitions": [
            "Noted. Let's test another dimension.",
            "That answer raises an interesting follow-up. Let me probe further.",
            "Understood. Moving on.",
        ],
    },
    "Marcus": {
        "name": "Marcus",
        "title": "Culture Lead · KETU",
        "avatar": "🌿",
        "style": "empathetic, culture-focused, and people-oriented",
        "greetings": [
            "I want this to feel like a real conversation — not an interrogation. I'm genuinely curious about who you are.",
            "Welcome! I focus on values, teamwork, and the human side of work. Let's explore that together.",
        ],
        "transitions": [
            "I appreciate your honesty. Let's explore another dimension.",
            "That's really telling — thank you. Let me ask about something different now.",
            "Good. Let's keep going.",
        ],
    }
}

QUESTION_TYPES = {
    "rapport":     ("💬", "badge-rapport",     "Rapport"),
    "technical":   ("⚙️", "badge-technical",   "Technical"),
    "behavioral":  ("🧠", "badge-behavioral",  "Behavioral"),
    "situational": ("🎯", "badge-situational", "Situational"),
    "ambition":    ("🚀", "badge-ambition",    "Forward-looking"),
}

INTERVIEW_MODES = {
    "Casual":   {"pressure": "low",  "followup_threshold": 4.0, "max_followups": 1, "emoji": "🌿", "desc": "Relaxed pace, supportive tone"},
    "Standard": {"pressure": "med",  "followup_threshold": 5.5, "max_followups": 2, "emoji": "⚡", "desc": "Professional, balanced assessment"},
    "Intense":  {"pressure": "high", "followup_threshold": 7.0, "max_followups": 3, "emoji": "🔥", "desc": "High-pressure, rigorous evaluation"},
}

COMPETENCY_FRAMEWORKS = {
    "Engineering":    ["Problem Solving", "Technical Depth", "System Design", "Collaboration", "Communication", "Growth Mindset"],
    "Management":     ["Leadership", "Strategic Thinking", "Communication", "Conflict Resolution", "Decision Making", "Mentoring"],
    "Product":        ["Product Sense", "Data Analysis", "User Empathy", "Prioritization", "Communication", "Execution"],
    "Design":         ["Visual Thinking", "User Research", "Communication", "Iteration", "Craft", "Business Acumen"],
    "Sales/BD":       ["Persuasion", "Relationship Building", "Resilience", "Product Knowledge", "Communication", "Closing"],
    "Data/Analytics": ["Statistical Thinking", "SQL/Tooling", "Communication", "Problem Framing", "Visualization", "Business Acumen"],
}

POSITIVE_TONE_SIGNALS = {"Confident", "Structured", "Concise", "Detailed", "Passionate", "Analytical", "Creative", "Experienced", "Thoughtful"}
NEGATIVE_TONE_SIGNALS = {"Vague", "Nervous", "Hesitant", "Rambling", "Unprepared"}

# ============================================================
# SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "screen":             "setup",
        "questions":          [],
        "q_types":            [],
        "q_competencies":     [],
        "q_difficulties":     [],
        "current":            0,
        "scores":             [],
        "feedback_list":      [],
        "resume_text":        "",
        "jd_text":            "",
        "candidate_name":     "",
        "role_title":         "",
        "num_questions":      8,
        "session_start":      None,
        "q_start":            None,
        "time_per_q":         [],
        "tts_enabled":        True,
        "submitted":          False,
        "current_feedback":   None,
        "Ketu_message":       "",
        "is_followup":        False,
        "followup_count":     0,
        "transcript":         [],
        "persona":            "Ketu",
        "interview_mode":     "Standard",
        "competency_scores":  {},
        "filler_word_counts": [],
        "word_counts":        [],
        "ai_summary":         None,
        "star_analyses":      [],
        "category_tag":       "Engineering",
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
# ADVANCED QUESTION GENERATION
# ============================================================
def generate_questions(jd: str, resume: str, role: str, n: int, llm, persona_name: str, mode: str, category: str) -> tuple:
    persona = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    competencies = COMPETENCY_FRAMEWORKS.get(category, COMPETENCY_FRAMEWORKS["Engineering"])
    comp_str = ", ".join(competencies)
    mode_cfg = INTERVIEW_MODES.get(mode, INTERVIEW_MODES["Standard"])

    prompt = f"""You are {persona['name']}, a {persona['style']} AI interviewer at KETU.
You are conducting a {mode} ({mode_cfg['pressure']}-pressure) interview for: {role}

KEY COMPETENCIES TO ASSESS: {comp_str}

JOB DESCRIPTION:
{jd[:2800]}

CANDIDATE RESUME:
{resume[:2800]}

Generate exactly {n} insightful, tailored interview questions.

Structure:
- Q1-2: rapport (warm, personal, background-focused)
- Q3-{max(4, n-3)}: technical/behavioral (deep skill assessment, map to competencies)
- Q{max(4, n-3)+1}-{n-1}: situational/behavioral (STAR-method scenarios, real challenges)
- Q{n}: ambition (forward-looking, growth-oriented)

Rules for {mode} mode:
{"- Keep tone light and supportive. Focus on broad understanding." if mode == "Casual" else ""}
{"- Balanced depth. Probe for specifics without being aggressive." if mode == "Standard" else ""}
{"- Ask hard, multi-part questions. Push for edge cases, trade-offs, failures." if mode == "Intense" else ""}
- Reference SPECIFIC skills and experiences from the resume
- Vary sentence structure — some short, some multi-part
- Assign difficulty: easy/medium/hard
- Map each question to one competency from: {comp_str}

Return ONLY this JSON (no markdown, no extra text):
{{
  "questions": ["q1", "q2", ...],
  "types": ["rapport", "technical", ...],
  "competencies": ["Communication", "Technical Depth", ...],
  "difficulties": ["easy", "medium", "hard", ...]
}}

Types: rapport, technical, behavioral, situational, ambition
"""
    response = llm.invoke(prompt)
    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        data = json.loads(raw)
        questions    = data.get("questions", [])[:n]
        types        = data.get("types", ["technical"] * n)[:n]
        competencies_list = data.get("competencies", ["Technical Depth"] * n)[:n]
        difficulties = data.get("difficulties", ["medium"] * n)[:n]
        while len(types) < len(questions):        types.append("technical")
        while len(competencies_list) < len(questions): competencies_list.append("Technical Depth")
        while len(difficulties) < len(questions): difficulties.append("medium")
        return questions, types, competencies_list, difficulties
    except Exception:
        questions, types, comps, diffs = [], [], [], []
        for line in response.content.splitlines():
            line = line.strip()
            if line and re.match(r'^\d+[.)\-]', line):
                cleaned = re.sub(r'^\d+[.)\-]\s*', '', line).strip()
                if cleaned:
                    questions.append(cleaned)
                    types.append("technical")
                    comps.append("Technical Depth")
                    diffs.append("medium")
        return questions[:n], types[:n], comps[:n], diffs[:n]

# ============================================================
# FILLER WORD ANALYSIS
# ============================================================
FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "literally", "actually", "sort of", "kind of", "i mean", "right", "so yeah", "honestly"}

def analyze_answer_quality(answer: str) -> dict:
    words = answer.lower().split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = word_count / sentence_count

    # Filler word count
    text_lower = answer.lower()
    filler_count = sum(text_lower.count(fw) for fw in FILLER_WORDS)
    filler_density = filler_count / max(word_count, 1)

    # STAR detection
    star_signals = {
        "Situation": bool(re.search(r'\b(when|once|at my|in my|during|while working|we were|i was)\b', text_lower)),
        "Task":      bool(re.search(r'\b(had to|needed to|responsible for|my role was|tasked with|goal was|objective)\b', text_lower)),
        "Action":    bool(re.search(r'\b(i did|i built|i wrote|i led|i implemented|i created|i designed|i worked|i developed|i resolved)\b', text_lower)),
        "Result":    bool(re.search(r'\b(result|outcome|achieved|improved|reduced|increased|saved|delivered|shipped|launched|percent|%|metric)\b', text_lower)),
    }
    star_score = sum(star_signals.values())

    # Verbosity (ideal: 80-250 words)
    if word_count < 30:      verbosity = "too_short"
    elif word_count < 80:    verbosity = "short"
    elif word_count <= 250:  verbosity = "ideal"
    elif word_count <= 400:  verbosity = "long"
    else:                    verbosity = "too_long"

    # Coaching hint
    if verbosity == "too_short":
        hint = ("warn", "💡", "Your answer is very brief — try to expand with context and a specific example.")
    elif verbosity == "short":
        hint = ("warn", "✍️", "Consider adding more detail — what was the outcome or result?")
    elif filler_density > 0.06:
        hint = ("warn", "🎙️", f"High use of filler words detected ({filler_count}x). Try to speak more deliberately.")
    elif star_score == 4 and verbosity == "ideal":
        hint = ("good", "✨", "Great structure — Situation, Task, Action and Result are all covered!")
    elif verbosity == "too_long":
        hint = ("warn", "✂️", "Consider tightening your answer — aim for the most impactful details.")
    elif star_score >= 3:
        hint = ("good", "🟢", "Good structure! You're covering the key components well.")
    else:
        hint = ("info", "💡", "Try to ground your answer in a specific real example with a measurable outcome.")

    return {
        "word_count":     word_count,
        "filler_count":   filler_count,
        "filler_density": filler_density,
        "star_signals":   star_signals,
        "star_score":     star_score,
        "verbosity":      verbosity,
        "coaching_hint":  hint,
        "avg_sentence_len": avg_sentence_len,
    }

# ============================================================
# ADVANCED ANSWER EVALUATION
# ============================================================
def evaluate_answer(question: str, answer: str, role: str, q_type: str, competency: str, mode: str, persona_name: str, llm, conversation_context: list = None) -> dict:
    persona = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    mode_cfg = INTERVIEW_MODES.get(mode, INTERVIEW_MODES["Standard"])
    context_str = ""
    if conversation_context:
        recent = conversation_context[-4:]
        context_str = "\n".join([f"{m['role'].upper()}: {m['content'][:200]}" for m in recent])

    prompt = f"""You are {persona['name']}, a {persona['style']} AI interviewer evaluating a {role} candidate.
Interview mode: {mode} ({mode_cfg['pressure']}-pressure)
Question type: {q_type}
Competency being assessed: {competency}

{"RECENT CONVERSATION CONTEXT:\n" + context_str if context_str else ""}

QUESTION: {question}

CANDIDATE ANSWER: {answer}

Evaluate with expert precision. Consider the {mode} mode — {"be lenient" if mode == "Casual" else "be rigorous" if mode == "Intense" else "be balanced"}.

Return ONLY this JSON (no markdown):
{{
  "score": <0-10 float>,
  "competency_score": <0-10 float for specifically '{competency}'>,
  "verdict": "<Exceptional|Strong|Solid|Average|Weak>",
  "strength": "<specific 1-sentence strength>",
  "weakness": "<specific 1-sentence gap or missing element>",
  "suggestion": "<concrete, actionable 1-sentence tip>",
  "star_feedback": "<1 sentence on STAR method use if behavioral, else empty string>",
  "tone_signals": ["<signal1>", "<signal2>", "<signal3>"],
  "needs_followup": <true|false>,
  "followup_question": "<a natural follow-up if score < {mode_cfg['followup_threshold']} or answer was vague, else empty string>",
  "Ketu_reaction": "<1 short sentence {persona['name']} would say after this answer, in first person, as {persona['name']}, conversational>",
  "ideal_answer_hint": "<1-2 sentence outline of what a strong answer would have included>"
}}

tone_signals: pick 3 from [Confident, Structured, Vague, Concise, Detailed, Nervous, Passionate, Hesitant, Analytical, Creative, Experienced, Rambling, Thoughtful, Unprepared, Authentic, Polished]
"""
    response = llm.invoke(prompt)
    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        result = json.loads(raw)
        result["score"] = min(10.0, max(0.0, float(result.get("score", 5))))
        result["competency_score"] = min(10.0, max(0.0, float(result.get("competency_score", result["score"]))))
        return result
    except Exception:
        return {
            "score": 5.0, "competency_score": 5.0, "verdict": "Average",
            "strength": "Answer was provided.",
            "weakness": "Could not fully evaluate.",
            "suggestion": "Try to be more specific and structured using the STAR method.",
            "star_feedback": "",
            "tone_signals": ["Thoughtful"],
            "needs_followup": False,
            "followup_question": "",
            "Ketu_reaction": "Thanks for sharing that.",
            "ideal_answer_hint": "Include a specific example with a measurable outcome.",
        }

# ============================================================
# POST-INTERVIEW SUMMARY
# ============================================================
def generate_summary(feedback_list: list, role: str, candidate_name: str, avg_score: float, persona_name: str, mode: str, llm) -> str:
    persona = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    qa_pairs = "\n\n".join([
        f"Q{i+1} [{item.get('type','?')} · {item.get('competency','?')}]: {item['q']}\nAnswer: {item['a'][:300]}…\nScore: {item['eval']['score']}/10 — {item['eval']['verdict']}"
        for i, item in enumerate(feedback_list)
    ])
    prompt = f"""You are {persona['name']}, a {persona['style']} AI interviewer writing a post-interview report.
Candidate: {candidate_name or 'the candidate'} | Role: {role} | Mode: {mode} | Overall: {avg_score:.1f}/10

INTERVIEW DATA:
{qa_pairs}

Write a structured post-interview assessment with exactly these 4 sections:

**OVERALL IMPRESSION**
2-3 sentences on general performance, calibre, and fit.

**KEY STRENGTHS**
2-3 sentences on the most impressive demonstrations across the interview.

**DEVELOPMENT AREAS**
2-3 sentences on specific gaps, with concrete references to answers.

**HIRING RECOMMENDATION**
1-2 sentences: clear recommendation (Strong Hire / Hire / Hold / No Hire) with brief rationale.

Use flowing prose, no bullet points. Professional but human. Be specific — reference actual answers.
Write as {persona['name']}, in first person.
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# ============================================================
# EXPORT HELPERS
# ============================================================
def build_json_export(state: dict) -> str:
    export = {
        "session_metadata": {
            "candidate": state.get("candidate_name", "Anonymous"),
            "role": state.get("role_title", ""),
            "mode": state.get("interview_mode", "Standard"),
            "persona": state.get("persona", "Ketu"),
            "date": datetime.now().isoformat(),
        },
        "summary_score": round(
            sum(s.get("score", 0) for s in state.get("scores", [])) / max(len(state.get("scores", [])), 1), 2
        ),
        "grade": grade_letter(
            sum(s.get("score", 0) for s in state.get("scores", [])) / max(len(state.get("scores", [])), 1)
        ),
        "qa_transcript": [
            {
                "question_num": i + 1,
                "question": item["q"],
                "type": item.get("type", ""),
                "competency": item.get("competency", ""),
                "difficulty": item.get("difficulty", ""),
                "answer": item["a"],
                "score": item["eval"].get("score", 0),
                "verdict": item["eval"].get("verdict", ""),
                "strength": item["eval"].get("strength", ""),
                "weakness": item["eval"].get("weakness", ""),
                "suggestion": item["eval"].get("suggestion", ""),
                "tone_signals": item["eval"].get("tone_signals", []),
                "time_seconds": item.get("time", 0),
            }
            for i, item in enumerate(state.get("feedback_list", []))
        ],
        "competency_scores": state.get("competency_scores", {}),
        "ai_assessment": state.get("ai_summary", ""),
    }
    return json.dumps(export, indent=2)

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
    return {
        "A+": "Outstanding — a rare calibre of candidate.",
        "A":  "Excellent performance — strong hire signal.",
        "B+": "Very good — above expectations in most areas.",
        "B":  "Solid candidate with clear strengths.",
        "C":  "Adequate but notable gaps remain.",
        "D":  "Significant development needed.",
    }.get(g, "Interview complete.")

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Cabinet Grotesk, sans-serif", color="#8899bb"),
    xaxis=dict(gridcolor="#111e30", zerolinecolor="#111e30"),
    yaxis=dict(gridcolor="#111e30", zerolinecolor="#111e30"),
    margin=dict(t=36, b=36, l=16, r=16),
)

# ============================================================
# SCREEN — SETUP
# ============================================================
def screen_setup():
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">⚡ Adaptive · Multi-Persona · Real-Time Intelligence</div>
        <div class="hero-title">KETU AI</div>
        <span class="hero-title-accent">next-generation interview intelligence</span>
        <p class="hero-sub">Meet your elite AI interviewer. Adaptive follow-ups, STAR analysis, competency mapping, and feedback that actually makes you better.</p>
        <div class="hero-stats">
            <div><div class="hero-stat-num">3</div><div class="hero-stat-label">Interviewer Personas</div></div>
            <div><div class="hero-stat-num">15+</div><div class="hero-stat-label">Competencies Tracked</div></div>
            <div><div class="hero-stat-num">∞</div><div class="hero-stat-label">Adaptive Follow-ups</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    llm = get_llm()
    if llm is None:
        st.error("⚠️ `GROQ_API_KEY` not found. Add it to `.streamlit/secrets.toml`.")
        return

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        # Persona selector
        st.markdown('<div class="panel panel-glow-cyan">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">🎭 Choose Your Interviewer</div>', unsafe_allow_html=True)

        persona_cols = st.columns(3)
        for i, (p_name, p_data) in enumerate(PERSONAS.items()):
            with persona_cols[i]:
                is_active = st.session_state.persona == p_name
                border_color = "rgba(0,229,255,0.5)" if is_active else "var(--border)"
                bg = "rgba(0,229,255,0.05)" if is_active else "var(--surface2)"
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {border_color};border-radius:12px;padding:1rem;text-align:center;margin-bottom:0.5rem">
                    <div style="font-size:1.8rem;margin-bottom:0.4rem">{p_data['avatar']}</div>
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.9rem;color:var(--text)">{p_name}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:var(--text3);margin-top:0.2rem">{p_data['title']}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"{'✓ Selected' if is_active else 'Select'}", key=f"persona_{p_name}", use_container_width=True):
                    st.session_state.persona = p_name
                    st.rerun()

        st.markdown("---")
        st.markdown('<div class="sec-label">📋 Job Context</div>', unsafe_allow_html=True)
        st.session_state.candidate_name = st.text_input("Your Name (optional)", placeholder="e.g. Arjun Mehta", value=st.session_state.candidate_name)
        st.session_state.role_title = st.text_input("Role / Job Title *", placeholder="e.g. Senior Backend Engineer", value=st.session_state.role_title)

        col_a, col_b = st.columns(2)
        with col_a:
            category = st.selectbox("Role Category", list(COMPETENCY_FRAMEWORKS.keys()), index=list(COMPETENCY_FRAMEWORKS.keys()).index(st.session_state.category_tag))
            st.session_state.category_tag = category
        with col_b:
            mode = st.selectbox("Interview Mode", list(INTERVIEW_MODES.keys()), index=list(INTERVIEW_MODES.keys()).index(st.session_state.interview_mode))
            st.session_state.interview_mode = mode

        mode_info = INTERVIEW_MODES[mode]
        st.markdown(f'<div class="tip-box">{mode_info["emoji"]} <b>{mode}</b>: {mode_info["desc"]}  ·  Follow-up threshold: ≥{mode_info["followup_threshold"]}/10</div>', unsafe_allow_html=True)

        st.session_state.jd_text = st.text_area("Job Description *", height=260, placeholder="Paste the full job description here…", value=st.session_state.jd_text)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel panel-glow-violet">', unsafe_allow_html=True)
        st.markdown('<div class="sec-label">📄 Your Resume</div>', unsafe_allow_html=True)
        resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf","docx","doc","txt"], label_visibility="collapsed")
        if resume_file:
            with st.spinner("Reading resume…"):
                st.session_state.resume_text = load_document(resume_file)
            words = len(st.session_state.resume_text.split())
            st.success(f"✅ Resume loaded — {words:,} words extracted")
            with st.expander("Preview extracted text"):
                st.text(st.session_state.resume_text[:800] + "…")

        st.markdown("---")
        st.markdown('<div class="sec-label">⚙️ Session Settings</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.num_questions = st.slider("Questions", 4, 15, st.session_state.num_questions)
        with col2:
            st.session_state.tts_enabled = st.toggle("🔊 Voice TTS", value=st.session_state.tts_enabled)

        # Competency preview
        if st.session_state.category_tag:
            comps = COMPETENCY_FRAMEWORKS.get(st.session_state.category_tag, [])
            comp_html = "".join([f'<span class="competency-pill">{c}</span>' for c in comps])
            st.markdown(f'<div class="sec-label" style="margin-top:1rem">📊 Competencies to Assess</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:0.25rem;margin-bottom:1rem">{comp_html}</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="tip-box">
        🧠 KETU AI tracks STAR method usage, filler word density, answer verbosity, and competency scores across all your answers — giving you data no human interviewer would share.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        persona = PERSONAS.get(st.session_state.persona, PERSONAS["Ketu"])
        if st.button(f"🚀  Begin Interview with {persona['name']}", use_container_width=True):
            if not st.session_state.jd_text.strip():
                st.error("Please paste a job description.")
            elif not st.session_state.resume_text.strip():
                st.error("Please upload a resume.")
            elif not st.session_state.role_title.strip():
                st.error("Please enter the role / job title.")
            else:
                with st.spinner(f"🤖 {persona['name']} is reviewing your profile and crafting tailored questions…"):
                    qs, types, comps_list, diffs = generate_questions(
                        st.session_state.jd_text,
                        st.session_state.resume_text,
                        st.session_state.role_title,
                        st.session_state.num_questions,
                        llm,
                        st.session_state.persona,
                        st.session_state.interview_mode,
                        st.session_state.category_tag,
                    )
                if not qs:
                    st.error("Could not generate questions. Check your API key.")
                else:
                    greeting = random.choice(persona["greetings"])
                    st.session_state.questions         = qs
                    st.session_state.q_types           = types
                    st.session_state.q_competencies    = comps_list
                    st.session_state.q_difficulties    = diffs
                    st.session_state.current           = 0
                    st.session_state.scores            = []
                    st.session_state.feedback_list     = []
                    st.session_state.time_per_q        = []
                    st.session_state.session_start     = time.time()
                    st.session_state.q_start           = time.time()
                    st.session_state.submitted         = False
                    st.session_state.Ketu_message      = greeting
                    st.session_state.is_followup       = False
                    st.session_state.followup_count    = 0
                    st.session_state.transcript        = []
                    st.session_state.competency_scores = {}
                    st.session_state.filler_word_counts= []
                    st.session_state.word_counts       = []
                    st.session_state.star_analyses     = []
                    st.session_state.ai_summary        = None
                    st.session_state.screen            = "interview"
                    st.rerun()

# ============================================================
# SCREEN — INTERVIEW
# ============================================================
def screen_interview():
    llm = get_llm()
    idx  = st.session_state.current
    questions    = st.session_state.questions
    q_types      = st.session_state.q_types
    q_comps      = st.session_state.q_competencies
    q_diffs      = st.session_state.q_difficulties
    persona      = PERSONAS.get(st.session_state.persona, PERSONAS["Ketu"])
    mode         = st.session_state.interview_mode
    mode_cfg     = INTERVIEW_MODES.get(mode, INTERVIEW_MODES["Standard"])
    n = len(questions)

    if idx >= n:
        st.session_state.screen = "results"
        st.rerun()

    q          = questions[idx]
    q_type     = q_types[idx]     if idx < len(q_types)  else "technical"
    competency = q_comps[idx]     if idx < len(q_comps)  else "Technical Depth"
    difficulty = q_diffs[idx]     if idx < len(q_diffs)  else "medium"
    q_info     = QUESTION_TYPES.get(q_type, ("❓", "badge-technical", q_type.title()))
    diff_color = {"easy": "var(--emerald)", "medium": "var(--amber)", "hard": "var(--rose)"}.get(difficulty, "var(--text3)")
    diff_icon  = {"easy": "▸", "medium": "▸▸", "hard": "▸▸▸"}.get(difficulty, "▸▸")

    # ── Top bar ──────────────────────────────────────────────
    tb1, tb2, tb3, tb4, tb5 = st.columns([4, 1, 1, 1, 1])
    with tb1:
        st.progress(idx / n)
        elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
        mins, secs = divmod(elapsed, 60)
        st.caption(f"Q{idx+1}/{n}  ·  {mins:02d}:{secs:02d}  ·  {mode} mode  ·  {persona['name']}")
    with tb2:
        avg = (sum(s.get("score",0) for s in st.session_state.scores) / len(st.session_state.scores)) if st.session_state.scores else 0.0
        st.metric("Avg", f"{avg:.1f}")
    with tb3:
        st.metric("Done", f"{len(st.session_state.scores)}/{n}")
    with tb4:
        total_words = sum(st.session_state.word_counts) if st.session_state.word_counts else 0
        st.metric("Words", f"{total_words}")
    with tb5:
        if st.button("⏹", help="End interview"):
            st.session_state.screen = "results"
            st.rerun()

    st.markdown("---")

    # ── Persona avatar ───────────────────────────────────────
    Ketu_msg = st.session_state.get("Ketu_message", "")
    is_followup = st.session_state.get("is_followup", False)
    speaking_class = "speaking" if Ketu_msg and not st.session_state.submitted else ""

    st.markdown(f"""
    <div class="interviewer-wrap">
        <div class="avatar-ring {speaking_class}">{persona['avatar']}</div>
        <div class="interviewer-meta">
            <div class="interviewer-name">{persona['name']}</div>
            <div class="interviewer-status"><span class="status-dot"></span> {persona['title']}</div>
        </div>
        <div class="interviewer-speech"><span class="speech-quote">"</span>{Ketu_msg or 'Ready for your answer…'}<span class="speech-quote">"</span></div>
    </div>
    """, unsafe_allow_html=True)

    if Ketu_msg and f"tts_done_{idx}_{Ketu_msg[:20]}" not in st.session_state:
        tts_autoplay(Ketu_msg)
        st.session_state[f"tts_done_{idx}_{Ketu_msg[:20]}"] = True

    # ── Question card ─────────────────────────────────────────
    if is_followup:
        st.markdown('<div class="followup-badge">🔄 Follow-up — Ketu wants to dig deeper</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="q-card">
        <div class="q-num">Question {idx+1} of {n}</div>
        <p class="q-text">{q}</p>
        <div class="q-meta">
            <span class="q-type-badge {q_info[1]}">{q_info[0]} {q_info[2]}</span>
            <span class="q-competency">📊 {competency}</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.62rem;color:{diff_color}">{diff_icon} {difficulty.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Answer area ───────────────────────────────────────────
    if not st.session_state.submitted:
        # Voice recording
        if HAS_AUDIO_RECORDER:
            st.markdown('<div class="sec-label">🎙️ Voice Answer</div>', unsafe_allow_html=True)
            st.markdown('<div class="rec-strip"><span class="rec-label"><span class="rec-dot"></span> Click to record · Powered by Groq Whisper</span><span style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:var(--text3)">Speak clearly for best accuracy</span></div>', unsafe_allow_html=True)
            audio_bytes = audio_recorder(text="", icon_size="2x", key=f"rec_{idx}")
            if audio_bytes and f"transcribed_{idx}" not in st.session_state:
                st.markdown(build_waveform(), unsafe_allow_html=True)
                with st.spinner("Transcribing…"):
                    text = transcribe_voice(audio_bytes)
                    if text:
                        st.session_state[f"answer_{idx}"] = text
                        st.session_state[f"transcribed_{idx}"] = True
                        st.rerun()

        st.markdown('<div class="sec-label">✍️ Written Answer</div>', unsafe_allow_html=True)
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

        # Live coaching analysis
        if ans.strip():
            qa = analyze_answer_quality(ans)
            hint_type, hint_icon, hint_text = qa["coaching_hint"]
            wc = qa["word_count"]
            # Word meter
            ideal_pct = min(wc / 250, 1.0)
            meter_color = "#10b981" if 80 <= wc <= 250 else "#f59e0b" if wc < 80 else "#f43f5e"
            st.markdown(f"""
            <div class="word-meter">
                <span class="word-meter-label">{wc} words</span>
                <div class="word-meter-bar"><div class="word-meter-fill" style="width:{ideal_pct*100:.0f}%;background:{meter_color}"></div></div>
                <span class="word-meter-label" style="color:{meter_color}">{'Ideal ✓' if 80<=wc<=250 else 'Too short' if wc<80 else 'Too long'}</span>
            </div>
            <div class="coaching-bar {hint_type}"><span class="coaching-icon">{hint_icon}</span>{hint_text}</div>
            """, unsafe_allow_html=True)

            # STAR quick indicator for behavioral
            if q_type in ("behavioral", "situational") and wc > 30:
                star = qa["star_signals"]
                star_html = "".join([
                    f'<div class="star-cell"><div class="star-cell-label">{k}</div><div class="star-cell-value {"star-found" if v else "star-missing"}">'
                    f'{"✓" if v else "○"}</div></div>'
                    for k, v in star.items()
                ])
                st.markdown(f'<div style="margin-top:0.6rem"><div class="sec-label" style="margin-bottom:0.4rem">⭐ STAR Check</div><div class="star-grid">{star_html}</div></div>', unsafe_allow_html=True)
        else:
            tip_map = {
                "technical":   "💡 Be specific — mention tools, architectures, and measurable outcomes.",
                "behavioral":  "💡 Use the STAR method: Situation · Task · Action · Result.",
                "rapport":     "💡 Be authentic — this is about knowing you, not testing you.",
                "situational": "💡 Walk through your thinking step-by-step. Trade-offs matter.",
                "ambition":    "💡 Connect your goals directly to what excites you about this role.",
            }
            st.markdown(f'<div class="tip-box">{tip_map.get(q_type, "💡 Take your time and be specific.")}</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            submit = st.button("✓  Submit Answer", use_container_width=True)
        with col2:
            skip = st.button("Skip →", use_container_width=True)
        with col3:
            if st.button("💡 Hint", use_container_width=True, help="See a coaching tip for this question"):
                st.session_state[f"show_hint_{idx}"] = not st.session_state.get(f"show_hint_{idx}", False)
                st.rerun()

        if st.session_state.get(f"show_hint_{idx}", False):
            st.markdown(f"""
            <div class="coaching-bar good">
                <span class="coaching-icon">🎯</span>
                For a <b>{q_type}</b> question on <b>{competency}</b>, focus on: specifics over generalities, quantified outcomes, and what YOU personally did vs what the team did.
            </div>
            """, unsafe_allow_html=True)

        if skip:
            st.session_state.transcript.append({"role": "user", "content": "[Skipped]", "q": q})
            st.session_state.current   += 1
            st.session_state.submitted  = False
            st.session_state.is_followup = False
            st.session_state.Ketu_message = random.choice(persona["transitions"])
            st.session_state.q_start = time.time()
            st.rerun()

        if submit:
            if not ans.strip():
                st.warning("Please provide an answer before submitting.")
            else:
                # Pre-compute quality analysis
                qa = analyze_answer_quality(ans)
                st.session_state.filler_word_counts.append(qa["filler_count"])
                st.session_state.word_counts.append(qa["word_count"])

                with st.spinner(f"{persona['name']} is analysing your response…"):
                    eval_res = evaluate_answer(
                        q, ans, st.session_state.role_title, q_type, competency, mode,
                        st.session_state.persona, llm,
                        st.session_state.transcript[-6:]
                    )
                    eval_res["_qa_analysis"] = qa  # attach local analysis

                st.session_state.transcript.append({"role": "user",  "content": ans, "q": q})
                st.session_state.transcript.append({"role": persona["name"], "content": eval_res.get("Ketu_reaction", "")})

                if not is_followup:
                    st.session_state.scores.append(eval_res)
                    st.session_state.feedback_list.append({
                        "q": q, "a": ans, "eval": eval_res,
                        "type": q_type, "competency": competency, "difficulty": difficulty,
                        "time": int(time.time() - (st.session_state.q_start or time.time())),
                        "qa": qa,
                    })
                    # Update competency scores
                    comp_sc = eval_res.get("competency_score", eval_res.get("score", 5.0))
                    if competency not in st.session_state.competency_scores:
                        st.session_state.competency_scores[competency] = []
                    st.session_state.competency_scores[competency].append(comp_sc)
                else:
                    if st.session_state.scores:
                        prev = st.session_state.scores[-1]
                        new_sc = min(10.0, (prev["score"] + eval_res["score"]) / 2 + 0.5)
                        st.session_state.scores[-1]["score"] = new_sc

                st.session_state.current_feedback = eval_res
                st.session_state.submitted = True
                needs_followup = (
                    eval_res.get("needs_followup", False)
                    and eval_res.get("followup_question", "")
                    and st.session_state.followup_count < mode_cfg["max_followups"]
                    and not is_followup
                )
                st.session_state._pending_followup = needs_followup
                st.rerun()

    # ── Feedback view ─────────────────────────────────────────
    else:
        f = st.session_state.current_feedback
        sc = f.get("score", 5.0)
        sc_class = score_class(sc)
        tones = f.get("tone_signals", [])
        Ketu_react = f.get("Ketu_reaction", "")
        qa_local = f.get("_qa_analysis", {})

        # Persona reaction
        if Ketu_react:
            st.markdown(f"""
            <div class="interviewer-wrap">
                <div class="avatar-ring">{persona['avatar']}</div>
                <div class="interviewer-meta">
                    <div class="interviewer-name">{persona['name']}</div>
                    <div class="interviewer-status"><span class="status-dot"></span> Reviewing your answer</div>
                </div>
                <div class="interviewer-speech"><span class="speech-quote">"</span>{Ketu_react}<span class="speech-quote">"</span></div>
            </div>
            """, unsafe_allow_html=True)
            if f"tts_react_{idx}" not in st.session_state:
                tts_autoplay(Ketu_react)
                st.session_state[f"tts_react_{idx}"] = True

        # Score ring + verdict
        tone_chips_html = ""
        for t in tones:
            cls = "positive" if t in POSITIVE_TONE_SIGNALS else "negative" if t in NEGATIVE_TONE_SIGNALS else ""
            tone_chips_html += f'<span class="tone-chip {cls}">{t}</span>'

        ring_svg = score_ring_svg(sc)
        st.markdown(f"""
        <div class="feedback-card">
            <div class="score-display">
                <div class="score-ring-wrap">
                    {ring_svg}
                    <div class="score-ring-num {sc_class}">{sc:.1f}</div>
                </div>
                <div>
                    <div class="verdict-text">{f.get('verdict','Average')}</div>
                    <div class="verdict-sub">{competency} · out of 10</div>
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
            {f'<div class="feedback-section"><div class="feedback-label label-star">⭐ STAR Analysis</div><div class="feedback-text">{f.get("star_feedback","")}</div></div>' if f.get('star_feedback') and q_type in ('behavioral','situational') else ''}
        </div>
        """, unsafe_allow_html=True)

        # Show ideal answer hint
        ideal = f.get("ideal_answer_hint", "")
        if ideal:
            with st.expander("💡 What a strong answer would have included"):
                st.markdown(f'<div class="tip-box" style="color:rgba(139,92,246,0.65);border-color:rgba(139,92,246,0.15)">{ideal}</div>', unsafe_allow_html=True)

        # Show local quality stats
        if qa_local:
            wc = qa_local.get("word_count", 0)
            fc = qa_local.get("filler_count", 0)
            star_score = qa_local.get("star_score", 0)
            cols = st.columns(3)
            cols[0].metric("Words", f"{wc}")
            cols[1].metric("Filler Words", f"{fc}")
            cols[2].metric("STAR Coverage", f"{star_score}/4")

        # Follow-up logic
        pending_followup = st.session_state.get("_pending_followup", False)
        fq = f.get("followup_question", "")
        if pending_followup and fq:
            st.markdown(f'<div class="tip-box" style="color:rgba(244,63,94,0.6);border-color:rgba(244,63,94,0.15)">🔍 {persona["name"]} wants to explore this further…</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄  Answer Follow-up", use_container_width=True):
                    st.session_state.questions.insert(idx + 1, fq)
                    st.session_state.q_types.insert(idx + 1, q_type)
                    st.session_state.q_competencies.insert(idx + 1, competency)
                    st.session_state.q_difficulties.insert(idx + 1, "hard")
                    st.session_state.current     += 1
                    st.session_state.submitted    = False
                    st.session_state.is_followup  = True
                    st.session_state.followup_count += 1
                    st.session_state._pending_followup = False
                    st.session_state.Ketu_message = f"Good. Let me push on this. {fq}"
                    st.session_state.q_start = time.time()
                    st.rerun()
            with c2:
                if st.button("Skip Follow-up →", use_container_width=True):
                    st.session_state.current    += 1
                    st.session_state.submitted   = False
                    st.session_state.is_followup = False
                    st.session_state._pending_followup = False
                    st.session_state.Ketu_message = random.choice(persona["transitions"])
                    st.session_state.q_start = time.time()
                    st.rerun()
        else:
            next_label = "Finish Interview →" if idx + 1 >= n else f"Next Question → Q{idx+2}"
            if st.button(next_label, use_container_width=True):
                st.session_state.current     += 1
                st.session_state.submitted    = False
                st.session_state.is_followup  = False
                st.session_state._pending_followup = False
                st.session_state.Ketu_message = random.choice(persona["transitions"])
                st.session_state.q_start = time.time()
                st.rerun()

# ============================================================
# SCREEN — RESULTS
# ============================================================
def screen_results():
    llm = get_llm()
    scores        = st.session_state.scores
    feedback_list = st.session_state.feedback_list
    persona       = PERSONAS.get(st.session_state.persona, PERSONAS["Ketu"])

    if not scores:
        st.warning("No answers were recorded.")
        if st.button("Start Over"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
        return

    avg_score = sum(s.get("score",0) for s in scores) / len(scores)
    grade     = grade_letter(avg_score)
    g_class   = grade_class(grade)
    tagline   = grade_tagline(grade)
    name      = st.session_state.candidate_name or "Candidate"
    role      = st.session_state.role_title
    elapsed   = int(time.time() - (st.session_state.session_start or time.time()))
    mins      = elapsed // 60
    n_total   = len(st.session_state.questions)
    mode      = st.session_state.interview_mode

    total_words   = sum(st.session_state.word_counts) if st.session_state.word_counts else 0
    total_fillers = sum(st.session_state.filler_word_counts) if st.session_state.filler_word_counts else 0
    avg_words     = total_words // max(len(st.session_state.word_counts), 1)
    star_scores   = [item.get("qa", {}).get("star_score", 0) for item in feedback_list if item.get("qa")]
    avg_star      = sum(star_scores) / max(len(star_scores), 1)

    # ── Hero ─────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-hero">
        <div class="hero-eyebrow">Interview Complete · {name} · {role} · {mode} Mode · {persona['name']}</div>
        <div class="result-grade {g_class}">{grade}</div>
        <div class="result-label">Final Grade · {avg_score:.1f} / 10</div>
        <div class="result-tagline">{tagline}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics row ───────────────────────────────────────────
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Score", f"{avg_score:.1f}/10")
    m2.metric("Answered", f"{len(scores)}/{n_total}")
    m3.metric("Duration", f"{mins}m")
    m4.metric("Avg Words", f"{avg_words}")
    m5.metric("Filler Words", f"{total_fillers}")
    m6.metric("STAR Avg", f"{avg_star:.1f}/4")

    st.markdown("---")

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        # ── Score timeline ────────────────────────────────────
        st.markdown('<div class="sec-label">📈 Score Timeline</div>', unsafe_allow_html=True)
        if len(scores) >= 2:
            vals     = [s.get("score",0) for s in scores]
            q_labels = [f"Q{i+1}" for i in range(len(scores))]
            fig_line = go.Figure()
            # Fill area
            fig_line.add_trace(go.Scatter(
                x=q_labels, y=vals, mode="lines+markers",
                line=dict(color="#00e5ff", width=2.5, shape="spline"),
                marker=dict(size=10, color=vals,
                    colorscale=[[0,"#f43f5e"],[0.5,"#f59e0b"],[1,"#10b981"]],
                    line=dict(color="#010306", width=2.5), symbol="circle"),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.05)",
                name="Score",
            ))
            fig_line.add_hline(y=avg_score, line_dash="dot", line_color="rgba(0,229,255,0.35)",
                annotation_text=f"avg {avg_score:.1f}", annotation_font_color="#00e5ff")
            fig_line.update_layout(**{**PLOTLY_LAYOUT, "height": 240, "showlegend": False,
                "yaxis": {**PLOTLY_LAYOUT.get("yaxis",{}), "range": [0, 10.5]},
                "xaxis": {**PLOTLY_LAYOUT.get("xaxis",{})}})
            st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})

        # ── Word count per Q ──────────────────────────────────
        if st.session_state.word_counts and len(st.session_state.word_counts) >= 2:
            st.markdown('<div class="sec-label">📝 Words per Answer</div>', unsafe_allow_html=True)
            wc_vals = st.session_state.word_counts
            wc_labels = [f"Q{i+1}" for i in range(len(wc_vals))]
            fig_wc = go.Figure(go.Bar(
                x=wc_labels, y=wc_vals,
                marker_color=["#10b981" if 80<=w<=250 else "#f59e0b" if w<80 else "#f43f5e" for w in wc_vals],
                marker_line_width=0,
            ))
            fig_wc.add_hline(y=80,  line_dash="dot", line_color="rgba(16,185,129,0.3)", annotation_text="min ideal", annotation_font_color="rgba(16,185,129,0.5)", annotation_font_size=10)
            fig_wc.add_hline(y=250, line_dash="dot", line_color="rgba(244,63,94,0.3)",  annotation_text="max ideal", annotation_font_color="rgba(244,63,94,0.5)",  annotation_font_size=10)
            fig_wc.update_layout(**PLOTLY_LAYOUT, height=180, showlegend=False)
            st.plotly_chart(fig_wc, use_container_width=True, config={"displayModeBar": False})

        # ── Q-by-Q breakdown ──────────────────────────────────
        st.markdown('<div class="sec-label">📋 Question Breakdown</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        for i, item in enumerate(feedback_list):
            sc = item["eval"].get("score", 0)
            verdict = item["eval"].get("verdict", "—")
            q_t = item.get("type","technical")
            q_info = QUESTION_TYPES.get(q_t, ("❓","badge-technical",q_t.title()))
            comp = item.get("competency","—")
            diff = item.get("difficulty","medium")
            diff_color = {"easy":"#10b981","medium":"#f59e0b","hard":"#f43f5e"}.get(diff,"#7a93bb")
            time_taken = item.get("time", 0)
            score_color = "#10b981" if sc >= 7 else "#f59e0b" if sc >= 5 else "#f43f5e"
            wc = item.get("qa", {}).get("word_count", 0)
            fc = item.get("qa", {}).get("filler_count", 0)

            st.markdown(f"""
            <div class="q-timeline-item">
                <div class="q-timeline-num">Q{i+1}</div>
                <div class="q-timeline-content">
                    <div class="q-timeline-q">{item['q'][:90]}{'…' if len(item['q'])>90 else ''}</div>
                    <div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-top:0.3rem;align-items:center">
                        <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:{score_color};background:rgba(0,0,0,0.2);border:1px solid {score_color}40;padding:0.18rem 0.55rem;border-radius:99px">
                            {sc:.1f}/10 · {verdict}
                        </span>
                        <span class="q-type-badge {q_info[1]}">{q_info[0]} {q_info[2]}</span>
                        <span class="q-competency">{comp}</span>
                        <span style="font-family:'DM Mono',monospace;font-size:0.6rem;color:{diff_color}">{diff.upper()}</span>
                        {f'<span style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:var(--text3)">⏱ {time_taken}s · {wc}w · {fc} fillers</span>' if wc else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            with st.expander(f"Full feedback — Q{i+1}: {item['q'][:50]}…"):
                st.write(f"**Answer:** {item['a']}")
                tones = item['eval'].get('tone_signals', [])
                if tones:
                    st.write(f"**Tone signals:** {' · '.join(tones)}")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.success(f"**Strength:** {item['eval'].get('strength','—')}")
                    st.info(f"**Suggestion:** {item['eval'].get('suggestion','—')}")
                with col_b:
                    st.error(f"**Gap:** {item['eval'].get('weakness','—')}")
                    if item['eval'].get('ideal_answer_hint'):
                        st.markdown(f'<div class="tip-box">💡 **Ideal:** {item["eval"]["ideal_answer_hint"]}</div>', unsafe_allow_html=True)
                # STAR analysis
                qa = item.get("qa", {})
                if qa.get("star_signals") and item.get("type") in ("behavioral","situational"):
                    star = qa["star_signals"]
                    star_html = "".join([
                        f'<div class="star-cell"><div class="star-cell-label">{k}</div><div class="star-cell-value {"star-found" if v else "star-missing"}">{"✓" if v else "○"}</div></div>'
                        for k, v in star.items()
                    ])
                    st.markdown(f'<div class="star-grid" style="max-width:300px">{star_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # ── Competency radar ──────────────────────────────────
        comp_scores_agg = {k: sum(v)/len(v) for k, v in st.session_state.competency_scores.items() if v}
        if len(comp_scores_agg) >= 3:
            st.markdown('<div class="sec-label">🕸️ Competency Radar</div>', unsafe_allow_html=True)
            cats   = list(comp_scores_agg.keys())
            vals_r = list(comp_scores_agg.values())
            cats_c = cats + [cats[0]]
            vals_c = vals_r + [vals_r[0]]
            fig_r = go.Figure(go.Scatterpolar(
                r=vals_c, theta=cats_c, fill="toself",
                fillcolor="rgba(0,229,255,0.07)",
                line=dict(color="#00e5ff", width=2.5),
                marker=dict(color="#00e5ff", size=7),
            ))
            fig_r.update_layout(**{**PLOTLY_LAYOUT, "polar": dict(
                bgcolor="rgba(0,0,0,0)",
                angularaxis=dict(color="#2d4464", gridcolor="#111e30", tickfont=dict(size=10)),
                radialaxis=dict(range=[0,10], color="#2d4464", gridcolor="#111e30"),
            )}, height=320)
            st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

            # Competency score list
            for comp, s in sorted(comp_scores_agg.items(), key=lambda x: -x[1]):
                bar_color = "#10b981" if s >= 7 else "#f59e0b" if s >= 5 else "#f43f5e"
                bar_pct = s / 10 * 100
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.5rem">
                    <div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:var(--text2);min-width:120px">{comp}</div>
                    <div style="flex:1;height:3px;background:var(--border);border-radius:99px;overflow:hidden">
                        <div style="height:100%;width:{bar_pct:.0f}%;background:{bar_color};border-radius:99px"></div>
                    </div>
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:0.8rem;color:{bar_color};min-width:28px">{s:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Score distribution ────────────────────────────────
        st.markdown('<div class="sec-label" style="margin-top:1.5rem">📊 Score Distribution</div>', unsafe_allow_html=True)
        if scores:
            vals = [s.get("score",0) for s in scores]
            bins = {"0–4": 0, "5–6": 0, "7–8": 0, "9–10": 0}
            for v in vals:
                if v <= 4: bins["0–4"] += 1
                elif v <= 6: bins["5–6"] += 1
                elif v <= 8: bins["7–8"] += 1
                else: bins["9–10"] += 1
            fig_b = go.Figure(go.Bar(
                x=list(bins.keys()), y=list(bins.values()),
                marker_color=["#f43f5e","#f59e0b","#00e5ff","#10b981"],
                marker_line_width=0, text=list(bins.values()), textposition="outside",
                textfont=dict(color="#7a93bb", size=11),
            ))
            fig_b.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False)
            st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar": False})

        # ── Filler word analysis ──────────────────────────────
        if total_fillers > 0 or total_words > 0:
            st.markdown('<div class="sec-label" style="margin-top:1.5rem">🎙️ Communication Quality</div>', unsafe_allow_html=True)
            filler_pct = (total_fillers / max(total_words, 1)) * 100
            filler_color = "#10b981" if filler_pct < 3 else "#f59e0b" if filler_pct < 6 else "#f43f5e"
            filler_label = "Excellent" if filler_pct < 3 else "Acceptable" if filler_pct < 6 else "High — needs work"
            st.markdown(f"""
            <div class="panel" style="padding:1.5rem">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem">
                    <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:var(--text3)">FILLER DENSITY</div>
                    <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.1rem;color:{filler_color}">{filler_pct:.1f}% · {filler_label}</div>
                </div>
                <div style="height:4px;background:var(--border);border-radius:99px;overflow:hidden">
                    <div style="height:100%;width:{min(filler_pct/10*100,100):.0f}%;background:{filler_color};border-radius:99px"></div>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:var(--text3);margin-top:0.6rem">
                    {total_fillers} filler words across {total_words} total words · avg {avg_words} words/answer
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── AI Summary ────────────────────────────────────────
        st.markdown('<div class="sec-label" style="margin-top:1.5rem">🤖 {persona["name"]}\'s Assessment</div>', unsafe_allow_html=True)
        if "ai_summary" not in st.session_state or not st.session_state.ai_summary:
            with st.spinner(f"{persona['name']} is writing your assessment…"):
                st.session_state.ai_summary = generate_summary(
                    feedback_list, role, name, avg_score,
                    st.session_state.persona, mode, llm
                )
        st.markdown(f"""
        <div class="panel panel-glow-violet">
            <div style="font-family:'Fraunces',serif;font-style:italic;font-size:0.96rem;color:var(--text2);line-height:1.8;">
                {st.session_state.ai_summary.replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Export + Actions ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="sec-label">⬇️ Export & Actions</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        json_data = build_json_export(st.session_state)
        st.download_button(
            "📦 Download JSON Report",
            data=json_data,
            file_name=f"ketu_interview_{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        if st.button("🔄  New Interview", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    with col3:
        if st.button("📋  Same Role Again", use_container_width=True):
            resume = st.session_state.resume_text
            jd     = st.session_state.jd_text
            role_t = st.session_state.role_title
            cat    = st.session_state.category_tag
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.resume_text  = resume
            st.session_state.jd_text      = jd
            st.session_state.role_title   = role_t
            st.session_state.category_tag = cat
            st.rerun()
    with col4:
        if st.button("🔥  Intense Mode", use_container_width=True, help="Retry in Intense mode"):
            resume = st.session_state.resume_text
            jd     = st.session_state.jd_text
            role_t = st.session_state.role_title
            cat    = st.session_state.category_tag
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.session_state.resume_text   = resume
            st.session_state.jd_text       = jd
            st.session_state.role_title    = role_t
            st.session_state.category_tag  = cat
            st.session_state.interview_mode = "Intense"
            st.rerun()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:1.9rem;color:#00e5ff;margin-bottom:0.2rem;letter-spacing:-0.02em">KETU AI</div>', unsafe_allow_html=True)
    st.caption("Elite · Adaptive · Intelligent")
    st.markdown("---")

    screen = st.session_state.get("screen", "setup")
    persona = PERSONAS.get(st.session_state.get("persona","Ketu"), PERSONAS["Ketu"])

    if screen == "interview":
        idx = st.session_state.current
        n   = len(st.session_state.questions)
        st.progress(idx / n if n else 0)
        c1, c2 = st.columns(2)
        c1.metric("Question", f"{idx}/{n}")
        if st.session_state.scores:
            avg = sum(s.get("score",0) for s in st.session_state.scores) / len(st.session_state.scores)
            c2.metric("Grade", grade_letter(avg))
        st.markdown(f"**Interviewer:** {persona['avatar']} {persona['name']}")
        st.markdown(f"**Mode:** {st.session_state.interview_mode}")
        st.markdown("---")
        if st.button("⏹ End Interview", use_container_width=True):
            st.session_state.screen = "results"
            st.rerun()

    elif screen == "results":
        if st.session_state.scores:
            avg = sum(s.get("score",0) for s in st.session_state.scores) / len(st.session_state.scores)
            st.success(f"Interview complete · {grade_letter(avg)}")
            st.metric("Final Score", f"{avg:.1f}/10")

    st.markdown("---")
    st.markdown("##### ⚡ Capabilities")
    feats = [
        "3 distinct interviewer personas",
        "6 role competency frameworks",
        "3 interview pressure modes",
        "Adaptive follow-up intelligence",
        "Live STAR method tracking",
        "Filler word density analysis",
        "Answer verbosity coaching",
        "Real-time word count meter",
        "Competency radar charts",
        "Ideal answer hints post-answer",
        "Groq Whisper voice input",
        "TTS question delivery",
        "JSON export with full report",
        "Llama 3.3-70B via Groq",
    ]
    for f in feats:
        st.markdown(f"<div style='font-family:DM Mono,monospace;font-size:0.68rem;color:#3d5278;padding:0.18rem 0'>· {f}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(f"{datetime.now().strftime('%H:%M · %d %b %Y')}")

# ============================================================
# ROUTER
# ============================================================
if st.session_state.screen == "setup":
    screen_setup()
elif st.session_state.screen == "interview":
    screen_interview()
elif st.session_state.screen == "results":
    screen_results()
