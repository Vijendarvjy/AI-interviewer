# ============================================================
# KETU AI v2.1 — ELITE INTERVIEW INTELLIGENCE
# Upgraded: Advanced Live Camera with TensorFlow.js BlazeFace,
# Eye Contact Tracking, Expression Analysis, Posture Scoring,
# Live Coaching Overlays, Confidence Meter + All v2.0 Features
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
import re, io, time, base64, tempfile, json, random, hashlib, math
from io import BytesIO
from datetime import datetime
from collections import Counter, defaultdict

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
    page_title="KETU AI · Elite Interviewer v2.1",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# DESIGN SYSTEM v2.1 — Refined Dark Luxury
# ============================================================
DESIGN = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist+Mono:wght@300;400;500;600&family=Geist:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg:           #02040a;
    --bg-elevated:  #060c18;
    --surface:      #080f1f;
    --surface2:     #0b1628;
    --surface3:     #0f1e35;
    --surface4:     #132442;
    --b1:           #0e1a2e;
    --b2:           #14253e;
    --b3:           #1a3050;
    --b4:           #213c62;
    --electric:     #00d4ff;
    --electric-dim: rgba(0,212,255,0.55);
    --electric-ghost: rgba(0,212,255,0.08);
    --neon:         #7c3aed;
    --neon-dim:     rgba(124,58,237,0.55);
    --neon-ghost:   rgba(124,58,237,0.08);
    --plasma:       #f0abfc;
    --fire:         #fb923c;
    --acid:         #a3e635;
    --crimson:      #fb2c36;
    --jade:         #00c896;
    --gold:         #fbbf24;
    --silver:       #94a3b8;
    --t1:           #f0f4ff;
    --t2:           #8a9fc4;
    --t3:           #3d5580;
    --t4:           #1e3258;
    --r1:           8px;
    --r2:           12px;
    --r3:           16px;
    --r4:           24px;
    --r5:           32px;
    --glow-e:   0 0 60px rgba(0,212,255,0.1), 0 0 120px rgba(0,212,255,0.05);
    --glow-n:   0 0 60px rgba(124,58,237,0.1), 0 0 120px rgba(124,58,237,0.05);
    --glow-c:   0 0 40px rgba(0,200,150,0.12);
    --glow-r:   0 0 40px rgba(251,44,54,0.1);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--t1) !important;
    font-family: 'Geist', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 120% 50% at 0% -10%,  rgba(0,212,255,0.035)  0%, transparent 50%),
        radial-gradient(ellipse 80%  60% at 100% 110%, rgba(124,58,237,0.04) 0%, transparent 50%),
        radial-gradient(ellipse 60%  40% at 50% 50%,  rgba(6,12,24,0.6)     0%, transparent 100%),
        var(--bg) !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image:
        linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 60px 60px;
    opacity: 0.5;
}

[data-testid="stHeader"], footer, #MainMenu { display: none !important; }

[data-testid="stSidebar"] {
    background: var(--bg-elevated) !important;
    border-right: 1px solid var(--b2) !important;
}
[data-testid="stSidebar"] * { font-family: 'Geist', sans-serif !important; }

h1, h2, h3 { font-family: 'Geist', sans-serif !important; font-weight: 800 !important; }

::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--b4); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--electric-dim); }

.stButton > button {
    background: var(--surface2) !important;
    border: 1px solid var(--b3) !important;
    color: var(--t2) !important;
    border-radius: var(--r2) !important;
    font-family: 'Geist Mono', monospace !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    padding: 0.6rem 1.4rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    position: relative !important; overflow: hidden !important;
}
.stButton > button:hover {
    border-color: var(--electric) !important;
    color: var(--electric) !important;
    background: rgba(0,212,255,0.05) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.12), inset 0 0 20px rgba(0,212,255,0.03) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.stTextArea textarea, .stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--b2) !important;
    border-radius: var(--r2) !important;
    color: var(--t1) !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.84rem !important;
    line-height: 1.7 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(0,212,255,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.08), 0 0 20px rgba(0,212,255,0.06) !important;
    outline: none !important;
}
.stTextArea label, .stTextInput label { color: var(--t3) !important; font-family: 'Geist Mono', monospace !important; font-size: 0.72rem !important; }

.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--b2) !important;
    border-radius: var(--r2) !important;
    color: var(--t1) !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.82rem !important;
}

[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--b3) !important;
    border-radius: var(--r3) !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(124,58,237,0.5) !important; background: var(--neon-ghost) !important; }

.stProgress > div > div { background: var(--b2) !important; border-radius: 99px !important; height: 2px !important; }
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--neon), var(--electric)) !important;
    border-radius: 99px !important;
    box-shadow: 0 0 8px rgba(0,212,255,0.4) !important;
}

[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--b2) !important;
    border-radius: var(--r3) !important;
    padding: 1.1rem 1.3rem !important;
    position: relative !important; overflow: hidden !important;
}
[data-testid="stMetric"]::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,212,255,0.4), transparent);
}
[data-testid="stMetricValue"] {
    font-family: 'Geist', sans-serif !important;
    font-size: 1.7rem !important; font-weight: 800 !important;
    color: var(--electric) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.62rem !important; color: var(--t3) !important;
    letter-spacing: 0.14em !important; text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] { font-family: 'Geist Mono', monospace !important; font-size: 0.72rem !important; }

.stSuccess { background: rgba(0,200,150,0.05) !important; border: 1px solid rgba(0,200,150,0.25) !important; border-radius: var(--r2) !important; }
.stError   { background: rgba(251,44,54,0.05) !important;  border: 1px solid rgba(251,44,54,0.25) !important;  border-radius: var(--r2) !important; }
.stWarning { background: rgba(251,191,36,0.05) !important; border: 1px solid rgba(251,191,36,0.25) !important; border-radius: var(--r2) !important; }
.stInfo    { background: rgba(0,212,255,0.04) !important;  border: 1px solid rgba(0,212,255,0.2) !important;   border-radius: var(--r2) !important; }

.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stToggle > label { color: var(--t3) !important; font-family: 'Geist Mono', monospace !important; font-size: 0.75rem !important; }

.streamlit-expanderHeader {
    background: var(--surface2) !important; border: 1px solid var(--b2) !important;
    border-radius: var(--r2) !important; font-family: 'Geist Mono', monospace !important;
    font-size: 0.75rem !important; color: var(--t3) !important; transition: all 0.2s !important;
}
.streamlit-expanderHeader:hover { border-color: var(--b3) !important; color: var(--t2) !important; }

.stTabs [data-baseweb="tab-list"] { gap: 0.25rem; background: var(--surface2) !important; border-radius: var(--r2) !important; padding: 4px !important; border: 1px solid var(--b2) !important; }
.stTabs [data-baseweb="tab"] {
    border-radius: var(--r1) !important; font-family: 'Geist Mono', monospace !important;
    font-size: 0.73rem !important; color: var(--t3) !important; transition: all 0.2s !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] { background: var(--surface4) !important; color: var(--electric) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

hr { border-color: var(--b2) !important; margin: 1.2rem 0 !important; }

/* ── Custom components ── */
.hero { text-align: center; padding: 5.5rem 1rem 3rem; position: relative; }
.hero-kicker {
    display: inline-flex; align-items: center; gap: 0.5rem;
    font-family: 'Geist Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.3em; text-transform: uppercase;
    color: var(--electric-dim); margin-bottom: 2rem;
    background: var(--electric-ghost); border: 1px solid rgba(0,212,255,0.15);
    border-radius: 99px; padding: 0.3rem 1rem;
}
.hero-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--electric); animation: pulse-dot 2s ease infinite; }
.hero-wordmark {
    font-family: 'Geist', sans-serif; font-weight: 900;
    font-size: clamp(5rem, 14vw, 10rem); line-height: 0.85; letter-spacing: -0.05em;
    background: linear-gradient(145deg, #ffffff 0%, var(--electric) 35%, var(--neon) 70%, var(--plasma) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    filter: drop-shadow(0 0 80px rgba(0,212,255,0.2));
    animation: hero-in 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
.hero-sub-title {
    font-family: 'Instrument Serif', serif; font-style: italic;
    font-size: clamp(1rem, 2.5vw, 1.5rem); color: var(--t2);
    margin-top: 0.8rem; letter-spacing: 0.02em;
    animation: hero-in 0.8s 0.1s cubic-bezier(0.16,1,0.3,1) both;
}
.hero-desc { font-size: 1rem; color: var(--t2); max-width: 480px; margin: 1.2rem auto 0; line-height: 1.7; animation: hero-in 0.8s 0.2s cubic-bezier(0.16,1,0.3,1) both; }
.hero-badges { display: flex; justify-content: center; gap: 0.6rem; flex-wrap: wrap; margin-top: 2rem; animation: hero-in 0.8s 0.3s cubic-bezier(0.16,1,0.3,1) both; }
.hero-badge { font-family: 'Geist Mono', monospace; font-size: 0.65rem; letter-spacing: 0.08em; color: var(--t3); background: var(--surface2); border: 1px solid var(--b2); border-radius: 99px; padding: 0.25rem 0.8rem; }

.glass { background: rgba(8,15,31,0.8); backdrop-filter: blur(20px); border: 1px solid var(--b2); border-radius: var(--r4); padding: 1.8rem; position: relative; overflow: hidden; animation: panel-in 0.4s ease both; }
.glass::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent); }
.glass-electric { border-color: rgba(0,212,255,0.18);  box-shadow: var(--glow-e); }
.glass-neon     { border-color: rgba(124,58,237,0.18); box-shadow: var(--glow-n); }
.glass-jade     { border-color: rgba(0,200,150,0.18);  box-shadow: var(--glow-c); }
.glass-fire     { border-color: rgba(251,44,54,0.18);  box-shadow: var(--glow-r); }
.glass-gold     { border-color: rgba(251,191,36,0.2);  box-shadow: 0 0 40px rgba(251,191,36,0.06); }

.sec { font-family: 'Geist Mono', monospace; font-size: 0.62rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--t4); display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.9rem; }
.sec::after { content: ''; flex: 1; height: 1px; background: var(--b2); }
.sec-electric { color: rgba(0,212,255,0.45); }
.sec-neon     { color: rgba(124,58,237,0.45); }
.sec-jade     { color: rgba(0,200,150,0.45); }

.persona-card { background: var(--surface2); border: 1px solid var(--b2); border-radius: var(--r3); padding: 1.2rem; text-align: center; cursor: pointer; transition: all 0.2s cubic-bezier(0.4,0,0.2,1); position: relative; overflow: hidden; }
.persona-card:hover { transform: translateY(-3px); border-color: var(--b4); }
.persona-card.active { border-color: rgba(0,212,255,0.4); background: rgba(0,212,255,0.04); }
.persona-card.active::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--electric), var(--neon)); }
.persona-emoji { font-size: 2rem; margin-bottom: 0.5rem; }
.persona-name { font-family: 'Geist', sans-serif; font-weight: 700; font-size: 0.95rem; color: var(--t1); }
.persona-role { font-family: 'Geist Mono', monospace; font-size: 0.6rem; color: var(--t3); letter-spacing: 0.06em; margin-top: 0.2rem; }

.mode-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.6rem; }
.mode-card { background: var(--surface2); border: 1px solid var(--b2); border-radius: var(--r2); padding: 0.9rem 1rem; cursor: pointer; transition: all 0.2s ease; }
.mode-card.m-casual   { border-color: rgba(0,200,150,0.35); background: rgba(0,200,150,0.04); }
.mode-card.m-standard { border-color: rgba(0,212,255,0.35); background: rgba(0,212,255,0.04); }
.mode-card.m-intense  { border-color: rgba(251,44,54,0.35);  background: rgba(251,44,54,0.04); }
.mode-title { font-family: 'Geist', sans-serif; font-weight: 700; font-size: 0.88rem; color: var(--t1); }
.mode-desc  { font-family: 'Geist Mono', monospace; font-size: 0.62rem; color: var(--t3); margin-top: 0.3rem; line-height: 1.5; }
.mode-pill  { display: inline-block; font-family: 'Geist Mono', monospace; font-size: 0.58rem; letter-spacing: 0.08em; border-radius: 99px; padding: 0.15rem 0.5rem; margin-top: 0.4rem; }
.pill-jade   { background: rgba(0,200,150,0.12); color: rgba(0,200,150,0.8); border: 1px solid rgba(0,200,150,0.25); }
.pill-elec   { background: rgba(0,212,255,0.1);  color: rgba(0,212,255,0.75); border: 1px solid rgba(0,212,255,0.2); }
.pill-fire   { background: rgba(251,44,54,0.1);  color: rgba(251,100,100,0.8); border: 1px solid rgba(251,44,54,0.25); }

.resume-profile { background: var(--surface3); border: 1px solid var(--b3); border-radius: var(--r3); padding: 1.4rem; position: relative; overflow: hidden; }
.resume-profile::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: linear-gradient(180deg, var(--neon), var(--electric)); }
.rp-name  { font-family: 'Geist', sans-serif; font-weight: 800; font-size: 1.3rem; color: var(--t1); }
.rp-role  { font-family: 'Geist Mono', monospace; font-size: 0.72rem; color: var(--electric-dim); margin-top: 0.2rem; letter-spacing: 0.06em; }
.rp-chips { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.8rem; }
.rp-chip  { font-family: 'Geist Mono', monospace; font-size: 0.62rem; letter-spacing: 0.04em; background: var(--surface4); border: 1px solid var(--b3); border-radius: var(--r1); padding: 0.2rem 0.55rem; color: var(--t2); }
.rp-chip.highlight { background: rgba(0,212,255,0.08); border-color: rgba(0,212,255,0.2); color: var(--electric-dim); }
.rp-stats { display: flex; gap: 1.5rem; margin-top: 1rem; padding-top: 0.8rem; border-top: 1px solid var(--b2); }
.rp-stat-num { font-family: 'Geist', sans-serif; font-weight: 700; font-size: 1.2rem; color: var(--t1); }
.rp-stat-lbl { font-family: 'Geist Mono', monospace; font-size: 0.58rem; color: var(--t4); letter-spacing: 0.1em; text-transform: uppercase; }

.avatar-bar { display: flex; align-items: flex-start; gap: 1.2rem; padding: 1.3rem 1.5rem; background: var(--surface2); border: 1px solid var(--b2); border-radius: var(--r3); position: relative; overflow: hidden; margin-bottom: 1.2rem; animation: slide-down 0.4s ease both; }
.avatar-bar::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: linear-gradient(180deg, var(--electric), var(--neon), rgba(240,171,252,0.3)); }
.avatar-icon { width: 52px; height: 52px; border-radius: 50%; flex-shrink: 0; background: var(--surface3); border: 1.5px solid rgba(0,212,255,0.3); display: flex; align-items: center; justify-content: center; font-size: 1.4rem; box-shadow: 0 0 16px rgba(0,212,255,0.12), inset 0 0 16px rgba(0,212,255,0.04); }
.avatar-icon.speaking { animation: speak-pulse 1.8s ease-in-out infinite; }
.avatar-meta { flex-shrink: 0; }
.avatar-name   { font-family: 'Geist', sans-serif; font-weight: 700; font-size: 0.9rem; color: var(--t1); }
.avatar-status { font-family: 'Geist Mono', monospace; font-size: 0.62rem; color: var(--jade); letter-spacing: 0.08em; display: flex; align-items: center; gap: 0.35rem; margin-top: 0.2rem; }
.status-led    { width: 5px; height: 5px; border-radius: 50%; background: var(--jade); animation: blink-led 2s ease infinite; }
.status-led.busy { background: var(--gold); }
.avatar-speech { font-family: 'Instrument Serif', serif; font-style: italic; font-size: 0.98rem; color: var(--t2); line-height: 1.65; flex: 1; }
.speech-open  { color: rgba(0,212,255,0.25); font-size: 1.2rem; vertical-align: -0.2em; }
.speech-close { color: rgba(0,212,255,0.25); font-size: 1.2rem; vertical-align: -0.2em; }

.q-card { background: linear-gradient(135deg, rgba(0,212,255,0.025) 0%, rgba(124,58,237,0.025) 100%); border: 1px solid var(--b3); border-radius: var(--r4); padding: 2.2rem 2.5rem; margin: 1rem 0; position: relative; overflow: hidden; animation: panel-in 0.35s ease both; }
.q-card::before { content: ''; position: absolute; top: -1px; left: 12%; right: 12%; height: 2px; background: linear-gradient(90deg, transparent, var(--electric), var(--neon), var(--plasma), transparent); opacity: 0.6; }
.q-counter { font-family: 'Geist Mono', monospace; font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--t4); margin-bottom: 0.9rem; }
.q-text     { font-family: 'Geist', sans-serif; font-size: clamp(1.05rem, 2vw, 1.35rem); font-weight: 600; line-height: 1.45; color: var(--t1); }
.q-meta     { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 1rem; }
.q-badge    { display: inline-flex; align-items: center; gap: 0.3rem; font-family: 'Geist Mono', monospace; font-size: 0.6rem; letter-spacing: 0.06em; border-radius: 99px; padding: 0.22rem 0.65rem; }
.qb-tech     { background: rgba(124,58,237,0.1); color: rgba(167,139,250,0.8); border: 1px solid rgba(124,58,237,0.25); }
.qb-behav    { background: rgba(0,212,255,0.08); color: rgba(0,212,255,0.7); border: 1px solid rgba(0,212,255,0.2); }
.qb-sit      { background: rgba(251,191,36,0.08); color: rgba(251,191,36,0.7); border: 1px solid rgba(251,191,36,0.25); }
.qb-rapport  { background: rgba(0,200,150,0.08); color: rgba(0,200,150,0.7); border: 1px solid rgba(0,200,150,0.25); }
.qb-ambition { background: rgba(251,44,54,0.08); color: rgba(251,100,100,0.7); border: 1px solid rgba(251,44,54,0.25); }
.q-comp-tag  { font-family: 'Geist Mono', monospace; font-size: 0.6rem; letter-spacing: 0.04em; background: var(--surface3); border: 1px solid var(--b3); border-radius: var(--r1); padding: 0.2rem 0.55rem; color: var(--t3); }
.q-diff { font-family: 'Geist Mono', monospace; font-size: 0.6rem; letter-spacing: 0.08em; padding: 0.2rem 0.55rem; border-radius: 99px; border: 1px solid; }
.diff-e { color: var(--jade);   border-color: rgba(0,200,150,0.3); background: rgba(0,200,150,0.06); }
.diff-m { color: var(--gold);   border-color: rgba(251,191,36,0.3); background: rgba(251,191,36,0.06); }
.diff-h { color: var(--crimson); border-color: rgba(251,44,54,0.3); background: rgba(251,44,54,0.06); }

.coach-bar { display: flex; align-items: flex-start; gap: 0.6rem; padding: 0.7rem 1rem; border-radius: var(--r2); border: 1px solid; font-family: 'Geist Mono', monospace; font-size: 0.72rem; line-height: 1.55; margin-top: 0.5rem; transition: all 0.3s ease; }
.coach-info    { color: rgba(0,212,255,0.55);  border-color: rgba(0,212,255,0.12); background: rgba(0,212,255,0.03); }
.coach-warn    { color: rgba(251,191,36,0.65); border-color: rgba(251,191,36,0.15); background: rgba(251,191,36,0.03); }
.coach-success { color: rgba(0,200,150,0.65);  border-color: rgba(0,200,150,0.15); background: rgba(0,200,150,0.03); }
.coach-icon    { font-size: 0.9rem; flex-shrink: 0; margin-top: 0.05rem; }

.star-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.4rem; }
.star-cell { background: var(--surface3); border: 1px solid var(--b2); border-radius: var(--r1); padding: 0.5rem; text-align: center; }
.star-label { font-family: 'Geist Mono', monospace; font-size: 0.58rem; color: var(--t4); letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.2rem; }
.star-val   { font-family: 'Geist', sans-serif; font-weight: 700; font-size: 1rem; }
.star-y { color: var(--jade); }
.star-n { color: var(--t4); }
.star-cell.active { border-color: rgba(0,200,150,0.35); background: rgba(0,200,150,0.04); }

.word-meter { display: flex; align-items: center; gap: 0.7rem; margin-top: 0.4rem; }
.wm-count  { font-family: 'Geist Mono', monospace; font-size: 0.68rem; color: var(--t3); min-width: 52px; }
.wm-track  { flex: 1; height: 2px; background: var(--b2); border-radius: 99px; overflow: hidden; }
.wm-fill   { height: 100%; border-radius: 99px; transition: width 0.3s ease, background 0.3s ease; }
.wm-status { font-family: 'Geist Mono', monospace; font-size: 0.62rem; min-width: 52px; text-align: right; }

.wave { display: flex; align-items: center; justify-content: center; gap: 2.5px; height: 36px; }
.wave-b { width: 2.5px; border-radius: 99px; background: var(--electric); animation: wave-dance var(--spd) ease-in-out infinite alternate; }

.fb-card { background: var(--surface2); border: 1px solid var(--b2); border-radius: var(--r3); padding: 1.8rem; margin-top: 1.2rem; position: relative; overflow: hidden; animation: slide-right 0.35s ease both; }
.fb-card::before { content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 2px; background: linear-gradient(180deg, var(--neon), var(--electric)); }
.fb-score-row { display: flex; align-items: center; gap: 1.3rem; margin-bottom: 1.3rem; flex-wrap: wrap; }
.fb-ring-wrap { position: relative; width: 80px; height: 80px; flex-shrink: 0; }
.fb-verdict { font-family: 'Geist', sans-serif; font-weight: 700; font-size: 1.15rem; color: var(--t1); }
.fb-sub     { font-family: 'Geist Mono', monospace; font-size: 0.68rem; color: var(--t3); margin-top: 0.15rem; }
.tone-chips { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.4rem; }
.tone-chip  { font-family: 'Geist Mono', monospace; font-size: 0.6rem; letter-spacing: 0.04em; padding: 0.18rem 0.55rem; border-radius: 99px; border: 1px solid; }
.tc-pos { background: rgba(0,200,150,0.08); color: rgba(0,200,150,0.7); border-color: rgba(0,200,150,0.2); }
.tc-neg { background: rgba(251,44,54,0.07); color: rgba(251,100,100,0.65); border-color: rgba(251,44,54,0.2); }
.tc-neu { background: rgba(124,58,237,0.08); color: rgba(167,139,250,0.65); border-color: rgba(124,58,237,0.2); }
.fb-sec { margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--b1); }
.fb-sec:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.fb-lbl { font-family: 'Geist Mono', monospace; font-size: 0.6rem; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.4rem; }
.fb-lbl-str { color: var(--jade); }
.fb-lbl-gap { color: var(--crimson); }
.fb-lbl-sug { color: var(--gold); }
.fb-lbl-ide { color: rgba(124,58,237,0.7); }
.fb-text { font-family: 'Geist', sans-serif; font-size: 0.88rem; color: var(--t2); line-height: 1.65; }

.ring-svg { transform: rotate(-90deg); }

.result-hero { text-align: center; padding: 4rem 2rem; background: linear-gradient(145deg, var(--surface) 0%, var(--surface2) 100%); border: 1px solid var(--b2); border-radius: var(--r5); margin-bottom: 2rem; position: relative; overflow: hidden; }
.result-hero::before { content: ''; position: absolute; inset: 0; background: radial-gradient(ellipse 60% 50% at 50% 0%, rgba(0,212,255,0.04), transparent); }
.result-grade { font-family: 'Geist', sans-serif; font-size: clamp(5rem, 14vw, 11rem); font-weight: 900; line-height: 0.88; letter-spacing: -0.04em; }
.grade-A { background: linear-gradient(135deg, #10b981, #00d4ff, #ffffff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; filter: drop-shadow(0 0 40px rgba(0,200,150,0.35)); }
.grade-B { background: linear-gradient(135deg, #00d4ff, #7c3aed); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; filter: drop-shadow(0 0 40px rgba(0,212,255,0.25)); }
.grade-C { background: linear-gradient(135deg, #fbbf24, #fb923c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.grade-D { background: linear-gradient(135deg, #fb2c36, #dc2626); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-score-row { font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: var(--t3); letter-spacing: 0.18em; text-transform: uppercase; margin-top: 1rem; }
.result-tagline   { font-family: 'Instrument Serif', serif; font-style: italic; font-size: 1.2rem; color: var(--t2); margin-top: 0.4rem; }

.qbt-item { display: flex; gap: 1rem; align-items: flex-start; padding: 1rem 0; border-bottom: 1px solid var(--b1); }
.qbt-item:last-child { border-bottom: none; }
.qbt-num  { font-family: 'Geist Mono', monospace; font-size: 0.62rem; color: var(--t4); min-width: 22px; padding-top: 0.12rem; }
.qbt-body { flex: 1; }
.qbt-q    { font-family: 'Geist', sans-serif; font-weight: 600; font-size: 0.88rem; color: var(--t1); line-height: 1.4; margin-bottom: 0.35rem; }
.qbt-tags { display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem; }
.score-pill { display: inline-flex; align-items: center; font-family: 'Geist Mono', monospace; font-size: 0.65rem; padding: 0.18rem 0.55rem; border-radius: 99px; border: 1px solid; }

.comp-row { display: flex; align-items: center; gap: 0.7rem; margin-bottom: 0.45rem; }
.comp-name  { font-family: 'Geist Mono', monospace; font-size: 0.66rem; color: var(--t2); min-width: 130px; }
.comp-track { flex: 1; height: 2px; background: var(--b2); border-radius: 99px; overflow: hidden; }
.comp-fill  { height: 100%; border-radius: 99px; }
.comp-score { font-family: 'Geist', sans-serif; font-weight: 700; font-size: 0.78rem; min-width: 28px; text-align: right; }

.followup-strip { display: inline-flex; align-items: center; gap: 0.4rem; font-family: 'Geist Mono', monospace; font-size: 0.62rem; padding: 0.25rem 0.75rem; border-radius: 99px; letter-spacing: 0.06em; background: rgba(251,44,54,0.08); border: 1px solid rgba(251,44,54,0.2); color: rgba(251,100,100,0.75); margin-bottom: 0.7rem; }
.rec-strip { display: flex; align-items: center; justify-content: space-between; padding: 0.6rem 1rem; background: rgba(251,44,54,0.04); border: 1px solid rgba(251,44,54,0.15); border-radius: var(--r2); margin-bottom: 0.7rem; }
.rec-left { display: flex; align-items: center; gap: 0.45rem; font-family: 'Geist Mono', monospace; font-size: 0.7rem; color: rgba(251,100,100,0.7); }
.rec-dot  { width: 6px; height: 6px; border-radius: 50%; background: var(--crimson); animation: blink-led 1s ease infinite; }

.tip { background: rgba(0,212,255,0.02); border: 1px solid rgba(0,212,255,0.1); border-radius: var(--r2); padding: 0.7rem 1rem; font-family: 'Geist Mono', monospace; font-size: 0.72rem; color: rgba(0,212,255,0.45); line-height: 1.6; }
.tip.neon { color: rgba(167,139,250,0.55); border-color: rgba(124,58,237,0.15); background: rgba(124,58,237,0.02); }

.skills-match { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.6rem; }
.skill-tag { font-family: 'Geist Mono', monospace; font-size: 0.62rem; padding: 0.22rem 0.65rem; border-radius: var(--r1); border: 1px solid; }
.sk-match   { background: rgba(0,200,150,0.08); color: rgba(0,200,150,0.7); border-color: rgba(0,200,150,0.2); }
.sk-gap     { background: rgba(251,44,54,0.06); color: rgba(251,100,100,0.6); border-color: rgba(251,44,54,0.18); }
.sk-neutral { background: var(--surface3); color: var(--t3); border-color: var(--b2); }

.export-block { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.3rem; background: rgba(251,191,36,0.03); border: 1px solid rgba(251,191,36,0.12); border-radius: var(--r3); }
.export-icon  { font-size: 1.3rem; }
.export-title { font-family: 'Geist', sans-serif; font-weight: 600; font-size: 0.88rem; color: var(--t1); }
.export-desc  { font-family: 'Geist Mono', monospace; font-size: 0.62rem; color: var(--t3); margin-top: 0.15rem; }

.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.5rem; }
.stat-pill { display: flex; flex-direction: column; align-items: center; background: var(--surface3); border: 1px solid var(--b2); border-radius: var(--r2); padding: 0.5rem 0.8rem; min-width: 56px; }
.stat-num { font-family: 'Geist', sans-serif; font-weight: 800; font-size: 1.1rem; color: var(--electric); }
.stat-lbl { font-family: 'Geist Mono', monospace; font-size: 0.58rem; color: var(--t4); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 0.15rem; }

.analysis-box { background: var(--surface3); border: 1px solid var(--b2); border-radius: var(--r2); padding: 0.9rem 1.1rem; margin-top: 0.8rem; }
.ab-title { font-family: 'Geist Mono', monospace; font-size: 0.63rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--t3); margin-bottom: 0.5rem; }

@keyframes hero-in { from { opacity:0; transform:translateY(20px); filter:blur(8px); } to { opacity:1; transform:translateY(0); filter:blur(0); } }
@keyframes panel-in { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:translateY(0); } }
@keyframes slide-down  { from { opacity:0; transform:translateY(-8px); } to { opacity:1; transform:translateY(0); } }
@keyframes slide-right { from { opacity:0; transform:translateX(-8px); } to { opacity:1; transform:translateX(0); } }
@keyframes pulse-dot { 0%,100%{box-shadow:0 0 0 0 rgba(0,212,255,0.4)} 50%{box-shadow:0 0 0 4px rgba(0,212,255,0)} }
@keyframes blink-led { 0%,100%{opacity:1} 50%{opacity:0.25} }
@keyframes speak-pulse { 0%,100%{box-shadow:0 0 16px rgba(0,212,255,0.12)} 50%{box-shadow:0 0 28px rgba(0,212,255,0.3)} }
@keyframes wave-dance { from{height:3px} to{height:var(--maxh)} }
"""

st.markdown(f"<style>{DESIGN}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# UTILITY HTML BUILDERS
# ─────────────────────────────────────────────────────────────
def waveform_html(n=28, accent="var(--electric)"):
    bars = "".join([
        f'<div class="wave-b" style="--spd:{random.uniform(0.3,0.8):.2f}s;--maxh:{random.randint(12,38)}px;height:{random.randint(3,12)}px;opacity:{random.uniform(0.4,0.85):.2f};background:{accent if i%3!=2 else "var(--neon)"};"></div>'
        for i in range(n)
    ])
    return f'<div class="wave">{bars}</div>'

def ring_svg(score: float, size=80, stroke=5):
    r = (size / 2) - stroke
    circ = 2 * math.pi * r
    pct = min(score / 10.0, 1.0)
    dash = pct * circ
    color = "#00c896" if score >= 7 else "#fbbf24" if score >= 5 else "#fb2c36"
    glow = color + "55"
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" class="ring-svg">
        <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#0f1e35" stroke-width="{stroke}"/>
        <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
            stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"
            style="filter:drop-shadow(0 0 5px {glow})"/>
    </svg>"""

# ─────────────────────────────────────────────────────────────
# INTERVIEWER PERSONAS
# ─────────────────────────────────────────────────────────────
PERSONAS = {
    "Ketu": {
        "name": "Ketu", "title": "Senior AI Interviewer", "avatar": "🤖",
        "style": "balanced, thoughtful, and genuinely curious",
        "color": "var(--electric)",
        "greetings": [
            "Great to meet you! I've carefully reviewed your profile. I'm genuinely excited to explore your journey.",
            "Welcome! I've gone through your background and the role requirements thoroughly. Let's have an honest conversation.",
            "Hello! I've prepared some tailored questions for you today. Take your time — I'm here to understand you, not catch you out.",
        ],
        "transitions": [
            "Interesting — thank you for sharing that. Let me move to the next area.",
            "I appreciate your openness. Let's continue.",
            "That's helpful context. Here's my next question.",
            "Good. Moving on…",
        ],
        "thinking": ["Reviewing your answer…", "Analysing your response…", "Processing what you said…"],
    },
    "Aria": {
        "name": "Aria", "title": "Technical Director", "avatar": "🧬",
        "style": "technical, precise, analytically rigorous, and direct",
        "color": "var(--neon)",
        "greetings": [
            "I'll be direct: I want to understand how you think, not just what you've memorised. Let's begin.",
            "I've reviewed your technical background in detail. I'll be asking you to go deep — that's how we find out what you're genuinely capable of.",
        ],
        "transitions": [
            "Noted. Let's test another dimension.",
            "That's an interesting angle. Moving on.",
            "Understood. Next question.",
        ],
        "thinking": ["Running technical analysis…", "Evaluating depth of answer…", "Checking technical precision…"],
    },
    "Marcus": {
        "name": "Marcus", "title": "Culture & Values Lead", "avatar": "🌿",
        "style": "empathetic, culture-focused, human-centred, and values-driven",
        "color": "var(--jade)",
        "greetings": [
            "I want this to feel like a real conversation — not an interrogation. I'm genuinely curious about who you are.",
            "Welcome! I focus on values, teamwork, and the human side of work. Let's explore that together today.",
        ],
        "transitions": [
            "I really appreciate your honesty there. Let's explore another dimension.",
            "That's genuinely telling — thank you. Let me ask about something different.",
            "Good. Let's keep going.",
        ],
        "thinking": ["Reflecting on your answer…", "Considering cultural fit…", "Reading between the lines…"],
    },
    "Nova": {
        "name": "Nova", "title": "Product Strategy Lead", "avatar": "🚀",
        "style": "product-minded, strategic, data-driven, and challenge-seeking",
        "color": "var(--plasma)",
        "greetings": [
            "I'm going to push you to think like a product thinker — not just a practitioner. Let's explore your strategic instincts.",
            "Hi! I care about how you frame problems and measure success. Ready to think out loud with me?",
        ],
        "transitions": [
            "That's a good instinct. Let me challenge you further.",
            "Noted. Let's dig into product thinking next.",
            "Interesting framing. Moving on.",
        ],
        "thinking": ["Evaluating strategic thinking…", "Assessing product instincts…", "Analysing your reasoning…"],
    },
}

QUESTION_TYPES = {
    "rapport":     ("💬", "qb-rapport",  "Rapport"),
    "technical":   ("⚙️", "qb-tech",    "Technical"),
    "behavioral":  ("🧠", "qb-behav",   "Behavioral"),
    "situational": ("🎯", "qb-sit",     "Situational"),
    "ambition":    ("🚀", "qb-ambition","Forward-looking"),
}

INTERVIEW_MODES = {
    "Casual":   {"pressure": "low",  "followup_threshold": 4.0, "max_followups": 1, "emoji": "🌿", "desc": "Relaxed pace, supportive tone", "pill_cls": "pill-jade"},
    "Standard": {"pressure": "med",  "followup_threshold": 5.5, "max_followups": 2, "emoji": "⚡", "desc": "Professional, balanced assessment", "pill_cls": "pill-elec"},
    "Intense":  {"pressure": "high", "followup_threshold": 7.0, "max_followups": 3, "emoji": "🔥", "desc": "High-pressure, rigorous evaluation", "pill_cls": "pill-fire"},
}

COMPETENCY_FRAMEWORKS = {
    "Engineering":    ["Problem Solving", "Technical Depth", "System Design", "Code Quality", "Collaboration", "Communication", "Growth Mindset"],
    "Management":     ["Leadership", "Strategic Thinking", "Communication", "Conflict Resolution", "Decision Making", "Mentoring", "Execution"],
    "Product":        ["Product Sense", "Data Analysis", "User Empathy", "Prioritization", "Communication", "Execution", "Strategy"],
    "Design":         ["Visual Thinking", "User Research", "Communication", "Craft", "Iteration", "Business Acumen", "Storytelling"],
    "Sales/BD":       ["Persuasion", "Relationship Building", "Resilience", "Product Knowledge", "Communication", "Pipeline Management", "Closing"],
    "Data/Analytics": ["Statistical Thinking", "SQL/Tooling", "Communication", "Problem Framing", "Visualization", "Business Acumen", "Experimentation"],
    "Marketing":      ["Brand Thinking", "Data Analysis", "Creativity", "Communication", "Campaign Strategy", "Growth Mindset", "Audience Insight"],
    "Operations":     ["Process Design", "Problem Solving", "Stakeholder Management", "Execution", "Data-Driven Decision Making", "Resilience", "Communication"],
}

POSITIVE_TONE = {"Confident", "Structured", "Concise", "Detailed", "Passionate", "Analytical", "Creative", "Experienced", "Thoughtful", "Authentic", "Polished"}
NEGATIVE_TONE = {"Vague", "Nervous", "Hesitant", "Rambling", "Unprepared"}
FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "literally", "actually", "sort of", "kind of", "i mean", "right", "so yeah", "honestly", "obviously", "clearly"}

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "screen": "setup",
        "questions": [], "q_types": [], "q_competencies": [], "q_difficulties": [],
        "current": 0, "scores": [], "feedback_list": [],
        "resume_text": "", "jd_text": "", "resume_profile": None,
        "candidate_name": "", "role_title": "",
        "num_questions": 8, "session_start": None, "q_start": None,
        "tts_enabled": True, "submitted": False,
        "current_feedback": None, "ketu_message": "",
        "is_followup": False, "followup_count": 0,
        "transcript": [],
        "persona": "Ketu", "interview_mode": "Standard", "category_tag": "Engineering",
        "competency_scores": {}, "filler_counts": [], "word_counts": [],
        "ai_summary": None, "session_history": [], "camera_enabled": False,
        "show_hints": True,
        # Camera analytics (persisted across reloads via session state)
        "cam_eye_contact_avg": 0.0,
        "cam_confidence_avg": 0.0,
        "cam_expression_log": [],
        "cam_posture_log": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_llm():
    try:
        return ChatGroq(temperature=0.4, model_name="llama-3.3-70b-versatile", api_key=st.secrets["GROQ_API_KEY"])
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def get_embeddings():
    class Local:
        def __init__(self): self.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        def embed_documents(self, texts): return self.m.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()
        def embed_query(self, text): return self.m.encode([text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0].tolist()
    return Local()

# ─────────────────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────────────────
def tts_play(text: str):
    if not st.session_state.get("tts_enabled", True) or not text:
        return
    try:
        buf = BytesIO()
        gTTS(text=text[:50000], lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        st.markdown(f'<audio autoplay style="display:none"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION
# ─────────────────────────────────────────────────────────────
def transcribe(audio_bytes: bytes) -> str:
    try:
        from groq import Groq
        gc = Groq(api_key=st.secrets["GROQ_API_KEY"])
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.wav"
        return gc.audio.transcriptions.create(model="whisper-large-v3-turbo", file=buf).text.strip()
    except Exception as e:
        st.warning(f"⚠️ Transcription failed: {e}")
        return ""

# ─────────────────────────────────────────────────────────────
# DOCUMENT LOADER
# ─────────────────────────────────────────────────────────────
def load_doc(f) -> str:
    ext = f.name.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(f.getvalue()); path = tmp.name
    try:
        if ext == "pdf":   return "\n".join(d.page_content for d in PyPDFLoader(path).load())
        if ext in ("docx","doc"): docs = Docx2txtLoader(path).load(); return docs[0].page_content if docs else ""
        return f.getvalue().decode("utf-8", errors="ignore")
    finally:
        if os.path.exists(path): os.remove(path)

# ─────────────────────────────────────────────────────────────
# RESUME ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyze_resume(resume_text: str, jd_text: str, role: str, llm) -> dict:
    prompt = f"""Analyse this resume against the job description for: {role}

RESUME:
{resume_text[:3000]}

JOB DESCRIPTION:
{jd_text[:2000]}

Return ONLY this JSON (no markdown, no extra text):
{{
  "candidate_name": "<extracted name or 'Candidate'>",
  "current_role": "<most recent job title>",
  "years_experience": "<estimated years, e.g. '5 years'>",
  "top_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
  "matching_skills": ["skill that matches JD", ...],
  "gap_skills": ["skill in JD but not resume", ...],
  "education": "<highest degree + field>",
  "companies": ["company1", "company2", "company3"],
  "strengths": ["strength1", "strength2", "strength3"],
  "red_flags": ["potential concern1"] or [],
  "overall_fit_score": <0-10 float>,
  "fit_rationale": "<1-2 sentences on candidate-role fit>"
}}
"""
    try:
        raw = llm.invoke(prompt).content.strip()
        if raw.startswith("```"): raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()
        return json.loads(raw)
    except Exception:
        return {
            "candidate_name": "Candidate", "current_role": "Professional",
            "years_experience": "N/A", "top_skills": [], "matching_skills": [],
            "gap_skills": [], "education": "N/A", "companies": [],
            "strengths": [], "red_flags": [], "overall_fit_score": 5.0,
            "fit_rationale": "Resume analysis unavailable."
        }

# ─────────────────────────────────────────────────────────────
# QUESTION GENERATION
# ─────────────────────────────────────────────────────────────
def gen_questions(jd, resume, role, n, llm, persona_name, mode, category, resume_profile=None) -> tuple:
    persona = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    comps = COMPETENCY_FRAMEWORKS.get(category, COMPETENCY_FRAMEWORKS["Engineering"])
    comp_str = ", ".join(comps)
    mode_cfg = INTERVIEW_MODES.get(mode, INTERVIEW_MODES["Standard"])

    profile_context = ""
    if resume_profile:
        skills_str = ", ".join(resume_profile.get("top_skills", [])[:5])
        gap_str = ", ".join(resume_profile.get("gap_skills", [])[:3])
        profile_context = f"\nCANDIDATE PROFILE INSIGHTS: Skills: {skills_str} | Gaps to probe: {gap_str}"

    prompt = f"""You are {persona['name']}, a {persona['style']} AI interviewer.
Interview type: {mode} ({mode_cfg['pressure']}-pressure) | Role: {role}
Competency framework: {comp_str}{profile_context}

JOB DESCRIPTION:
{jd[:2500]}

CANDIDATE RESUME:
{resume[:2500]}

Generate exactly {n} tailored questions. Structure:
- Q1-2: rapport (warm, personal, background-focused)
- Q3-{max(4,n-3)}: technical/behavioral (deep, mapped to specific competencies, reference resume details)
- Q{max(4,n-3)+1}-{n-1}: situational (STAR-worthy scenarios, real challenges)
- Q{n}: ambition/growth (forward-looking)

Mode {mode} rules:
{"Supportive, accessible, broad strokes." if mode=="Casual" else ""}
{"Professional depth, balanced probing." if mode=="Standard" else ""}
{"Multi-part, edge cases, trade-offs, failures, pushback." if mode=="Intense" else ""}

Return ONLY valid JSON:
{{
  "questions": ["q1",...],
  "types": ["rapport","technical",...],
  "competencies": ["Communication",...],
  "difficulties": ["easy","medium","hard",...]
}}
Types: rapport, technical, behavioral, situational, ambition
"""
    resp = llm.invoke(prompt)
    try:
        raw = resp.content.strip()
        if raw.startswith("```"): raw = re.sub(r"```(?:json)?","",raw).strip().rstrip("```").strip()
        d = json.loads(raw)
        qs   = d.get("questions",[])[:n]
        ts   = d.get("types",["technical"]*n)[:n]
        cs   = d.get("competencies",["Technical Depth"]*n)[:n]
        dfs  = d.get("difficulties",["medium"]*n)[:n]
        while len(ts) < len(qs):  ts.append("technical")
        while len(cs) < len(qs):  cs.append("Technical Depth")
        while len(dfs) < len(qs): dfs.append("medium")
        return qs, ts, cs, dfs
    except Exception:
        qs, ts, cs, dfs = [], [], [], []
        for line in resp.content.splitlines():
            line = line.strip()
            if re.match(r'^\d+[.)\-]', line):
                cleaned = re.sub(r'^\d+[.)\-]\s*','',line).strip()
                if cleaned:
                    qs.append(cleaned); ts.append("technical"); cs.append("Technical Depth"); dfs.append("medium")
        return qs[:n], ts[:n], cs[:n], dfs[:n]

# ─────────────────────────────────────────────────────────────
# ANSWER QUALITY ANALYSIS
# ─────────────────────────────────────────────────────────────
def analyze_quality(answer: str) -> dict:
    words = answer.lower().split()
    wc = len(words)
    text = answer.lower()
    sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if s.strip()]
    sc = max(len(sentences), 1)

    filler_count = sum(text.count(f) for f in FILLER_WORDS)
    filler_pct   = filler_count / max(wc, 1)

    star = {
        "Situation": bool(re.search(r'\b(when|once|at my|in my|during|while working|we were|i was|back at|last year|previously)\b', text)),
        "Task":      bool(re.search(r'\b(had to|needed to|responsible for|my role|tasked with|goal was|objective|challenge was)\b', text)),
        "Action":    bool(re.search(r'\b(i did|i built|i wrote|i led|i implemented|i created|i designed|i worked|i developed|i resolved|i decided|i introduced|i proposed|i initiated)\b', text)),
        "Result":    bool(re.search(r'\b(result|outcome|achieved|improved|reduced|increased|saved|delivered|shipped|launched|percent|%|metric|measur|impact|generated|won|closed)\b', text)),
    }
    star_score = sum(star.values())

    has_numbers     = bool(re.search(r'\b\d+\b', answer))
    has_percentages = bool(re.search(r'\d+\s*%', answer))
    has_timeframes  = bool(re.search(r'\b(week|month|quarter|year|day|sprint|cycle)\b', text))
    specificity = sum([has_numbers, has_percentages, has_timeframes])

    if wc < 30:     verbosity = "too_short"
    elif wc < 80:   verbosity = "short"
    elif wc <= 250: verbosity = "ideal"
    elif wc <= 400: verbosity = "long"
    else:           verbosity = "too_long"

    if verbosity == "too_short":    hint = ("warn",    "💡", "Very brief — try expanding with context and a real example.")
    elif verbosity == "short":       hint = ("warn",    "✍️", "Consider adding more detail — what was the measurable outcome?")
    elif filler_pct > 0.07:          hint = ("warn",    "🎙️", f"High filler word density detected ({filler_count}×). Speak more deliberately.")
    elif star_score == 4 and 80<=wc<=250: hint = ("success","✨", "Excellent! Full STAR coverage with ideal length — strong response.")
    elif verbosity == "too_long":    hint = ("warn",    "✂️", "Consider tightening up — focus on the most impactful details.")
    elif star_score >= 3:            hint = ("success", "🟢", "Good structure — you're covering the key STAR components.")
    elif specificity >= 2:           hint = ("success", "📊", "Good use of specifics and data — that strengthens your answer.")
    else:                            hint = ("info",    "🎯", "Ground your answer in a specific real example with a measurable result.")

    return {
        "wc": wc, "filler_count": filler_count, "filler_pct": filler_pct,
        "star": star, "star_score": star_score, "verbosity": verbosity,
        "hint": hint, "specificity": specificity,
        "avg_sentence_len": wc / sc,
    }

# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def evaluate(q, answer, role, q_type, competency, mode, persona_name, llm, context=None) -> dict:
    persona = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    mode_cfg = INTERVIEW_MODES.get(mode, INTERVIEW_MODES["Standard"])
    ctx = ""
    if context:
        ctx = "\n".join([f"{m['role'].upper()}: {m['content'][:200]}" for m in context[-4:]])

    prompt = f"""You are {persona['name']}, a {persona['style']} AI interviewer evaluating a {role} candidate.
Mode: {mode} ({mode_cfg['pressure']}-pressure) | Q-type: {q_type} | Competency: {competency}
{"CONTEXT:\n"+ctx if ctx else ""}

QUESTION: {q}
ANSWER: {answer}

Be {"lenient and encouraging" if mode=="Casual" else "rigorous and exacting" if mode=="Intense" else "balanced and fair"}.

Return ONLY valid JSON:
{{
  "score": <0-10 float>,
  "competency_score": <0-10 float for '{competency}'>,
  "verdict": "<Exceptional|Strong|Solid|Average|Weak>",
  "strength": "<specific 1-sentence strength>",
  "weakness": "<specific 1-sentence gap>",
  "suggestion": "<concrete, actionable 1-sentence tip>",
  "star_feedback": "<1 sentence STAR feedback if behavioral/situational, else ''>",
  "tone_signals": ["<3 signals from: Confident, Structured, Vague, Concise, Detailed, Nervous, Passionate, Hesitant, Analytical, Creative, Experienced, Rambling, Thoughtful, Unprepared, Authentic, Polished>"],
  "needs_followup": <true|false>,
  "followup_question": "<natural follow-up if score below {mode_cfg['followup_threshold']+0.5}, else ''>",
  "interviewer_reaction": "<1 short conversational reaction from {persona['name']} in first person>",
  "ideal_hint": "<1-2 sentences: what a strong answer would have included>"
}}
"""
    try:
        raw = llm.invoke(prompt).content.strip()
        if raw.startswith("```"): raw = re.sub(r"```(?:json)?","",raw).strip().rstrip("```").strip()
        r = json.loads(raw)
        r["score"]            = min(10.0, max(0.0, float(r.get("score", 5))))
        r["competency_score"] = min(10.0, max(0.0, float(r.get("competency_score", r["score"]))))
        return r
    except Exception:
        return {
            "score":5.0,"competency_score":5.0,"verdict":"Average",
            "strength":"Answer provided.","weakness":"Could not fully evaluate.",
            "suggestion":"Use the STAR method with a specific, measurable example.",
            "star_feedback":"","tone_signals":["Thoughtful"],"needs_followup":False,
            "followup_question":"","interviewer_reaction":"Thanks for sharing that.",
            "ideal_hint":"Include a specific example with a measurable outcome.",
        }

# ─────────────────────────────────────────────────────────────
# AI SUMMARY
# ─────────────────────────────────────────────────────────────
def gen_summary(feedback_list, role, name, avg_score, persona_name, mode, llm) -> str:
    persona = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    qa_pairs = "\n\n".join([
        f"Q{i+1} [{item.get('type','?')} · {item.get('competency','?')}]: {item['q']}\n"
        f"Answer: {item['a'][:280]}…\nScore: {item['eval']['score']}/10 — {item['eval']['verdict']}"
        for i, item in enumerate(feedback_list)
    ])
    prompt = f"""You are {persona['name']}, a {persona['style']} AI interviewer writing a post-interview report.
Candidate: {name or 'the candidate'} | Role: {role} | Mode: {mode} | Overall: {avg_score:.1f}/10

INTERVIEW DATA:
{qa_pairs}

Write a structured assessment with exactly these 4 sections (use these exact headers):

**OVERALL IMPRESSION**
2-3 sentences on general performance, calibre, and role fit.

**KEY STRENGTHS**
2-3 sentences on standout demonstrations. Reference specific answers.

**DEVELOPMENT AREAS**
2-3 sentences on gaps with concrete references.

**HIRING RECOMMENDATION**
1-2 sentences with a clear call: Strong Hire / Hire / Hold / No Hire. Justify briefly.

Prose only. No bullets. Professional but human. Write as {persona['name']}, first person.
"""
    return llm.invoke(prompt).content.strip()

# ─────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────
def build_json(state) -> str:
    sc_list = state.get("scores", [])
    avg = sum(s.get("score",0) for s in sc_list) / max(len(sc_list), 1)
    return json.dumps({
        "meta": {
            "candidate": state.get("candidate_name","Anonymous"),
            "role": state.get("role_title",""),
            "mode": state.get("interview_mode","Standard"),
            "persona": state.get("persona","Ketu"),
            "date": datetime.now().isoformat(),
            "version": "2.1",
        },
        "summary": {"avg_score": round(avg,2), "grade": grade_letter(avg), "total_questions": len(sc_list)},
        "resume_profile": state.get("resume_profile"),
        "camera_analytics": {
            "eye_contact_avg": round(state.get("cam_eye_contact_avg",0),1),
            "confidence_avg": round(state.get("cam_confidence_avg",0),1),
        },
        "qa_transcript": [
            {
                "num": i+1, "question": item["q"], "type": item.get("type",""), "competency": item.get("competency",""),
                "difficulty": item.get("difficulty",""), "answer": item["a"],
                "score": item["eval"].get("score",0), "verdict": item["eval"].get("verdict",""),
                "strength": item["eval"].get("strength",""), "weakness": item["eval"].get("weakness",""),
                "suggestion": item["eval"].get("suggestion",""), "tone": item["eval"].get("tone_signals",[]),
                "time_sec": item.get("time",0), "word_count": item.get("qa",{}).get("wc",0),
                "filler_words": item.get("qa",{}).get("filler_count",0),
                "star_score": item.get("qa",{}).get("star_score",0),
            }
            for i, item in enumerate(state.get("feedback_list",[]))
        ],
        "competency_scores": {k: round(sum(v)/len(v),2) for k,v in state.get("competency_scores",{}).items() if v},
        "communication_stats": {
            "total_words": sum(state.get("word_counts",[])),
            "avg_words_per_answer": sum(state.get("word_counts",[]))//max(len(state.get("word_counts",[])),1),
            "total_filler_words": sum(state.get("filler_counts",[])),
        },
        "ai_assessment": state.get("ai_summary",""),
    }, indent=2)

def build_csv(state) -> str:
    rows = []
    for i, item in enumerate(state.get("feedback_list",[])):
        rows.append({
            "Q#": i+1, "Question": item["q"], "Type": item.get("type",""), "Competency": item.get("competency",""),
            "Difficulty": item.get("difficulty",""), "Answer": item["a"][:200]+"…",
            "Score": item["eval"].get("score",0), "Verdict": item["eval"].get("verdict",""),
            "Strength": item["eval"].get("strength",""), "Gap": item["eval"].get("weakness",""),
            "Suggestion": item["eval"].get("suggestion",""), "Words": item.get("qa",{}).get("wc",0),
            "Fillers": item.get("qa",{}).get("filler_count",0), "STAR Score": item.get("qa",{}).get("star_score",0),
        })
    return pd.DataFrame(rows).to_csv(index=False)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def score_color(s): return "#00c896" if s>=7 else "#fbbf24" if s>=5 else "#fb2c36"
def grade_letter(avg):
    if avg>=8.5: return "A+"
    if avg>=7.5: return "A"
    if avg>=6.5: return "B+"
    if avg>=5.5: return "B"
    if avg>=4.5: return "C"
    return "D"
def grade_css(g):
    if g.startswith("A"): return "grade-A"
    if g.startswith("B"): return "grade-B"
    if g.startswith("C"): return "grade-C"
    return "grade-D"
def grade_tagline(g): return {
    "A+":"Outstanding — a rare calibre of candidate.",
    "A":"Excellent performance — strong hire signal.",
    "B+":"Very good — above expectations in most areas.",
    "B":"Solid candidate with clear strengths.",
    "C":"Adequate but notable gaps remain.",
    "D":"Significant development needed.",
}.get(g,"Interview complete.")

PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Geist Mono, monospace", color="#3d5580"),
    xaxis=dict(gridcolor="#0e1a2e", zerolinecolor="#0e1a2e"),
    yaxis=dict(gridcolor="#0e1a2e", zerolinecolor="#0e1a2e"),
    margin=dict(t=30, b=30, l=12, r=12),
)

# ─────────────────────────────────────────────────────────────
# ADVANCED CAMERA PANEL v2.1
# Features:
#   • TensorFlow.js BlazeFace real-time face detection
#   • Landmark-based eye contact scoring (iris vs centre)
#   • Expression classification (smile / neutral / tense)
#   • Head pose / posture estimation (tilt angle)
#   • Confidence composite score (0-100)
#   • Live animated HUD overlays (face mesh, zones)
#   • Coaching tip carousel tied to metric thresholds
#   • Session statistics (avg eye contact, peak expression, posture drift)
#   • Snapshot capture to canvas (base64 PNG shown in sidebar)
#   • Mini sparkline for rolling confidence score
#   • Minimal, non-distracting design
# ─────────────────────────────────────────────────────────────
def camera_panel():
    import streamlit.components.v1 as components
    components.html("""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent; font-family: 'Geist Mono', 'Courier New', monospace; }

  #root {
    background: #060c18;
    border: 1px solid #14253e;
    border-radius: 14px;
    overflow: hidden;
    user-select: none;
  }

  /* ── Top bar ── */
  #topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 7px 11px;
    background: rgba(0,0,0,0.55);
    border-bottom: 1px solid #0e1a2e;
  }
  .dot-red   { width:7px;height:7px;border-radius:50%;background:#fb2c36;animation:blink 1.1s ease infinite; display:inline-block;margin-right:5px; }
  #live-lbl  { font-size:9.5px; color:#fb2c36; letter-spacing:.15em; }
  #mode-pill {
    font-size:9px; letter-spacing:.07em; padding:2px 8px; border-radius:99px;
    background:rgba(0,212,255,0.08); color:rgba(0,212,255,0.65); border:1px solid rgba(0,212,255,0.18);
  }
  #fps-lbl { font-size:9px; color:rgba(0,212,255,0.3); }

  /* ── Video ── */
  #vidwrap { position: relative; background: #02040a; }
  video { width: 100%; display: block; transform: scaleX(-1); }
  canvas#overlay { position: absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }

  /* ── HUD overlay elements ── */
  #hud {
    position: absolute; inset: 0; pointer-events: none;
    display: flex; flex-direction: column; justify-content: space-between; padding: 8px;
  }

  /* Confidence arc (top right) */
  #conf-ring {
    position: absolute; top: 8px; right: 8px;
    width: 54px; height: 54px;
  }
  #conf-ring svg { width:100%; height:100%; }
  #conf-center {
    position: absolute; inset:0;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
  }
  #conf-num { font-size:13px; font-weight:700; color:#f0f4ff; line-height:1; }
  #conf-lbl { font-size:7px; color:rgba(0,212,255,0.4); letter-spacing:.1em; text-transform:uppercase; margin-top:1px; }

  /* Eye contact badge (top left) */
  #eye-badge {
    position: absolute; top: 8px; left: 8px;
    background: rgba(6,12,24,0.75); border: 1px solid rgba(0,212,255,0.2);
    border-radius: 6px; padding: 4px 7px;
    display: flex; flex-direction: column; gap: 2px;
  }
  #eye-score { font-size:15px; font-weight:700; color:#00d4ff; line-height:1; }
  #eye-lbl   { font-size:7.5px; color:rgba(0,212,255,0.35); letter-spacing:.12em; text-transform:uppercase; }

  /* Expression badge (bottom left) */
  #expr-badge {
    position: absolute; bottom: 44px; left: 8px;
    background: rgba(6,12,24,0.75); border: 1px solid rgba(124,58,237,0.25);
    border-radius: 6px; padding: 4px 8px;
    display: flex; align-items: center; gap: 5px;
  }
  #expr-icon { font-size:14px; }
  #expr-text { font-size:9px; color:rgba(167,139,250,0.75); letter-spacing:.06em; }

  /* Posture badge (bottom right) */
  #posture-badge {
    position: absolute; bottom: 44px; right: 8px;
    background: rgba(6,12,24,0.75); border: 1px solid rgba(0,200,150,0.2);
    border-radius: 6px; padding: 4px 8px;
    display: flex; align-items: center; gap: 5px;
  }
  #posture-icon { font-size:12px; }
  #posture-text { font-size:9px; color:rgba(0,200,150,0.65); letter-spacing:.05em; }

  /* Scan line */
  #scan { position:absolute; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(0,212,255,0.18),transparent); animation:scan 4s linear infinite; pointer-events:none; }
  @keyframes scan { 0%{top:0} 100%{top:100%} }

  /* ── Sparkline ── */
  #spark-wrap {
    background: rgba(0,0,0,0.5); border-top: 1px solid #0e1a2e;
    padding: 5px 10px 4px; display:flex; align-items:center; gap:8px;
  }
  #spark-lbl { font-size:8.5px; color:rgba(0,212,255,0.28); letter-spacing:.14em; min-width:60px; }
  canvas#sparkline { flex:1; height:22px; }

  /* ── Coaching strip ── */
  #coach-strip {
    background: rgba(0,212,255,0.03); border-top: 1px solid rgba(0,212,255,0.08);
    padding: 6px 10px;
    display: flex; align-items: center; gap: 7px;
  }
  #coach-icon { font-size:12px; flex-shrink:0; }
  #coach-text { font-size:9.5px; color:rgba(0,212,255,0.42); line-height:1.45; letter-spacing:.02em; }

  /* ── Stats row ── */
  #stats-row {
    display:grid; grid-template-columns:repeat(4,1fr);
    background:#02040a; border-top:1px solid #0e1a2e;
  }
  .stat-cell {
    padding:5px 0; text-align:center; border-right:1px solid #0e1a2e;
  }
  .stat-cell:last-child { border-right:none; }
  .snum { font-size:12px; font-weight:700; color:#00d4ff; }
  .slbl { font-size:7.5px; color:#1e3258; letter-spacing:.1em; text-transform:uppercase; margin-top:1px; }

  /* ── Snapshot button ── */
  #snap-btn {
    display:block; width:calc(100% - 16px); margin:6px 8px;
    background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.25);
    border-radius:6px; color:rgba(167,139,250,0.7); font-family:inherit;
    font-size:9.5px; letter-spacing:.1em; text-transform:uppercase;
    padding:5px 0; cursor:pointer; transition:all .2s;
  }
  #snap-btn:hover { background:rgba(124,58,237,0.2); color:rgba(200,180,255,0.9); }

  /* ── No-camera fallback ── */
  #nocam {
    display:none; padding:2rem 1rem; text-align:center;
    font-size:11px; color:rgba(0,212,255,0.25); line-height:2;
    background:#02040a;
  }

  /* ── Snapshot preview ── */
  #snap-preview { display:none; padding:6px 8px; background:#02040a; border-top:1px solid #0e1a2e; }
  #snap-preview img { width:100%; border-radius:6px; border:1px solid rgba(124,58,237,0.2); }
  #snap-preview p { font-size:8px; color:rgba(124,58,237,0.4); text-align:center; margin-top:3px; }

  /* Animations */
  @keyframes blink { 0%,100%{opacity:1}50%{opacity:.15} }
  @keyframes fadein { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }
  .fadein { animation: fadein .3s ease both; }
</style>
</head>
<body>

<div id="root">

  <!-- Top bar -->
  <div id="topbar">
    <div><span class="dot-red"></span><span id="live-lbl">LIVE</span></div>
    <span id="mode-pill">Initialising TF.js…</span>
    <span id="fps-lbl">-- fps</span>
  </div>

  <!-- Video + HUD canvas -->
  <div id="vidwrap">
    <video id="vid" autoplay playsinline muted></video>
    <canvas id="overlay"></canvas>
    <div id="hud">
      <!-- Eye contact -->
      <div id="eye-badge">
        <div id="eye-score">--</div>
        <div id="eye-lbl">Eye Contact</div>
      </div>
      <!-- Confidence ring -->
      <div id="conf-ring">
        <svg viewBox="0 0 54 54" fill="none">
          <circle cx="27" cy="27" r="22" stroke="#0f1e35" stroke-width="4.5"/>
          <circle id="conf-arc" cx="27" cy="27" r="22"
            stroke="#00d4ff" stroke-width="4.5" stroke-linecap="round"
            stroke-dasharray="0 138.2"
            style="transform-origin:center;transform:rotate(-90deg);transition:stroke-dasharray .5s ease,stroke .4s ease;filter:drop-shadow(0 0 5px rgba(0,212,255,0.4))"/>
        </svg>
        <div id="conf-center">
          <div id="conf-num">--</div>
          <div id="conf-lbl">Conf</div>
        </div>
      </div>
      <!-- Expression -->
      <div id="expr-badge">
        <span id="expr-icon">😐</span>
        <span id="expr-text">Neutral</span>
      </div>
      <!-- Posture -->
      <div id="posture-badge">
        <span id="posture-icon">📐</span>
        <span id="posture-text">Upright</span>
      </div>
      <!-- Scan line -->
      <div id="scan"></div>
    </div>
    <!-- No-cam fallback -->
    <div id="nocam">
      📷<br>
      Camera access required.<br>
      Please allow camera permissions<br>
      to enable AI presence analysis.
    </div>
  </div>

  <!-- Sparkline -->
  <div id="spark-wrap">
    <span id="spark-lbl">CONFIDENCE</span>
    <canvas id="sparkline"></canvas>
  </div>

  <!-- Coaching strip -->
  <div id="coach-strip">
    <span id="coach-icon">👁️</span>
    <span id="coach-text">Initialising presence analysis…</span>
  </div>

  <!-- Stats row -->
  <div id="stats-row">
    <div class="stat-cell">
      <div class="snum" id="stat-eye">--</div>
      <div class="slbl">Avg Eye</div>
    </div>
    <div class="stat-cell">
      <div class="snum" id="stat-conf">--</div>
      <div class="slbl">Avg Conf</div>
    </div>
    <div class="stat-cell">
      <div class="snum" id="stat-expr">--</div>
      <div class="slbl">Smile %</div>
    </div>
    <div class="stat-cell">
      <div class="snum" id="stat-frames">0</div>
      <div class="slbl">Frames</div>
    </div>
  </div>

  <!-- Snapshot button -->
  <button id="snap-btn">📸 Capture Snapshot</button>

  <!-- Snapshot preview -->
  <div id="snap-preview">
    <img id="snap-img" src="" alt="Snapshot">
    <p id="snap-time"></p>
  </div>

</div><!-- /root -->

<canvas id="hidden-canvas" style="display:none"></canvas>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface@0.0.7/dist/blazeface.min.js"></script>
<script>
// ── DOM refs ────────────────────────────────────────────────
const vid       = document.getElementById('vid');
const overlay   = document.getElementById('overlay');
const ctx2d     = overlay.getContext('2d');
const modePill  = document.getElementById('mode-pill');
const fpsLbl    = document.getElementById('fps-lbl');
const eyeScore  = document.getElementById('eye-score');
const confNum   = document.getElementById('conf-num');
const confArc   = document.getElementById('conf-arc');
const exprIcon  = document.getElementById('expr-icon');
const exprText  = document.getElementById('expr-text');
const postureIc = document.getElementById('posture-icon');
const postureT  = document.getElementById('posture-text');
const coachIcon = document.getElementById('coach-icon');
const coachText = document.getElementById('coach-text');
const sparkCv   = document.getElementById('sparkline');
const sparkCtx  = sparkCv.getContext('2d');
const statEye   = document.getElementById('stat-eye');
const statConf  = document.getElementById('stat-conf');
const statExpr  = document.getElementById('stat-expr');
const statFrames= document.getElementById('stat-frames');
const snapBtn   = document.getElementById('snap-btn');
const snapPrev  = document.getElementById('snap-preview');
const snapImg   = document.getElementById('snap-img');
const snapTime  = document.getElementById('snap-time');
const hiddenCv  = document.getElementById('hidden-canvas');
const nocam     = document.getElementById('nocam');

// ── State ───────────────────────────────────────────────────
let model = null;
let frameCount = 0, lastFpsTime = performance.now(), fps = 0;
let running = false;

// Rolling history (last 60 frames)
const HIST = 60;
const eyeHist   = [];
const confHist  = [];
let smileFrames = 0;
let totalFrames = 0;

// Smoothing EMA
let emaEye  = 50, emaConf = 50;
const EMA_A = 0.25;

// Coaching messages keyed by metric thresholds
const COACHING = [
  { cond: (e,c) => e < 35,                 icon:'👁️', text:'Maintain eye contact with the camera — look directly into the lens, not at your face on screen.' },
  { cond: (e,c) => e >= 35 && e < 60,      icon:'🎯', text:'Eye contact is forming — try to keep your gaze steady and centred for longer stretches.' },
  { cond: (e,c) => e >= 60 && c < 40,      icon:'💪', text:'Good eye contact. Work on posture — sit upright and roll your shoulders back to project confidence.' },
  { cond: (e,c) => c >= 40 && c < 65,      icon:'📐', text:'Posture looks reasonable. Keep your head level and avoid tilting — it signals uncertainty.' },
  { cond: (e,c) => e >= 60 && c >= 65 && c<85, icon:'✨', text:'Strong presence detected. Remember to breathe steadily and pause before answering.' },
  { cond: (e,c) => e >= 60 && c >= 85,     icon:'🏆', text:'Excellent presence! You look composed, confident, and engaged. Keep it up.' },
  { cond: (e,c) => false,                  icon:'😊', text:'Try to smile naturally — warmth and openness resonate with interviewers.' },
];
let coachIdx = 0, coachTimer = 0;

// Expression thresholds (heuristic from landmark ratios)
const EXPR_THRESHOLDS = {
  smile:   { icon:'😊', text:'Warm',    color:'rgba(0,200,150,0.65)' },
  neutral: { icon:'😐', text:'Neutral', color:'rgba(0,212,255,0.5)'  },
  tense:   { icon:'😬', text:'Tense',   color:'rgba(251,191,36,0.65)'},
  none:    { icon:'🔍', text:'Scanning',color:'rgba(61,85,128,0.5)'  },
};
let currentExpr = 'none';

// Posture states
const POSTURE = {
  upright:  { icon:'🟢', text:'Upright'  },
  slight:   { icon:'🟡', text:'Slight tilt'},
  leaning:  { icon:'🟠', text:'Leaning'  },
  off:      { icon:'⚫', text:'Off-frame' },
};

// ── Load model ───────────────────────────────────────────────
async function loadModel() {
  try {
    modePill.textContent = 'Loading BlazeFace…';
    model = await blazeface.load({ maxFaces: 1, scoreThreshold: 0.6 });
    modePill.textContent = 'BlazeFace · Ready';
    modePill.style.color = 'rgba(0,200,150,0.7)';
    modePill.style.borderColor = 'rgba(0,200,150,0.25)';
    modePill.style.background = 'rgba(0,200,150,0.06)';
  } catch(e) {
    modePill.textContent = 'Model unavailable';
    console.warn('BlazeFace load error:', e);
  }
}

// ── Start camera ─────────────────────────────────────────────
async function startCam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode:'user', width:{ideal:320}, height:{ideal:240} },
      audio: false,
    });
    vid.srcObject = stream;
    vid.onloadedmetadata = () => {
      overlay.width  = vid.videoWidth  || 320;
      overlay.height = vid.videoHeight || 240;
      hiddenCv.width  = overlay.width;
      hiddenCv.height = overlay.height;
      running = true;
      requestAnimationFrame(loop);
    };
    nocam.style.display = 'none';
  } catch(err) {
    vid.style.display = 'none';
    nocam.style.display = 'block';
    modePill.textContent = 'No camera access';
    // still run fake metrics for demo
    runFallbackMode();
  }
}

// ── Main detection loop ───────────────────────────────────────
async function loop(ts) {
  if (!running) return;

  // FPS
  frameCount++;
  if (ts - lastFpsTime >= 1000) {
    fps = frameCount; frameCount = 0; lastFpsTime = ts;
    fpsLbl.textContent = fps + ' fps';
  }

  ctx2d.clearRect(0, 0, overlay.width, overlay.height);

  if (model && vid.readyState === 4) {
    let predictions = [];
    try {
      predictions = await model.estimateFaces(vid, false);
    } catch(e) { /* ignore */ }

    if (predictions.length > 0) {
      const face = predictions[0];
      drawFaceHUD(face);
      processMetrics(face);
    } else {
      // No face
      drawNoFace();
      updateMetrics({ eye:0, conf:20, expr:'none', posture:'off' });
    }
  } else if (!model) {
    // Model not loaded, show scanning overlay
    drawScanOverlay();
    fakeMetrics();
  }

  updateSparkline();
  updateStats();
  requestAnimationFrame(loop);
}

// ── Face HUD drawing ─────────────────────────────────────────
function drawFaceHUD(face) {
  const [x1,y1] = face.topLeft;
  const [x2,y2] = face.bottomRight;
  const w = x2 - x1, h = y2 - y1;
  const cx = (x1+x2)/2, cy = (y1+y2)/2;

  // Mirror x because video is flipped
  const mx1 = overlay.width - x2, mx2 = overlay.width - x1, mcx = overlay.width - cx;

  // Confidence value from model
  const modelConf = Math.round((face.probability?.[0] ?? 0.8) * 100);

  // Draw face bounding box with corner accents
  const alpha = 0.6;
  ctx2d.strokeStyle = `rgba(0,212,255,${alpha})`;
  ctx2d.lineWidth = 1.5;
  const cr = 8; // corner radius for the accent
  const clen = Math.min(w,h) * 0.2;

  // Top-left corner
  ctx2d.beginPath();
  ctx2d.moveTo(mx1 + clen, y1); ctx2d.lineTo(mx1 + cr, y1);
  ctx2d.arcTo(mx1, y1, mx1, y1+cr, cr);
  ctx2d.lineTo(mx1, y1 + clen);
  ctx2d.stroke();

  // Top-right corner
  ctx2d.beginPath();
  ctx2d.moveTo(mx2 - clen, y1); ctx2d.lineTo(mx2 - cr, y1);
  ctx2d.arcTo(mx2, y1, mx2, y1+cr, cr);
  ctx2d.lineTo(mx2, y1 + clen);
  ctx2d.stroke();

  // Bottom-left corner
  ctx2d.beginPath();
  ctx2d.moveTo(mx1 + clen, y2); ctx2d.lineTo(mx1 + cr, y2);
  ctx2d.arcTo(mx1, y2, mx1, y2-cr, cr);
  ctx2d.lineTo(mx1, y2 - clen);
  ctx2d.stroke();

  // Bottom-right corner
  ctx2d.beginPath();
  ctx2d.moveTo(mx2 - clen, y2); ctx2d.lineTo(mx2 - cr, y2);
  ctx2d.arcTo(mx2, y2, mx2, y2-cr, cr);
  ctx2d.lineTo(mx2, y2 - clen);
  ctx2d.stroke();

  // Landmarks (6 BlazeFace points: right eye, left eye, nose, mouth, right ear, left ear)
  if (face.landmarks) {
    const lmColors = ['#00d4ff','#00d4ff','#fbbf24','#fb923c','#94a3b8','#94a3b8'];
    face.landmarks.forEach((lm, i) => {
      const lx = overlay.width - lm[0], ly = lm[1];
      ctx2d.beginPath();
      ctx2d.arc(lx, ly, 2.5, 0, Math.PI*2);
      ctx2d.fillStyle = lmColors[i] || '#00d4ff';
      ctx2d.fill();
    });

    // Eye gaze lines
    if (face.landmarks.length >= 2) {
      const re = face.landmarks[0], le = face.landmarks[1];
      const rex = overlay.width - re[0], rey = re[1];
      const lex = overlay.width - le[0], ley = le[1];

      // Eye-to-centre horizontal line
      const centX = overlay.width / 2, centY = overlay.height / 2;
      ctx2d.beginPath();
      ctx2d.moveTo(rex, rey); ctx2d.lineTo(lex, ley);
      ctx2d.strokeStyle = 'rgba(0,212,255,0.15)';
      ctx2d.lineWidth = 1; ctx2d.stroke();

      // Vertical midline
      ctx2d.beginPath();
      ctx2d.moveTo(centX, y1 - 4); ctx2d.lineTo(centX, y2 + 4);
      ctx2d.strokeStyle = 'rgba(0,212,255,0.08)';
      ctx2d.setLineDash([3,5]); ctx2d.stroke(); ctx2d.setLineDash([]);
    }

    // Compute metrics
    const eyeScore  = computeEyeContact(face, overlay.width, overlay.height);
    const tiltAngle = computeTilt(face);
    const exprClass = computeExpression(face, h);
    const postState = tiltAngle < 8 ? 'upright' : tiltAngle < 18 ? 'slight' : 'leaning';

    updateMetrics({ eye:eyeScore, conf: Math.round(modelConf * 0.6 + eyeScore * 0.4), expr:exprClass, posture:postState });
  } else {
    updateMetrics({ eye:60, conf:modelConf, expr:'neutral', posture:'upright' });
  }
}

// ── Metric computations ───────────────────────────────────────
function computeEyeContact(face, vw, vh) {
  if (!face.landmarks || face.landmarks.length < 2) return 50;
  const re = face.landmarks[0], le = face.landmarks[1];
  // Mirror
  const rex = vw - re[0], ley_x = vw - le[0];
  const eyeMidX = (rex + ley_x) / 2;
  const eyeMidY = (re[1] + le[1]) / 2;
  const centerX = vw / 2, centerY = vh * 0.42;
  const dx = Math.abs(eyeMidX - centerX) / (vw * 0.5);
  const dy = Math.abs(eyeMidY - centerY) / (vh * 0.5);
  const dist = Math.sqrt(dx*dx + dy*dy);
  return Math.round(Math.max(0, Math.min(100, (1 - dist * 1.4) * 100)));
}

function computeTilt(face) {
  if (!face.landmarks || face.landmarks.length < 2) return 0;
  const re = face.landmarks[0], le = face.landmarks[1];
  const dx = le[0] - re[0], dy = le[1] - re[1];
  return Math.abs(Math.atan2(dy, dx) * 180 / Math.PI);
}

function computeExpression(face, faceH) {
  if (!face.landmarks || face.landmarks.length < 4) return 'neutral';
  const mouth = face.landmarks[3];
  const nose  = face.landmarks[2];
  // Heuristic: mouth y relative to nose y, normalised by face height
  const ratio = (mouth[1] - nose[1]) / Math.max(faceH * 0.5, 1);
  if (ratio > 0.55) return 'smile';
  if (ratio < 0.30) return 'tense';
  return 'neutral';
}

function drawNoFace() {
  // Dashed centre reticle
  ctx2d.strokeStyle = 'rgba(0,212,255,0.08)';
  ctx2d.lineWidth = 1;
  ctx2d.setLineDash([4,8]);
  ctx2d.strokeRect(overlay.width*0.25, overlay.height*0.15, overlay.width*0.5, overlay.height*0.7);
  ctx2d.setLineDash([]);
  ctx2d.fillStyle = 'rgba(0,212,255,0.06)';
  ctx2d.font = '10px Geist Mono, monospace';
  ctx2d.textAlign = 'center';
  ctx2d.fillText('Position face in frame', overlay.width/2, overlay.height*0.92);
}

function drawScanOverlay() {
  ctx2d.fillStyle = 'rgba(0,212,255,0.03)';
  ctx2d.fillRect(0, 0, overlay.width, overlay.height);
  ctx2d.fillStyle = 'rgba(0,212,255,0.08)';
  ctx2d.font = '10px Geist Mono, monospace';
  ctx2d.textAlign = 'center';
  ctx2d.fillText('Loading AI model…', overlay.width/2, overlay.height/2);
}

// ── Metric updates ────────────────────────────────────────────
function updateMetrics({ eye, conf, expr, posture }) {
  totalFrames++;
  if (expr === 'smile') smileFrames++;

  // EMA smoothing
  emaEye  = emaEye  * (1-EMA_A) + eye  * EMA_A;
  emaConf = emaConf * (1-EMA_A) + conf * EMA_A;

  const e = Math.round(emaEye), c = Math.round(emaConf);

  eyeHist.push(e);  if (eyeHist.length > HIST)  eyeHist.shift();
  confHist.push(c); if (confHist.length > HIST) confHist.shift();

  // Eye badge
  const eyeColor = e >= 65 ? '#00c896' : e >= 40 ? '#fbbf24' : '#fb2c36';
  eyeScore.textContent = e + '%';
  eyeScore.style.color = eyeColor;

  // Confidence ring
  const CIRC = 138.2;
  confNum.textContent = c;
  confNum.style.color = c>=65?'#00c896':c>=40?'#fbbf24':'#fb2c36';
  const dash = (c/100)*CIRC;
  confArc.setAttribute('stroke-dasharray', `${dash} ${CIRC}`);
  confArc.setAttribute('stroke', c>=65?'#00c896':c>=40?'#fbbf24':'#fb2c36');

  // Expression
  currentExpr = expr;
  const exInfo = EXPR_THRESHOLDS[expr] || EXPR_THRESHOLDS.neutral;
  exprIcon.textContent = exInfo.icon;
  exprText.textContent = exInfo.text;
  exprText.style.color = exInfo.color;

  // Posture
  const pInfo = POSTURE[posture] || POSTURE.upright;
  postureIc.textContent = pInfo.icon;
  postureT.textContent  = pInfo.text;

  // Coaching (rotate every 6 seconds)
  coachTimer++;
  if (coachTimer > fps * 6 || coachTimer === 1) {
    coachTimer = 0;
    const match = COACHING.find(c2 => c2.cond(e, c));
    if (match) { coachIcon.textContent = match.icon; coachText.textContent = match.text; }
  }
}

// ── Sparkline ─────────────────────────────────────────────────
function updateSparkline() {
  const sw = sparkCv.width  = sparkCv.offsetWidth  || 180;
  const sh = sparkCv.height = sparkCv.offsetHeight || 22;
  sparkCtx.clearRect(0, 0, sw, sh);

  if (confHist.length < 2) return;
  const step = sw / (HIST - 1);

  sparkCtx.beginPath();
  confHist.forEach((v, i) => {
    const x = i * step;
    const y = sh - (v/100) * sh;
    i === 0 ? sparkCtx.moveTo(x,y) : sparkCtx.lineTo(x,y);
  });
  sparkCtx.strokeStyle = 'rgba(0,212,255,0.45)';
  sparkCtx.lineWidth = 1.5;
  sparkCtx.stroke();

  // Fill area
  sparkCtx.lineTo((confHist.length-1)*step, sh);
  sparkCtx.lineTo(0, sh);
  sparkCtx.closePath();
  sparkCtx.fillStyle = 'rgba(0,212,255,0.06)';
  sparkCtx.fill();
}

// ── Session stats ─────────────────────────────────────────────
function updateStats() {
  if (eyeHist.length === 0) return;
  const avgEye  = Math.round(eyeHist.reduce((a,b)=>a+b,0)/eyeHist.length);
  const avgConf = Math.round(confHist.reduce((a,b)=>a+b,0)/confHist.length);
  const smilePct = totalFrames>0 ? Math.round((smileFrames/totalFrames)*100) : 0;

  statEye.textContent    = avgEye + '%';
  statConf.textContent   = avgConf;
  statExpr.textContent   = smilePct + '%';
  statFrames.textContent = totalFrames;

  statEye.style.color  = avgEye>=65?'#00c896':avgEye>=40?'#fbbf24':'#fb2c36';
  statConf.style.color = avgConf>=65?'#00c896':avgConf>=40?'#fbbf24':'#fb2c36';
}

// ── Fallback mode (no camera but demo metrics) ────────────────
function runFallbackMode() {
  modePill.textContent = 'No Camera · Demo Mode';
  let t = 0;
  setInterval(() => {
    t++;
    const e = 40 + Math.round(Math.sin(t*0.15)*25 + Math.random()*10);
    const c = 50 + Math.round(Math.cos(t*0.1)*20 + Math.random()*8);
    updateMetrics({ eye:e, conf:c, expr:t%20<3?'smile':'neutral', posture:Math.abs(e-50)>15?'slight':'upright' });
    updateSparkline(); updateStats();
  }, 150);
}

// ── Fake metrics when model loading ───────────────────────────
let _ft = 0;
function fakeMetrics() {
  _ft++;
  const e = 55 + Math.round(Math.sin(_ft*0.08)*12);
  updateMetrics({ eye:e, conf:60, expr:'neutral', posture:'upright' });
}

// ── Snapshot ──────────────────────────────────────────────────
snapBtn.addEventListener('click', () => {
  const hc = hiddenCv.getContext('2d');
  hc.save();
  hc.scale(-1,1); hc.drawImage(vid, -hiddenCv.width, 0);
  hc.restore();
  // Timestamp overlay
  const now = new Date().toLocaleTimeString();
  hc.fillStyle = 'rgba(0,0,0,0.55)';
  hc.fillRect(0, hiddenCv.height-22, hiddenCv.width, 22);
  hc.fillStyle = 'rgba(0,212,255,0.7)';
  hc.font = '9px Geist Mono, monospace';
  hc.fillText('KETU AI · ' + now, 8, hiddenCv.height-8);

  snapImg.src = hiddenCv.toDataURL('image/png');
  snapTime.textContent = 'Captured at ' + now;
  snapPrev.style.display = 'block';
  snapPrev.classList.add('fadein');
  setTimeout(()=>snapPrev.classList.remove('fadein'), 500);
});

// ── Boot ──────────────────────────────────────────────────────
(async () => {
  await loadModel();
  await startCam();
})();
</script>
</body>
</html>
""", height=570, scrolling=False)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown('<div style="font-family:Geist,sans-serif;font-weight:900;font-size:1.7rem;color:#00d4ff;margin-bottom:0.1rem;letter-spacing:-0.04em">KETU AI <span style="font-size:0.7rem;color:#3d5580;letter-spacing:0.2em;font-weight:400">v2.1</span></div>', unsafe_allow_html=True)
        st.markdown('<div style="font-family:Geist Mono,monospace;font-size:0.62rem;color:#1e3258;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:1rem">Elite Interview Intelligence</div>', unsafe_allow_html=True)
        st.markdown("---")

        screen = st.session_state.get("screen","setup")
        persona = PERSONAS.get(st.session_state.get("persona","Ketu"), PERSONAS["Ketu"])

        if screen == "interview":
            idx = st.session_state.current
            n   = len(st.session_state.questions)
            st.progress(idx / max(n,1))
            c1, c2 = st.columns(2)
            c1.metric("Question", f"{idx}/{n}")
            if st.session_state.scores:
                avg = sum(s.get("score",0) for s in st.session_state.scores) / len(st.session_state.scores)
                c2.metric("Grade", grade_letter(avg))
            st.markdown(f'<div style="font-family:Geist Mono,monospace;font-size:0.72rem;color:#3d5580;margin:0.5rem 0">Interviewer: {persona["avatar"]} {persona["name"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:Geist Mono,monospace;font-size:0.72rem;color:#3d5580;margin-bottom:0.8rem">Mode: {st.session_state.interview_mode}</div>', unsafe_allow_html=True)
            st.markdown("---")
            if st.button("⏹ End Interview", use_container_width=True):
                st.session_state.screen = "results"; st.rerun()

            # Camera toggle
            prev_cam = st.session_state.camera_enabled
            st.session_state.camera_enabled = st.toggle(
                "📷 AI Presence Monitor",
                value=st.session_state.get("camera_enabled", False),
                help="Enables BlazeFace-powered eye contact, confidence & expression tracking"
            )
            if st.session_state.camera_enabled:
                camera_panel()
            elif prev_cam and not st.session_state.camera_enabled:
                st.markdown('<div class="tip" style="margin-top:0.5rem">Camera monitoring paused.</div>', unsafe_allow_html=True)

        elif screen == "results":
            if st.session_state.scores:
                avg = sum(s.get("score",0) for s in st.session_state.scores)/len(st.session_state.scores)
                st.success(f"Interview complete · {grade_letter(avg)}")
                st.metric("Final Score", f"{avg:.1f}/10")
                st.markdown("---")
                if st.button("🔄 New Interview", use_container_width=True):
                    for k in list(st.session_state.keys()): del st.session_state[k]
                    st.rerun()

        st.markdown("---")
        st.markdown('<div style="font-family:Geist Mono,monospace;font-size:0.62rem;color:#1e3258;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.5rem">Capabilities v2.1</div>', unsafe_allow_html=True)
        features = [
            "4 interviewer personas","8 competency frameworks","3 pressure modes",
            "Resume deep analysis","Skills gap detection","Adaptive follow-ups",
            "Live STAR tracking","Filler word analysis","Real-time coaching",
            "Specificity scoring","Competency radar","Score timeline",
            "CSV + JSON export","Whisper voice input","TTS delivery",
            "BlazeFace AI detection","Eye contact scoring","Expression analysis",
            "Posture estimation","Confidence sparkline","Snapshot capture",
        ]
        for f in features:
            st.markdown(f'<div style="font-family:Geist Mono,monospace;font-size:0.65rem;color:#1e3258;padding:0.15rem 0">· {f}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption(datetime.now().strftime("%H:%M · %d %b %Y"))


# ─────────────────────────────────────────────────────────────
# SCREEN — SETUP
# ─────────────────────────────────────────────────────────────
def screen_setup():
    st.markdown("""
    <div class="hero">
        <div style="display:flex;justify-content:center;margin-bottom:2rem">
            <div class="hero-kicker"><div class="hero-dot"></div>Adaptive · Multi-Persona · AI Presence Analysis</div>
        </div>
        <div class="hero-wordmark">KETU AI</div>
        <div class="hero-sub-title">next-generation interview intelligence</div>
        <p class="hero-desc">Meet your elite AI interviewer. Adaptive follow-ups, resume analysis, STAR tracking, competency mapping, live camera presence scoring, and feedback that genuinely makes you better.</p>
        <div class="hero-badges">
            <span class="hero-badge">4 Personas</span>
            <span class="hero-badge">8 Competency Frameworks</span>
            <span class="hero-badge">Live STAR Tracking</span>
            <span class="hero-badge">Resume Intelligence</span>
            <span class="hero-badge">Voice Input</span>
            <span class="hero-badge">📷 AI Presence Monitor</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    llm = get_llm()
    if llm is None:
        st.error("⚠️ `GROQ_API_KEY` not found. Add it to `.streamlit/secrets.toml`.")
        return

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="glass glass-electric">', unsafe_allow_html=True)
        st.markdown('<div class="sec sec-electric">🎭 Choose Your Interviewer</div>', unsafe_allow_html=True)
        p_cols = st.columns(4)
        for i, (pn, pd_) in enumerate(PERSONAS.items()):
            with p_cols[i]:
                active = st.session_state.persona == pn
                st.markdown(f"""<div class="persona-card {"active" if active else ""}">
                    <div class="persona-emoji">{pd_['avatar']}</div>
                    <div class="persona-name">{pn}</div>
                    <div class="persona-role">{pd_['title']}</div>
                </div>""", unsafe_allow_html=True)
                if st.button("✓ Active" if active else "Select", key=f"p_{pn}", use_container_width=True):
                    st.session_state.persona = pn; st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">📋 Job Details</div>', unsafe_allow_html=True)
        st.session_state.candidate_name = st.text_input("Your Name (optional)", placeholder="e.g. Arjun Mehta", value=st.session_state.candidate_name)
        st.session_state.role_title = st.text_input("Role / Job Title *", placeholder="e.g. Senior Backend Engineer", value=st.session_state.role_title)

        c1, c2 = st.columns(2)
        with c1:
            cat = st.selectbox("Role Category", list(COMPETENCY_FRAMEWORKS.keys()), index=list(COMPETENCY_FRAMEWORKS.keys()).index(st.session_state.category_tag))
            st.session_state.category_tag = cat
        with c2:
            mode = st.selectbox("Interview Mode", list(INTERVIEW_MODES.keys()), index=list(INTERVIEW_MODES.keys()).index(st.session_state.interview_mode))
            st.session_state.interview_mode = mode

        st.markdown(f"""<div class="mode-grid" style="margin-top:0.5rem;margin-bottom:1rem">
            {''.join([
                f'<div class="mode-card {"m-"+m.lower() if st.session_state.interview_mode==m else ""}"><div class="mode-title">{INTERVIEW_MODES[m]["emoji"]} {m}</div><div class="mode-desc">{INTERVIEW_MODES[m]["desc"]}</div><span class="mode-pill {INTERVIEW_MODES[m]["pill_cls"]}">{INTERVIEW_MODES[m]["pressure"].upper()} PRESSURE</span></div>'
                for m in INTERVIEW_MODES
            ])}
        </div>""", unsafe_allow_html=True)

        st.session_state.jd_text = st.text_area("Job Description *", height=250, placeholder="Paste the full job description here…", value=st.session_state.jd_text)

        st.markdown('<div class="sec" style="margin-top:1rem">⚙️ Settings</div>', unsafe_allow_html=True)
        c3, c4, c5, c6 = st.columns(4)
        with c3: st.session_state.num_questions = st.slider("Questions", 4, 15, st.session_state.num_questions)
        with c4: st.session_state.tts_enabled = st.toggle("🔊 Voice TTS", value=st.session_state.tts_enabled)
        with c5: st.session_state.show_hints  = st.toggle("💡 Show Hints", value=st.session_state.show_hints)
        with c6: st.session_state.camera_enabled = st.toggle("📷 Camera", value=st.session_state.get("camera_enabled", False))

        comps = COMPETENCY_FRAMEWORKS.get(cat, [])
        comp_html = "".join([f'<span class="skill-tag sk-neutral">{c}</span>' for c in comps])
        st.markdown(f'<div class="sec" style="margin-top:0.8rem">📊 Competencies to Assess</div><div class="skills-match">{comp_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        # Camera preview on setup (if enabled)
        if st.session_state.camera_enabled:
            st.markdown('<div class="glass" style="padding:1rem;margin-bottom:0.8rem">', unsafe_allow_html=True)
            st.markdown('<div class="sec sec-electric">📷 AI Presence Monitor — Preview</div>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:Geist Mono,monospace;font-size:0.7rem;color:var(--t3);margin-bottom:0.7rem">BlazeFace · Eye Contact · Expression · Confidence · Posture</div>', unsafe_allow_html=True)
            camera_panel()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass glass-neon">', unsafe_allow_html=True)
        st.markdown('<div class="sec sec-neon">📄 Resume Upload & Analysis</div>', unsafe_allow_html=True)

        resume_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf","docx","doc","txt"], label_visibility="collapsed")
        if resume_file:
            with st.spinner("🔍 Reading & analysing resume…"):
                text = load_doc(resume_file)
                st.session_state.resume_text = text
                if st.session_state.jd_text.strip():
                    profile = analyze_resume(text, st.session_state.jd_text, st.session_state.role_title or "this role", llm)
                    st.session_state.resume_profile = profile
                    if profile.get("candidate_name") and profile["candidate_name"] != "Candidate":
                        if not st.session_state.candidate_name:
                            st.session_state.candidate_name = profile["candidate_name"]

            profile = st.session_state.resume_profile or {}
            match_html = "".join([f'<span class="skill-tag sk-match">✓ {s}</span>' for s in profile.get("matching_skills",[])[:4]])
            gap_html   = "".join([f'<span class="skill-tag sk-gap">✗ {s}</span>' for s in profile.get("gap_skills",[])[:3]])
            skill_html = "".join([f'<span class="skill-tag sk-neutral">{s}</span>' for s in profile.get("top_skills",[])[:5]])
            fit_score  = profile.get("overall_fit_score", 0)
            fit_color  = score_color(fit_score)

            st.markdown(f"""
            <div class="resume-profile" style="margin-top:1rem">
                <div class="rp-name">{profile.get('candidate_name','Candidate')}</div>
                <div class="rp-role">{profile.get('current_role','Professional')} · {profile.get('years_experience','N/A')} · {profile.get('education','N/A')}</div>
                <div style="margin-top:0.6rem;font-family:'Geist Mono',monospace;font-size:0.65rem;color:var(--t3)">
                    Companies: {' · '.join(profile.get('companies',[])[:3]) or 'N/A'}
                </div>
                <div class="rp-chips" style="margin-top:0.8rem">{skill_html}</div>
                {f'<div style="margin-top:0.6rem">{match_html}{gap_html}</div>' if match_html or gap_html else ''}
                <div class="rp-stats">
                    <div><div class="rp-stat-num" style="color:{fit_color}">{fit_score:.1f}/10</div><div class="rp-stat-lbl">Role Fit</div></div>
                    <div><div class="rp-stat-num">{len(profile.get('matching_skills',[]))}</div><div class="rp-stat-lbl">Skill Matches</div></div>
                    <div><div class="rp-stat-num">{len(profile.get('gap_skills',[]))}</div><div class="rp-stat-lbl">Skill Gaps</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if profile.get("fit_rationale"):
                st.markdown(f'<div class="tip neon" style="margin-top:0.7rem">🎯 {profile["fit_rationale"]}</div>', unsafe_allow_html=True)

            with st.expander("📋 Full resume text"):
                st.text(text[:900] + "…")
        else:
            st.markdown("""
            <div style="border:1px dashed rgba(124,58,237,0.3);border-radius:16px;padding:2rem 1.5rem;text-align:center;background:rgba(124,58,237,0.03)">
                <div style="font-size:2rem;margin-bottom:0.6rem">📄</div>
                <div style="font-family:'Geist',sans-serif;font-weight:600;font-size:0.9rem;color:var(--t1)">Drop your resume here</div>
                <div style="font-family:'Geist Mono',monospace;font-size:0.65rem;color:var(--t3);margin-top:0.3rem">PDF · DOCX · TXT · Auto-analysed against JD</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="tip" style="margin-top:0.8rem">
        🧠 KETU AI v2.1 adds live camera intelligence: TensorFlow.js BlazeFace tracks your eye contact, expression, and posture in real-time — giving you presence coaching alongside answer analysis.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        persona = PERSONAS.get(st.session_state.persona, PERSONAS["Ketu"])
        if st.button(f"⚡  Begin Interview with {persona['name']}", use_container_width=True):
            if not st.session_state.jd_text.strip():
                st.error("Please paste a job description.")
            elif not st.session_state.resume_text.strip():
                st.error("Please upload a resume.")
            elif not st.session_state.role_title.strip():
                st.error("Please enter the role / job title.")
            else:
                with st.spinner(f"🤖 {persona['name']} is reviewing your profile and crafting tailored questions…"):
                    qs, ts, cs, dfs = gen_questions(
                        st.session_state.jd_text, st.session_state.resume_text,
                        st.session_state.role_title, st.session_state.num_questions,
                        llm, st.session_state.persona, st.session_state.interview_mode,
                        st.session_state.category_tag, st.session_state.get("resume_profile"),
                    )
                if not qs:
                    st.error("Could not generate questions. Check your API key.")
                    return
                greeting = random.choice(persona["greetings"])
                for k in ["questions","q_types","q_competencies","q_difficulties","scores","feedback_list",
                          "transcript","competency_scores","filler_counts","word_counts","ai_summary"]:
                    st.session_state[k] = [] if k != "competency_scores" and k != "ai_summary" else ({} if k=="competency_scores" else None)
                st.session_state.questions        = qs
                st.session_state.q_types          = ts
                st.session_state.q_competencies   = cs
                st.session_state.q_difficulties   = dfs
                st.session_state.current          = 0
                st.session_state.session_start    = time.time()
                st.session_state.q_start          = time.time()
                st.session_state.submitted        = False
                st.session_state.ketu_message     = greeting
                st.session_state.is_followup      = False
                st.session_state.followup_count   = 0
                st.session_state.screen           = "interview"
                st.rerun()


# ─────────────────────────────────────────────────────────────
# SCREEN — INTERVIEW
# ─────────────────────────────────────────────────────────────
def screen_interview():
    llm        = get_llm()
    idx        = st.session_state.current
    questions  = st.session_state.questions
    q_types    = st.session_state.q_types
    q_comps    = st.session_state.q_competencies
    q_diffs    = st.session_state.q_difficulties
    persona    = PERSONAS.get(st.session_state.persona, PERSONAS["Ketu"])
    mode       = st.session_state.interview_mode
    mode_cfg   = INTERVIEW_MODES.get(mode, INTERVIEW_MODES["Standard"])
    n          = len(questions)

    if idx >= n:
        st.session_state.screen = "results"; st.rerun()

    q          = questions[idx]
    q_type     = q_types[idx]  if idx < len(q_types) else "technical"
    competency = q_comps[idx]  if idx < len(q_comps) else "Technical Depth"
    difficulty = q_diffs[idx]  if idx < len(q_diffs) else "medium"
    q_info     = QUESTION_TYPES.get(q_type, ("❓","qb-tech",q_type.title()))
    diff_cls   = {"easy":"diff-e","medium":"diff-m","hard":"diff-h"}.get(difficulty,"diff-m")

    elapsed = int(time.time() - (st.session_state.session_start or time.time()))
    mins, secs = divmod(elapsed, 60)
    avg_so_far = (sum(s.get("score",0) for s in st.session_state.scores) / len(st.session_state.scores)) if st.session_state.scores else 0.0

    tb1, tb2, tb3, tb4, tb5, tb6 = st.columns([4,1,1,1,1,1])
    with tb1:
        st.progress(idx / n)
        st.caption(f"Q{idx+1} / {n}  ·  {mins:02d}:{secs:02d}  ·  {mode} mode  ·  {persona['name']}")
    with tb2: st.metric("Avg", f"{avg_so_far:.1f}")
    with tb3: st.metric("Done", f"{len(st.session_state.scores)}/{n}")
    with tb4: st.metric("Words", f"{sum(st.session_state.word_counts)}")
    with tb5:
        fc = sum(st.session_state.filler_counts)
        st.metric("Fillers", f"{fc}")
    with tb6:
        if st.button("⏹ End", help="End interview"): st.session_state.screen = "results"; st.rerun()

    st.markdown("---")

    msg = st.session_state.get("ketu_message","")
    is_followup = st.session_state.get("is_followup", False)
    speaking = "speaking" if msg and not st.session_state.submitted else ""

    st.markdown(f"""
    <div class="avatar-bar">
        <div class="avatar-icon {speaking}">{persona['avatar']}</div>
        <div class="avatar-meta">
            <div class="avatar-name">{persona['name']}</div>
            <div class="avatar-status"><span class="status-led"></span>{persona['title']}</div>
        </div>
        <div class="avatar-speech"><span class="speech-open">"</span>{msg or "Ready for your answer…"}<span class="speech-close">"</span></div>
    </div>
    """, unsafe_allow_html=True)

    tts_key = f"tts_{idx}_{hash(msg)}"
    if msg and tts_key not in st.session_state:
        tts_play(msg); st.session_state[tts_key] = True

    if is_followup:
        st.markdown('<div class="followup-strip">🔄 Follow-up — Probing deeper</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="q-card">
        <div class="q-counter">Question {idx+1} of {n}</div>
        <p class="q-text">{q}</p>
        <div class="q-meta">
            <span class="q-badge {q_info[1]}">{q_info[0]} {q_info[2]}</span>
            <span class="q-comp-tag">📊 {competency}</span>
            <span class="q-diff {diff_cls}">{difficulty.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.submitted:
        if HAS_AUDIO_RECORDER:
            st.markdown('<div class="sec">🎙️ Voice Answer</div>', unsafe_allow_html=True)
            st.markdown('<div style="display:flex;align-items:center;justify-content:space-between;padding:0.6rem 1rem;background:rgba(251,44,54,0.04);border:1px solid rgba(251,44,54,0.15);border-radius:12px;margin-bottom:0.7rem"><div style="display:flex;align-items:center;gap:0.45rem;font-family:\'Geist Mono\',monospace;font-size:0.7rem;color:rgba(251,100,100,0.7)"><span style="width:6px;height:6px;border-radius:50%;background:#fb2c36;animation:blink-led 1s ease infinite;display:inline-block"></span>Click to record · Groq Whisper</div><span style="font-family:\'Geist Mono\',monospace;font-size:0.65rem;color:var(--t4)">Speak clearly</span></div>', unsafe_allow_html=True)
            audio_bytes = audio_recorder(text="", icon_size="2x", key=f"rec_{idx}")
            if audio_bytes and f"tr_{idx}" not in st.session_state:
                st.markdown(waveform_html(), unsafe_allow_html=True)
                with st.spinner("Transcribing…"):
                    text = transcribe(audio_bytes)
                    if text:
                        st.session_state[f"ans_{idx}"] = text
                        st.session_state[f"tr_{idx}"] = True
                        st.rerun()

        st.markdown('<div class="sec">✍️ Written Answer</div>', unsafe_allow_html=True)
        if f"tr_{idx}" in st.session_state:
            st.info(f"🎙️ Transcribed: *{st.session_state.get(f'ans_{idx}','')}*")

        ans = st.text_area(
            "Your response",
            value=st.session_state.get(f"ans_{idx}",""),
            key=f"in_{idx}", height=175,
            placeholder="Type your answer here, or use the voice recorder above…",
            label_visibility="collapsed",
        )

        if ans.strip():
            qa = analyze_quality(ans)
            wc = qa["wc"]
            pct = min(wc/250, 1.0)
            mc  = score_color(wc/25) if 80<=wc<=250 else "#fbbf24" if wc<80 else "#fb2c36"
            status = "Ideal ✓" if 80<=wc<=250 else "Too short" if wc<80 else "Too long"

            st.markdown(f"""
            <div class="word-meter">
                <span class="wm-count">{wc} words</span>
                <div class="wm-track"><div class="wm-fill" style="width:{pct*100:.0f}%;background:{mc}"></div></div>
                <span class="wm-status" style="color:{mc}">{status}</span>
            </div>""", unsafe_allow_html=True)

            htype, hicon, htext = qa["hint"]
            hint_cls = {"warn":"coach-warn","success":"coach-success","info":"coach-info"}.get(htype,"coach-info")
            st.markdown(f'<div class="coach-bar {hint_cls}"><span class="coach-icon">{hicon}</span>{htext}</div>', unsafe_allow_html=True)

            if q_type in ("behavioral","situational") and wc > 30:
                star = qa["star"]
                star_cells = "".join([
                    f'<div class="star-cell {"active" if v else ""}"><div class="star-label">{k}</div><div class="star-val {"star-y" if v else "star-n"}">{"✓" if v else "○"}</div></div>'
                    for k, v in star.items()
                ])
                st.markdown(f'<div style="margin-top:0.6rem"><div class="sec" style="margin-bottom:0.35rem">⭐ STAR Coverage</div><div class="star-grid">{star_cells}</div></div>', unsafe_allow_html=True)

            if qa["specificity"] >= 2:
                st.markdown(f'<div class="coach-bar coach-success"><span class="coach-icon">📊</span>Good use of specific details and numbers — that strengthens credibility.</div>', unsafe_allow_html=True)

        elif st.session_state.show_hints:
            tips = {
                "technical":  "⚙️ Mention specific tools, architectures, and measurable outcomes.",
                "behavioral": "🧠 Use the STAR method: Situation · Task · Action · Result.",
                "rapport":    "💬 Be authentic — this question is about knowing you, not testing you.",
                "situational":"🎯 Walk through your thinking step-by-step. Trade-offs matter.",
                "ambition":   "🚀 Connect your goals directly to what excites you about this role.",
            }
            st.markdown(f'<div class="tip">{tips.get(q_type,"💡 Take your time and be specific.")}</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([3,1,1])
        with c1: submit = st.button("✓  Submit Answer", use_container_width=True)
        with c2: skip   = st.button("Skip →",           use_container_width=True)
        with c3: hint   = st.button("💡 Hint",           use_container_width=True)

        if hint:
            st.markdown(f"""<div class="coach-bar coach-info">
                <span class="coach-icon">🎯</span>
                For a <b>{q_type}</b> question on <b>{competency}</b>: focus on specifics over generalities,
                quantified outcomes, and what YOU personally did vs what the team did.
                {"For behavioral: use STAR — describe the Situation, your Task, specific Actions, and Results." if q_type in ("behavioral","situational") else ""}
            </div>""", unsafe_allow_html=True)

        if skip:
            st.session_state.transcript.append({"role":"user","content":"[Skipped]","q":q})
            st.session_state.current   += 1
            st.session_state.submitted  = False
            st.session_state.is_followup = False
            st.session_state.ketu_message = random.choice(persona["transitions"])
            st.session_state.q_start = time.time()
            st.rerun()

        if submit:
            if not ans.strip():
                st.warning("Please provide an answer before submitting.")
            else:
                qa = analyze_quality(ans)
                st.session_state.filler_counts.append(qa["filler_count"])
                st.session_state.word_counts.append(qa["wc"])

                with st.spinner(random.choice(persona["thinking"])):
                    ev = evaluate(q, ans, st.session_state.role_title, q_type, competency,
                                  mode, st.session_state.persona, llm, st.session_state.transcript[-6:])
                    ev["_qa"] = qa

                st.session_state.transcript.append({"role":"user","content":ans,"q":q})
                st.session_state.transcript.append({"role":persona["name"],"content":ev.get("interviewer_reaction","")})

                if not is_followup:
                    st.session_state.scores.append(ev)
                    st.session_state.feedback_list.append({
                        "q":q,"a":ans,"eval":ev,"type":q_type,"competency":competency,
                        "difficulty":difficulty,"time":int(time.time()-(st.session_state.q_start or time.time())),
                        "qa":qa,
                    })
                    csc = ev.get("competency_score", ev.get("score",5.0))
                    if competency not in st.session_state.competency_scores:
                        st.session_state.competency_scores[competency] = []
                    st.session_state.competency_scores[competency].append(csc)
                else:
                    if st.session_state.scores:
                        prev = st.session_state.scores[-1]["score"]
                        st.session_state.scores[-1]["score"] = min(10.0, (prev + ev["score"])/2 + 0.5)

                st.session_state.current_feedback = ev
                st.session_state.submitted = True
                st.session_state._pending_followup = (
                    ev.get("needs_followup",False) and ev.get("followup_question","")
                    and st.session_state.followup_count < mode_cfg["max_followups"] and not is_followup
                )
                st.rerun()

    else:
        f = st.session_state.current_feedback
        sc = f.get("score",5.0)
        sc_color = score_color(sc)
        reaction = f.get("interviewer_reaction","")
        qa_local = f.get("_qa",{})
        tones = f.get("tone_signals",[])
        q_type = q_types[idx] if idx < len(q_types) else "technical"

        if reaction:
            st.markdown(f"""
            <div class="avatar-bar">
                <div class="avatar-icon">{persona['avatar']}</div>
                <div class="avatar-meta">
                    <div class="avatar-name">{persona['name']}</div>
                    <div class="avatar-status"><span class="status-led busy"></span>Reviewing</div>
                </div>
                <div class="avatar-speech"><span class="speech-open">"</span>{reaction}<span class="speech-close">"</span></div>
            </div>
            """, unsafe_allow_html=True)
            react_key = f"tr_{idx}_{hash(reaction)}"
            if react_key not in st.session_state:
                tts_play(reaction); st.session_state[react_key] = True

        tone_html = ""
        for t in tones:
            cls = "tc-pos" if t in POSITIVE_TONE else "tc-neg" if t in NEGATIVE_TONE else "tc-neu"
            tone_html += f'<span class="tone-chip {cls}">{t}</span>'

        ring = ring_svg(sc)
        st.markdown(f"""
        <div class="fb-card">
            <div class="fb-score-row">
                <div class="fb-ring-wrap">
                    {ring}
                    <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:'Geist',sans-serif;font-size:1.4rem;font-weight:800;color:{sc_color}">{sc:.1f}</div>
                </div>
                <div>
                    <div class="fb-verdict">{f.get('verdict','Average')}</div>
                    <div class="fb-sub">{competency} · /10</div>
                    <div class="tone-chips">{tone_html}</div>
                </div>
            </div>
            <div class="fb-sec"><div class="fb-lbl fb-lbl-str">✓ Strength</div><div class="fb-text">{f.get('strength','—')}</div></div>
            <div class="fb-sec"><div class="fb-lbl fb-lbl-gap">✗ Gap</div><div class="fb-text">{f.get('weakness','—')}</div></div>
            <div class="fb-sec"><div class="fb-lbl fb-lbl-sug">→ Suggestion</div><div class="fb-text">{f.get('suggestion','—')}</div></div>
            {f'<div class="fb-sec"><div class="fb-lbl" style="color:rgba(0,200,150,0.65)">⭐ STAR Analysis</div><div class="fb-text">{f.get("star_feedback","")}</div></div>' if f.get("star_feedback") and q_type in ("behavioral","situational") else ''}
        </div>
        """, unsafe_allow_html=True)

        ideal = f.get("ideal_hint","")
        if ideal:
            with st.expander("💡 What a strong answer would have included"):
                st.markdown(f'<div class="tip neon">{ideal}</div>', unsafe_allow_html=True)

        if qa_local:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Words",   f'{qa_local.get("wc",0)}')
            c2.metric("Fillers", f'{qa_local.get("filler_count",0)}')
            c3.metric("STAR",    f'{qa_local.get("star_score",0)}/4')
            c4.metric("Specificity", f'{qa_local.get("specificity",0)}/3')

        pending = st.session_state.get("_pending_followup", False)
        fq      = f.get("followup_question","")
        if pending and fq:
            st.markdown(f'<div class="tip" style="color:rgba(251,100,100,0.65);border-color:rgba(251,44,54,0.2);margin-top:0.5rem">🔍 {persona["name"]} wants to explore this further…</div>', unsafe_allow_html=True)
            fc1, fc2 = st.columns(2)
            with fc1:
                if st.button("🔄 Answer Follow-up", use_container_width=True):
                    st.session_state.questions.insert(idx+1, fq)
                    st.session_state.q_types.insert(idx+1, q_type)
                    st.session_state.q_competencies.insert(idx+1, competency)
                    st.session_state.q_difficulties.insert(idx+1, "hard")
                    st.session_state.current      += 1
                    st.session_state.submitted     = False
                    st.session_state.is_followup   = True
                    st.session_state.followup_count += 1
                    st.session_state._pending_followup = False
                    st.session_state.ketu_message  = f"Good. Let me push on this: {fq}"
                    st.session_state.q_start = time.time()
                    st.rerun()
            with fc2:
                if st.button("Skip Follow-up →", use_container_width=True):
                    st.session_state.current   += 1
                    st.session_state.submitted  = False
                    st.session_state.is_followup = False
                    st.session_state._pending_followup = False
                    st.session_state.ketu_message = random.choice(persona["transitions"])
                    st.session_state.q_start = time.time()
                    st.rerun()
        else:
            label = "Finish Interview →" if idx+1 >= len(questions) else f"Next Question → Q{idx+2}"
            if st.button(label, use_container_width=True):
                st.session_state.current   += 1
                st.session_state.submitted  = False
                st.session_state.is_followup = False
                st.session_state._pending_followup = False
                st.session_state.ketu_message = random.choice(persona["transitions"])
                st.session_state.q_start = time.time()
                st.rerun()


# ─────────────────────────────────────────────────────────────
# SCREEN — RESULTS
# ─────────────────────────────────────────────────────────────
def screen_results():
    llm           = get_llm()
    scores        = st.session_state.scores
    feedback_list = st.session_state.feedback_list
    persona       = PERSONAS.get(st.session_state.persona, PERSONAS["Ketu"])

    if not scores:
        st.warning("No answers were recorded.")
        if st.button("Start Over"):
            for k in list(st.session_state.keys()): del st.session_state[k]
            st.rerun()
        return

    avg     = sum(s.get("score",0) for s in scores) / len(scores)
    grade   = grade_letter(avg)
    g_cls   = grade_css(grade)
    tagline = grade_tagline(grade)
    name    = st.session_state.candidate_name or "Candidate"
    role    = st.session_state.role_title
    elapsed = int(time.time() - (st.session_state.session_start or time.time()))
    mins    = elapsed // 60
    n_total = len(st.session_state.questions)
    mode    = st.session_state.interview_mode
    total_words   = sum(st.session_state.word_counts) if st.session_state.word_counts else 0
    total_fillers = sum(st.session_state.filler_counts) if st.session_state.filler_counts else 0
    avg_words     = total_words // max(len(st.session_state.word_counts), 1)
    star_scores   = [item.get("qa",{}).get("star_score",0) for item in feedback_list if item.get("qa")]
    avg_star      = sum(star_scores) / max(len(star_scores),1)
    filler_pct    = (total_fillers / max(total_words,1)) * 100

    st.markdown(f"""
    <div class="result-hero">
        <div class="hero-kicker"><div class="hero-dot"></div>{name} · {role} · {mode} Mode · {persona['name']}</div>
        <div class="result-grade {g_cls}">{grade}</div>
        <div class="result-score-row">Final Score · {avg:.1f} / 10</div>
        <div class="result-tagline">{tagline}</div>
    </div>
    """, unsafe_allow_html=True)

    m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
    m1.metric("Score",   f"{avg:.1f}/10")
    m2.metric("Answered",f"{len(scores)}/{n_total}")
    m3.metric("Duration",f"{mins}m")
    m4.metric("Avg Words",f"{avg_words}")
    m5.metric("Fillers", f"{total_fillers}")
    m6.metric("STAR Avg",f"{avg_star:.1f}/4")
    m7.metric("Filler %",f"{filler_pct:.1f}%")

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Analytics", "📋 Breakdown", "🤖 AI Assessment", "📄 Resume Profile", "⬇️ Export"])

    with tab1:
        col_l, col_r = st.columns([1.2, 0.8], gap="large")
        with col_l:
            if len(scores) >= 2:
                st.markdown('<div class="sec">📈 Score Timeline</div>', unsafe_allow_html=True)
                vals = [s.get("score",0) for s in scores]
                ql   = [f"Q{i+1}" for i in range(len(scores))]
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=ql, y=vals, mode="lines+markers",
                    line=dict(color="#00d4ff", width=2.5, shape="spline"),
                    marker=dict(size=10, color=vals, colorscale=[[0,"#fb2c36"],[0.5,"#fbbf24"],[1,"#00c896"]],
                        line=dict(color="#02040a",width=2.5)),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.04)",
                ))
                fig_line.add_hline(y=avg, line_dash="dot", line_color="rgba(0,212,255,0.35)",
                    annotation_text=f"avg {avg:.1f}", annotation_font_color="#00d4ff", annotation_font_size=10)
                fig_line.update_layout(**{**PLOTLY,"height":240,"showlegend":False,"yaxis":{**PLOTLY.get("yaxis",{}),"range":[0,10.5]}})
                st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar":False})

            if st.session_state.word_counts and len(st.session_state.word_counts) >= 2:
                st.markdown('<div class="sec">📝 Words per Answer</div>', unsafe_allow_html=True)
                wc_v = st.session_state.word_counts
                wc_l = [f"Q{i+1}" for i in range(len(wc_v))]
                fig_wc = go.Figure(go.Bar(
                    x=wc_l, y=wc_v,
                    marker_color=["#00c896" if 80<=w<=250 else "#fbbf24" if w<80 else "#fb2c36" for w in wc_v],
                    marker_line_width=0, text=wc_v, textposition="outside",
                    textfont=dict(size=10, color="#3d5580"),
                ))
                fig_wc.add_hline(y=80,  line_dash="dot", line_color="rgba(0,200,150,0.3)", annotation_text="min", annotation_font_size=9, annotation_font_color="rgba(0,200,150,0.5)")
                fig_wc.add_hline(y=250, line_dash="dot", line_color="rgba(251,44,54,0.3)",  annotation_text="max", annotation_font_size=9, annotation_font_color="rgba(251,44,54,0.5)")
                fig_wc.update_layout(**PLOTLY, height=180, showlegend=False)
                st.plotly_chart(fig_wc, use_container_width=True, config={"displayModeBar":False})

            comp_agg = {k: sum(v)/len(v) for k,v in st.session_state.competency_scores.items() if v}
            if comp_agg:
                st.markdown('<div class="sec">🏆 Competency Scores</div>', unsafe_allow_html=True)
                sorted_comps = sorted(comp_agg.items(), key=lambda x: x[1])
                fig_comp = go.Figure(go.Bar(
                    x=[v for _,v in sorted_comps],
                    y=[k for k,_ in sorted_comps],
                    orientation="h",
                    marker_color=["#00c896" if v>=7 else "#fbbf24" if v>=5 else "#fb2c36" for _,v in sorted_comps],
                    marker_line_width=0,
                    text=[f"{v:.1f}" for _,v in sorted_comps],
                    textposition="outside",
                    textfont=dict(size=10, color="#8a9fc4"),
                ))
                fig_comp.update_layout(height=max(180, len(sorted_comps) * 36),showlegend=False,xaxis=dict(range=[0, 11]))
                st.plotly_chart(fig_comp, use_container_width=True, config={"displayModeBar":False})

        with col_r:
            if len(comp_agg) >= 3:
                st.markdown('<div class="sec">🕸️ Competency Radar</div>', unsafe_allow_html=True)
                cats_r = list(comp_agg.keys()); vals_r = list(comp_agg.values())
                fig_rad = go.Figure(go.Scatterpolar(
                    r=vals_r+[vals_r[0]], theta=cats_r+[cats_r[0]],
                    fill="toself", fillcolor="rgba(0,212,255,0.05)",
                    line=dict(color="#00d4ff",width=2), marker=dict(color="#00d4ff",size=6),
                ))
                fig_rad.update_layout(**{**PLOTLY,"polar":dict(
                    bgcolor="rgba(0,0,0,0)",
                    angularaxis=dict(color="#1e3258",gridcolor="#0e1a2e",tickfont=dict(size=9,family="Geist Mono")),
                    radialaxis=dict(range=[0,10],color="#1e3258",gridcolor="#0e1a2e"),
                ),"height":300})
                st.plotly_chart(fig_rad, use_container_width=True, config={"displayModeBar":False})

            st.markdown('<div class="sec">📊 Score Distribution</div>', unsafe_allow_html=True)
            bins = {"0-4":0,"5-6":0,"7-8":0,"9-10":0}
            for s in scores:
                v = s.get("score",0)
                if v<=4: bins["0-4"]+=1
                elif v<=6: bins["5-6"]+=1
                elif v<=8: bins["7-8"]+=1
                else: bins["9-10"]+=1
            fig_dist = go.Figure(go.Bar(
                x=list(bins.keys()), y=list(bins.values()),
                marker_color=["#fb2c36","#fbbf24","#00d4ff","#00c896"],
                marker_line_width=0, text=list(bins.values()),
                textposition="outside", textfont=dict(size=11, color="#3d5580"),
            ))
            fig_dist.update_layout(**PLOTLY, height=200, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar":False})

            type_counts = Counter(item.get("type","technical") for item in feedback_list)
            if type_counts:
                st.markdown('<div class="sec">🏷️ Question Type Distribution</div>', unsafe_allow_html=True)
                type_colors = {"technical":"#7c3aed","behavioral":"#00d4ff","situational":"#fbbf24","rapport":"#00c896","ambition":"#fb923c"}
                fig_type = go.Figure(go.Pie(
                    labels=list(type_counts.keys()),
                    values=list(type_counts.values()),
                    hole=0.6,
                    marker_colors=[type_colors.get(k,"#94a3b8") for k in type_counts.keys()],
                    textfont=dict(family="Geist Mono",size=9),
                ))
                fig_type.update_layout(**{**PLOTLY,"height":200,"showlegend":True,
                    "legend":dict(font=dict(family="Geist Mono",size=9,color="#3d5580"),bgcolor="rgba(0,0,0,0)")})
                st.plotly_chart(fig_type, use_container_width=True, config={"displayModeBar":False})

            st.markdown('<div class="sec">🎙️ Communication Quality</div>', unsafe_allow_html=True)
            fp_color = "#00c896" if filler_pct<3 else "#fbbf24" if filler_pct<6 else "#fb2c36"
            fp_label = "Excellent" if filler_pct<3 else "Acceptable" if filler_pct<6 else "Needs work"
            st.markdown(f"""
            <div class="glass" style="padding:1.2rem">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem">
                    <div style="font-family:'Geist Mono',monospace;font-size:0.62rem;color:var(--t4)">FILLER DENSITY</div>
                    <div style="font-family:'Geist',sans-serif;font-weight:700;font-size:0.9rem;color:{fp_color}">{filler_pct:.1f}% · {fp_label}</div>
                </div>
                <div style="height:3px;background:var(--b2);border-radius:99px;overflow:hidden">
                    <div style="height:100%;width:{min(filler_pct/10*100,100):.0f}%;background:{fp_color};border-radius:99px"></div>
                </div>
                <div style="font-family:'Geist Mono',monospace;font-size:0.62rem;color:var(--t4);margin-top:0.5rem">
                    {total_fillers} filler words · {total_words} total words · avg {avg_words} per answer
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass" style="padding:1.5rem">', unsafe_allow_html=True)
        for i, item in enumerate(feedback_list):
            sc = item["eval"].get("score",0)
            sc_c = score_color(sc)
            verdict = item["eval"].get("verdict","—")
            qt      = item.get("type","technical")
            qinfo   = QUESTION_TYPES.get(qt,("❓","qb-tech",qt.title()))
            comp    = item.get("competency","—")
            diff    = item.get("difficulty","medium")
            diff_c  = {"easy":"diff-e","medium":"diff-m","hard":"diff-h"}.get(diff,"diff-m")
            wc      = item.get("qa",{}).get("wc",0)
            fc      = item.get("qa",{}).get("filler_count",0)
            star    = item.get("qa",{}).get("star_score",0)
            t_secs  = item.get("time",0)

            st.markdown(f"""
            <div class="qbt-item">
                <div class="qbt-num">Q{i+1}</div>
                <div class="qbt-body">
                    <div class="qbt-q">{item['q'][:95]}{'…' if len(item['q'])>95 else ''}</div>
                    <div class="qbt-tags">
                        <span class="score-pill" style="color:{sc_c};border-color:{sc_c}40;background:rgba(0,0,0,0.2)">{sc:.1f}/10 · {verdict}</span>
                        <span class="q-badge {qinfo[1]}">{qinfo[0]} {qinfo[2]}</span>
                        <span class="q-comp-tag">{comp}</span>
                        <span class="q-diff {diff_c}">{diff.upper()}</span>
                        {'<span style="font-family:\'Geist Mono\',monospace;font-size:0.6rem;color:var(--t4)">⏱ '+str(t_secs)+'s · '+str(wc)+'w · '+str(fc)+' fillers · ⭐'+str(star)+'/4</span>' if wc else ''}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"Full feedback — Q{i+1}: {item['q'][:55]}…"):
                st.markdown(f"**Answer:** {item['a']}")
                tones = item["eval"].get("tone_signals",[])
                if tones:
                    chips = "".join([f'<span class="tone-chip {"tc-pos" if t in POSITIVE_TONE else "tc-neg" if t in NEGATIVE_TONE else "tc-neu"}">{t}</span>' for t in tones])
                    st.markdown(f'<div class="tone-chips">{chips}</div>', unsafe_allow_html=True)
                ca, cb = st.columns(2)
                with ca:
                    st.success(f"**Strength:** {item['eval'].get('strength','—')}")
                    st.info(f"**Suggestion:** {item['eval'].get('suggestion','—')}")
                with cb:
                    st.error(f"**Gap:** {item['eval'].get('weakness','—')}")
                    ideal = item["eval"].get("ideal_hint","")
                    if ideal:
                        st.markdown(f'<div class="tip neon">💡 {ideal}</div>', unsafe_allow_html=True)
                qa_d = item.get("qa",{})
                if qa_d.get("star") and item.get("type") in ("behavioral","situational"):
                    star_cells = "".join([
                        f'<div class="star-cell {"active" if v else ""}"><div class="star-label">{k}</div><div class="star-val {"star-y" if v else "star-n"}">{"✓" if v else "○"}</div></div>'
                        for k,v in qa_d["star"].items()
                    ])
                    st.markdown(f'<div style="max-width:300px;margin-top:0.5rem"><div class="star-grid">{star_cells}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        if not st.session_state.ai_summary:
            with st.spinner(f"✍️ {persona['name']} is writing your assessment…"):
                st.session_state.ai_summary = gen_summary(
                    feedback_list, role, name, avg, st.session_state.persona, mode, llm
                )
        st.markdown(f"""
        <div class="glass glass-neon">
            <div class="sec sec-neon">{persona['avatar']} {persona['name']}'s Assessment</div>
            <div style="font-family:'Instrument Serif',serif;font-style:italic;font-size:0.97rem;color:var(--t2);line-height:1.85">
                {st.session_state.ai_summary.replace(chr(10),'<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        profile = st.session_state.get("resume_profile")
        if profile:
            fit = profile.get("overall_fit_score",0)
            fc_ = score_color(fit)
            st.markdown(f"""
            <div class="glass glass-electric">
                <div class="sec sec-electric">📄 Resume Intelligence</div>
                <div class="rp-name">{profile.get('candidate_name','Candidate')}</div>
                <div class="rp-role">{profile.get('current_role','Professional')} · {profile.get('years_experience','N/A')}</div>
                <div style="margin:0.6rem 0;font-family:'Geist Mono',monospace;font-size:0.7rem;color:var(--t3)">Education: {profile.get('education','N/A')}</div>
                <div style="font-family:'Geist Mono',monospace;font-size:0.7rem;color:var(--t3);margin-bottom:0.8rem">Companies: {' · '.join(profile.get('companies',[])[:4]) or 'N/A'}</div>
            """, unsafe_allow_html=True)

            c_r, c_l = st.columns(2)
            with c_r:
                st.markdown(f'<div style="font-family:Geist Mono,monospace;font-size:0.62rem;color:var(--t4);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem">Role Fit Score</div><div style="font-family:Geist,sans-serif;font-weight:900;font-size:2.5rem;color:{fc_}">{fit:.1f}/10</div><div style="font-family:Geist Mono,monospace;font-size:0.7rem;color:var(--t3);margin-top:0.3rem">{profile.get("fit_rationale","")}</div>', unsafe_allow_html=True)
            with c_l:
                if profile.get("strengths"):
                    st.markdown('<div style="font-family:Geist Mono,monospace;font-size:0.62rem;color:rgba(0,200,150,0.6);letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.4rem">Top Strengths</div>', unsafe_allow_html=True)
                    for s in profile["strengths"][:3]:
                        st.markdown(f'<div style="font-family:Geist,sans-serif;font-size:0.82rem;color:var(--t2);padding:0.2rem 0">✓ {s}</div>', unsafe_allow_html=True)

            if profile.get("matching_skills"):
                st.markdown('<div style="font-family:Geist Mono,monospace;font-size:0.62rem;color:rgba(0,200,150,0.6);letter-spacing:0.14em;text-transform:uppercase;margin:1rem 0 0.4rem">Skill Matches</div>', unsafe_allow_html=True)
                st.markdown('<div class="skills-match">' + "".join([f'<span class="skill-tag sk-match">✓ {s}</span>' for s in profile["matching_skills"]]) + '</div>', unsafe_allow_html=True)

            if profile.get("gap_skills"):
                st.markdown('<div style="font-family:Geist Mono,monospace;font-size:0.62rem;color:rgba(251,44,54,0.6);letter-spacing:0.14em;text-transform:uppercase;margin:0.8rem 0 0.4rem">Skill Gaps</div>', unsafe_allow_html=True)
                st.markdown('<div class="skills-match">' + "".join([f'<span class="skill-tag sk-gap">✗ {s}</span>' for s in profile["gap_skills"]]) + '</div>', unsafe_allow_html=True)

            if profile.get("red_flags"):
                st.markdown("<br>", unsafe_allow_html=True)
                st.warning("⚠️ Potential concerns: " + " · ".join(profile["red_flags"]))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Resume profile analysis was not run (JD not pasted before upload, or LLM call failed).")

    with tab5:
        st.markdown('<div class="sec">⬇️ Download Your Report</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.download_button("📦 JSON Report", data=build_json(st.session_state),
                file_name=f"ketu_v2_1_{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json", use_container_width=True)
        with c2:
            st.download_button("📊 CSV Export", data=build_csv(st.session_state),
                file_name=f"ketu_v2_1_{name.replace(' ','_')}.csv",
                mime="text/csv", use_container_width=True)
        with c3:
            if st.button("🔄 New Interview", use_container_width=True):
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.rerun()
        with c4:
            if st.button("📋 Same Role", use_container_width=True):
                r,j,t,c_ = st.session_state.resume_text, st.session_state.jd_text, st.session_state.role_title, st.session_state.category_tag
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.session_state.resume_text=r; st.session_state.jd_text=j; st.session_state.role_title=t; st.session_state.category_tag=c_
                st.rerun()
        with c5:
            if st.button("🔥 Intense Mode", use_container_width=True):
                r,j,t,c_ = st.session_state.resume_text, st.session_state.jd_text, st.session_state.role_title, st.session_state.category_tag
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.session_state.resume_text=r; st.session_state.jd_text=j; st.session_state.role_title=t; st.session_state.category_tag=c_
                st.session_state.interview_mode="Intense"
                st.rerun()

        st.markdown("""
        <div class="export-block" style="margin-top:1rem">
            <div class="export-icon">📦</div>
            <div>
                <div class="export-title">Full Interview Report (JSON)</div>
                <div class="export-desc">Includes metadata, per-question scores, AI assessment, competency breakdown, STAR analysis, communication stats, resume profile, and camera analytics.</div>
            </div>
        </div>
        <div class="export-block" style="margin-top:0.5rem">
            <div class="export-icon">📊</div>
            <div>
                <div class="export-title">Tabular Export (CSV)</div>
                <div class="export-desc">Question-by-question breakdown — perfect for tracking progress across sessions.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
render_sidebar()

screen = st.session_state.screen
if screen == "setup":
    screen_setup()
elif screen == "interview":
    screen_interview()
elif screen == "results":
    screen_results()
