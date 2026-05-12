# ============================================================
# KETU AI v3.0 — VIVID EDITION
# Redesigned: Vibrant aurora palette, fluid animations,
# human-warmth interview flow, immersive typing effects,
# live reaction system, emotional presence indicators
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

import re, io, time, base64, tempfile, json, math, random
from io import BytesIO
from datetime import datetime
from collections import Counter
from functools import lru_cache

import streamlit as st
try:
    st.set_option("server.fileWatcherType", "none")
except Exception:
    pass

import pandas as pd
import plotly.graph_objects as go
from gtts import gTTS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

try:
    from audio_recorder_streamlit import audio_recorder
    HAS_AUDIO_RECORDER = True
except ImportError:
    HAS_AUDIO_RECORDER = False

st.set_page_config(
    page_title="KETU AI · Vivid Edition",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Pre-compiled regexes ─────────────────────────────────────
_RE_SITUATION = re.compile(r'\b(when|once|at my|in my|during|while working|we were|i was|back at|last year|previously)\b', re.I)
_RE_TASK      = re.compile(r'\b(had to|needed to|responsible for|my role|tasked with|goal was|objective|challenge was)\b', re.I)
_RE_ACTION    = re.compile(r'\b(i did|i built|i wrote|i led|i implemented|i created|i designed|i worked|i developed|i resolved|i decided|i introduced|i proposed|i initiated)\b', re.I)
_RE_RESULT    = re.compile(r'\b(result|outcome|achieved|improved|reduced|increased|saved|delivered|shipped|launched|percent|%|metric|measur|impact|generated|won|closed)\b', re.I)
_RE_NUMBERS   = re.compile(r'\b\d+\b')
_RE_PERCENT   = re.compile(r'\d+\s*%')
_RE_TIME      = re.compile(r'\b(week|month|quarter|year|day|sprint|cycle)\b', re.I)
_RE_JSON_FENCE = re.compile(r'```(?:json)?')

FILLER_WORDS = frozenset({"um","uh","like","you know","basically","literally","actually","sort of","kind of","i mean","right","so yeah","honestly","obviously","clearly"})
POSITIVE_TONE = frozenset({"Confident","Structured","Concise","Detailed","Passionate","Analytical","Creative","Experienced","Thoughtful","Authentic","Polished"})
NEGATIVE_TONE = frozenset({"Vague","Nervous","Hesitant","Rambling","Unprepared"})

ROLE_BENCHMARKS = {
    "Engineering":    {"p50": 5.8, "p75": 7.2, "label": "Software Engineer"},
    "Management":     {"p50": 6.0, "p75": 7.5, "label": "Manager"},
    "Product":        {"p50": 5.9, "p75": 7.3, "label": "Product Manager"},
    "Design":         {"p50": 5.7, "p75": 7.1, "label": "Designer"},
    "Sales/BD":       {"p50": 6.1, "p75": 7.4, "label": "Sales / BD"},
    "Data/Analytics": {"p50": 5.8, "p75": 7.2, "label": "Data Analyst"},
    "Marketing":      {"p50": 5.6, "p75": 7.0, "label": "Marketer"},
    "Operations":     {"p50": 5.7, "p75": 7.1, "label": "Ops"},
}

STAR_TEMPLATES = {
    "behavioral": "Situation: [Describe the context — when, where, what team/project]...\n\nTask: [What was your specific responsibility or challenge]...\n\nAction: [What YOU personally did — steps, tools, decisions]...\n\nResult: [Measurable outcome — %, time saved, revenue, team impact]...",
    "situational": "How I'd approach this: [Frame the problem / constraints first]...\n\nMy immediate steps: [List 2-3 concrete actions with reasoning]...\n\nStakeholders I'd involve: [Who and why]...\n\nSuccess metric: [How I'd measure the outcome]...",
    "technical": "My understanding of the problem: [Technical framing]...\n\nApproach / architecture: [Key decisions and trade-offs]...\n\nImplementation details: [Tools, methods, edge cases]...\n\nOutcome & lessons: [What shipped, what I'd change]...",
    "rapport": "Background: [Briefly who you are and your journey]...\n\nWhat drives me: [Your core motivation or passion]...\n\nA key moment: [One experience that shaped your career]...\n\nWhy this role: [Specific, genuine connection to the opportunity]...",
    "ambition": "Where I want to be in 2-3 years: [Specific, role-relevant goal]...\n\nSkills I'm actively building: [With examples of how]...\n\nWhy this company/role: [Genuine alignment, not flattery]...\n\nWhat success looks like for me: [Concrete, measurable vision]...",
}

LANGUAGES = {"English":"en","Hindi":"hi","Spanish":"es","French":"fr","German":"de","Portuguese":"pt","Arabic":"ar","Japanese":"ja","Chinese":"zh"}

_POSITIVE_SIGNALS = frozenset({"achieved","built","led","improved","grew","delivered","launched","proud","excited","loved","succeeded","won","increased","reduced","excellent","strong","passionate","thrilled","great"})
_NEGATIVE_SIGNALS = frozenset({"failed","struggled","difficult","challenging","mistake","wrong","problem","issue","conflict","missed","lost","dropped","unfortunately","never","couldn't","didn't","hard","tough","frustrated"})

# ============================================================
# PERSONAS — richer human warmth
# ============================================================
PERSONAS = {
    "Ketu": {
        "name": "Ketu", "title": "Senior AI Interviewer", "avatar": "✦",
        "color": "#6366f1", "glow": "rgba(99,102,241,0.25)", "accent": "#818cf8",
        "style": "balanced, warm, and genuinely curious",
        "mood": "curious",
        "greetings": [
            "Great to meet you! I've carefully reviewed your profile and I'm genuinely excited to learn about your journey. Let's make this a real conversation.",
            "Welcome! I've gone through your background thoroughly. Take your time — I'm here to understand you, not catch you out.",
            "Hello! I've prepared some tailored questions based on your experience. No tricks — just an honest conversation.",
        ],
        "reactions": {
            "strong": ["That's a really strong answer. I can see exactly why you're proud of that.", "Excellent — that's the kind of specificity I love to hear."],
            "average": ["Thanks for that. There's something interesting in what you said there.", "Good start — let's dig a little deeper on this."],
            "weak": ["I appreciate your honesty. Let's think about this from a different angle.", "Thanks for sharing that. These situations are always complex."],
        },
        "transitions": ["Interesting. Let me move to the next area.", "I appreciate your openness. Let's continue.", "That's helpful context. Here's my next question.", "Good. Moving on…"],
        "thinking": ["Reviewing your answer…", "Analysing your response…", "Processing what you said…"],
    },
    "Aria": {
        "name": "Aria", "title": "Technical Director", "avatar": "◈",
        "color": "#06b6d4", "glow": "rgba(6,182,212,0.25)", "accent": "#22d3ee",
        "style": "technical, precise, and analytically rigorous",
        "mood": "focused",
        "greetings": [
            "I'll be direct: I want to understand how you think, not what you've memorised. Let's begin.",
            "I've reviewed your technical background. I'll be asking you to go deep — that's how I learn what you're truly capable of.",
        ],
        "reactions": {
            "strong": ["Strong technical reasoning. I can see you've dealt with this at scale.", "Exactly — the trade-off analysis you described is mature engineering thinking."],
            "average": ["Noted. Let me push a little further on the implementation side.", "Reasonable approach. What would you have done differently with hindsight?"],
            "weak": ["That's a starting point. Let's pressure-test the architecture a bit.", "Fair attempt — let's go deeper on the failure modes here."],
        },
        "transitions": ["Noted. Let's test another dimension.", "That's an interesting angle. Moving on.", "Understood. Next question."],
        "thinking": ["Running technical analysis…", "Evaluating depth of answer…", "Checking technical precision…"],
    },
    "Marcus": {
        "name": "Marcus", "title": "Culture & Values Lead", "avatar": "❋",
        "color": "#10b981", "glow": "rgba(16,185,129,0.25)", "accent": "#34d399",
        "style": "empathetic, culture-focused, and values-driven",
        "mood": "warm",
        "greetings": [
            "I want this to feel like a real conversation — not an interrogation. I'm genuinely curious about who you are.",
            "Welcome! I focus on values and the human side of work. Let's explore that together today.",
        ],
        "reactions": {
            "strong": ["That takes real self-awareness to articulate. I genuinely respect that.", "That story tells me a lot about who you are. Thank you for sharing."],
            "average": ["I sense there's more to that story. What was going through your mind at the time?", "That's a good foundation — what did you learn about yourself from it?"],
            "weak": ["These situations are never easy. What would you have done if you could go back?", "Thank you for being honest. How do you feel about that experience now?"],
        },
        "transitions": ["I really appreciate your honesty there. Let's explore another dimension.", "That's genuinely telling — thank you. Let me ask about something different.", "Good. Let's keep going."],
        "thinking": ["Reflecting on your answer…", "Considering cultural fit…", "Reading between the lines…"],
    },
    "Nova": {
        "name": "Nova", "title": "Product Strategy Lead", "avatar": "⬡",
        "color": "#f59e0b", "glow": "rgba(245,158,11,0.25)", "accent": "#fcd34d",
        "style": "product-minded, strategic, and challenge-seeking",
        "mood": "energetic",
        "greetings": [
            "I'm going to push you to think like a product thinker, not just a practitioner. Ready?",
            "Hi! I care about how you frame problems and measure success. Let's think out loud together.",
        ],
        "reactions": {
            "strong": ["That's exactly the kind of user-centric thinking I look for. Sharp.", "Love the framework — you moved from insight to metric to decision really cleanly."],
            "average": ["Good instinct. How would you validate that assumption before building?", "Interesting framing. What's the riskiest assumption in that approach?"],
            "weak": ["Let me challenge you — what if the user doesn't behave the way you're expecting?", "Worth pressure-testing that. How would you measure success here?"],
        },
        "transitions": ["That's a good instinct. Let me challenge you further.", "Noted. Let's dig into product thinking next.", "Interesting framing. Moving on."],
        "thinking": ["Evaluating strategic thinking…", "Assessing product instincts…", "Analysing your reasoning…"],
    },
}

QUESTION_TYPES = {
    "rapport":     ("💬", "badge-rapport",    "Rapport"),
    "technical":   ("⚙",  "badge-technical",  "Technical"),
    "behavioral":  ("◎",  "badge-behavioral", "Behavioral"),
    "situational": ("◈",  "badge-situational","Situational"),
    "ambition":    ("⬆",  "badge-ambition",   "Forward-looking"),
}

INTERVIEW_MODES = {
    "Casual":   {"pressure": "low",  "followup_threshold": 4.0, "max_followups": 1, "emoji": "🌿", "desc": "Relaxed pace, supportive tone", "color": "#10b981"},
    "Standard": {"pressure": "med",  "followup_threshold": 5.5, "max_followups": 2, "emoji": "⚡", "desc": "Professional, balanced assessment", "color": "#6366f1"},
    "Intense":  {"pressure": "high", "followup_threshold": 7.0, "max_followups": 3, "emoji": "🔥", "desc": "High-pressure, rigorous evaluation", "color": "#ef4444"},
}

COMPETENCY_FRAMEWORKS = {
    "Engineering":    ["Problem Solving","Technical Depth","System Design","Code Quality","Collaboration","Communication","Growth Mindset"],
    "Management":     ["Leadership","Strategic Thinking","Communication","Conflict Resolution","Decision Making","Mentoring","Execution"],
    "Product":        ["Product Sense","Data Analysis","User Empathy","Prioritization","Communication","Execution","Strategy"],
    "Design":         ["Visual Thinking","User Research","Communication","Craft","Iteration","Business Acumen","Storytelling"],
    "Sales/BD":       ["Persuasion","Relationship Building","Resilience","Product Knowledge","Communication","Pipeline Management","Closing"],
    "Data/Analytics": ["Statistical Thinking","SQL/Tooling","Communication","Problem Framing","Visualization","Business Acumen","Experimentation"],
    "Marketing":      ["Brand Thinking","Data Analysis","Creativity","Communication","Campaign Strategy","Growth Mindset","Audience Insight"],
    "Operations":     ["Process Design","Problem Solving","Stakeholder Management","Execution","Data-Driven Decision Making","Resilience","Communication"],
}

# ============================================================
# CSS — Vivid Edition Design System
# ============================================================
_CSS_VERSION = "3.0.0"

DESIGN = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400;1,500&family=DM+Mono:wght@300;400;500&family=Playfair+Display:ital,wght@0,700;1,600&display=swap');

:root {
  /* Core palette — deep aurora */
  --bg: #080a12;
  --bg2: #0c0f1a;
  --bg3: #111525;
  --surface: #141828;
  --surface2: #191e30;
  --surface3: #1e253a;
  --surface4: #242c44;
  --border: rgba(255,255,255,0.06);
  --border2: rgba(255,255,255,0.10);
  --border3: rgba(255,255,255,0.15);

  /* Vivid accent palette */
  --violet: #7c3aed;
  --violet-l: #a78bfa;
  --violet-d: #4c1d95;
  --indigo: #6366f1;
  --indigo-l: #818cf8;
  --cyan: #06b6d4;
  --cyan-l: #67e8f9;
  --emerald: #10b981;
  --emerald-l: #34d399;
  --amber: #f59e0b;
  --amber-l: #fcd34d;
  --rose: #f43f5e;
  --rose-l: #fb7185;
  --pink: #ec4899;
  --orange: #f97316;

  /* Text */
  --t1: #f1f5ff;
  --t2: #94a3c8;
  --t3: #4b5980;
  --t4: #2a3355;

  /* Glow effects */
  --glow-v: 0 0 40px rgba(124,58,237,0.15), 0 0 80px rgba(124,58,237,0.08);
  --glow-i: 0 0 40px rgba(99,102,241,0.15), 0 0 80px rgba(99,102,241,0.08);
  --glow-c: 0 0 40px rgba(6,182,212,0.12);
  --glow-e: 0 0 40px rgba(16,185,129,0.12);

  /* Radii */
  --r1: 6px; --r2: 10px; --r3: 14px; --r4: 20px; --r5: 28px;
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
  background: var(--bg) !important;
  color: var(--t1) !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* Animated mesh background */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 80% 60% at -10% -20%, rgba(124,58,237,0.08) 0%, transparent 50%),
    radial-gradient(ellipse 60% 50% at 110% 120%, rgba(6,182,212,0.06) 0%, transparent 50%),
    radial-gradient(ellipse 40% 40% at 50% 50%, rgba(99,102,241,0.04) 0%, transparent 60%),
    var(--bg) !important;
}

[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0;
  pointer-events: none; z-index: 0;
  background:
    repeating-linear-gradient(0deg, transparent, transparent 59px, rgba(255,255,255,0.015) 60px),
    repeating-linear-gradient(90deg, transparent, transparent 59px, rgba(255,255,255,0.015) 60px);
  background-size: 60px 60px;
}

[data-testid="stHeader"], footer, #MainMenu { display: none !important; }
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif !important; }

h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; font-weight: 700 !important; }

::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface4); border-radius: 99px; }

/* ── Buttons ── */
.stButton > button {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  color: var(--t2) !important;
  border-radius: var(--r2) !important;
  font-family: 'DM Mono', monospace !important;
  font-weight: 500 !important;
  font-size: .78rem !important;
  padding: .55rem 1.3rem !important;
  letter-spacing: .03em !important;
  transition: all .18s ease !important;
}
.stButton > button:hover {
  border-color: var(--indigo-l) !important;
  color: var(--indigo-l) !important;
  background: rgba(99,102,241,0.08) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
  transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stTextArea textarea, .stTextInput input {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r3) !important;
  color: var(--t1) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .9rem !important;
  line-height: 1.7 !important;
  transition: border-color .2s, box-shadow .2s !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
  border-color: rgba(124,58,237,0.5) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.1), 0 0 20px rgba(124,58,237,0.06) !important;
  outline: none !important;
}
.stTextArea label, .stTextInput label {
  color: var(--t3) !important;
  font-family: 'DM Mono', monospace !important;
  font-size: .7rem !important;
  letter-spacing: .06em !important;
  text-transform: uppercase !important;
}

/* ── Selects ── */
.stSelectbox > div > div, .stMultiSelect > div > div {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  border-radius: var(--r2) !important;
  color: var(--t1) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: .85rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--surface2) !important;
  border: 2px dashed var(--border2) !important;
  border-radius: var(--r4) !important;
  transition: all .2s !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(124,58,237,0.4) !important;
  background: rgba(124,58,237,0.04) !important;
}

/* ── Progress ── */
.stProgress > div > div { background: var(--surface3) !important; border-radius: 99px !important; height: 3px !important; }
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--violet), var(--indigo), var(--cyan)) !important;
  border-radius: 99px !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r3) !important;
  padding: 1rem 1.2rem !important;
  position: relative !important;
  overflow: hidden !important;
}
[data-testid="stMetric"]::after {
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 2px; background: linear-gradient(90deg, var(--violet), var(--cyan));
}
[data-testid="stMetricValue"] { font-family: 'DM Sans', sans-serif !important; font-size: 1.65rem !important; font-weight: 700 !important; color: var(--t1) !important; }
[data-testid="stMetricLabel"] { font-family: 'DM Mono', monospace !important; font-size: .6rem !important; color: var(--t3) !important; letter-spacing: .12em !important; text-transform: uppercase !important; }

/* ── Status bars ── */
.stSuccess { background: rgba(16,185,129,0.06) !important; border: 1px solid rgba(16,185,129,0.2) !important; border-radius: var(--r2) !important; }
.stError   { background: rgba(244,63,94,0.06)  !important; border: 1px solid rgba(244,63,94,0.2)  !important; border-radius: var(--r2) !important; }
.stWarning { background: rgba(245,158,11,0.06) !important; border: 1px solid rgba(245,158,11,0.2) !important; border-radius: var(--r2) !important; }
.stInfo    { background: rgba(99,102,241,0.06) !important; border: 1px solid rgba(99,102,241,0.2) !important; border-radius: var(--r2) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: .2rem; background: var(--surface2) !important; border-radius: var(--r2) !important; padding: 4px !important; border: 1px solid var(--border) !important; }
.stTabs [data-baseweb="tab"] { border-radius: var(--r1) !important; font-family: 'DM Mono', monospace !important; font-size: .72rem !important; color: var(--t3) !important; transition: all .2s !important; padding: .4rem 1rem !important; }
.stTabs [aria-selected="true"] { background: var(--surface4) !important; color: var(--indigo-l) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── Expanders ── */
.streamlit-expanderHeader { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: var(--r2) !important; font-family: 'DM Mono', monospace !important; font-size: .73rem !important; color: var(--t3) !important; transition: all .2s !important; }
.streamlit-expanderHeader:hover { border-color: var(--border2) !important; color: var(--t2) !important; }
.stToggle > label { color: var(--t3) !important; font-family: 'DM Mono', monospace !important; font-size: .73rem !important; }
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ══════════════════════════════════════════════════════════
   CUSTOM COMPONENTS — VIVID EDITION
══════════════════════════════════════════════════════════ */

/* ── Hero Section ── */
.hero {
  text-align: center;
  padding: 5rem 2rem 3rem;
  position: relative;
}
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: .5rem;
  font-family: 'DM Mono', monospace; font-size: .65rem;
  letter-spacing: .3em; text-transform: uppercase;
  color: rgba(124,58,237,0.6);
  background: rgba(124,58,237,0.08);
  border: 1px solid rgba(124,58,237,0.18);
  border-radius: 99px; padding: .3rem 1.1rem;
  margin-bottom: 2.5rem;
  animation: fadeUp .6s ease both;
}
.hero-pulse { width: 6px; height: 6px; border-radius: 50%; background: var(--violet-l); animation: pulse 2s infinite; }
.hero-title {
  font-family: 'DM Sans', sans-serif; font-weight: 700;
  font-size: clamp(4.5rem, 12vw, 9rem); line-height: .9;
  letter-spacing: -.06em;
  background: linear-gradient(135deg, #e0e7ff 0%, #a78bfa 30%, #6366f1 55%, #67e8f9 80%, #34d399 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
  animation: fadeUp .6s .1s ease both;
}
.hero-italic {
  font-family: 'Playfair Display', serif; font-style: italic; font-weight: 600;
  font-size: clamp(.9rem, 2.2vw, 1.3rem); color: var(--t2);
  margin-top: .8rem; margin-bottom: 1.2rem;
  animation: fadeUp .6s .2s ease both;
}
.hero-desc {
  font-size: .95rem; color: var(--t2); max-width: 460px;
  margin: 0 auto; line-height: 1.8;
  animation: fadeUp .6s .3s ease both;
}
.hero-tags {
  display: flex; justify-content: center; gap: .5rem; flex-wrap: wrap;
  margin-top: 2rem; animation: fadeUp .6s .4s ease both;
}
.hero-tag {
  font-family: 'DM Mono', monospace; font-size: .63rem;
  letter-spacing: .05em; color: var(--t3);
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: 99px; padding: .22rem .75rem;
}

/* ── Glass Card ── */
.glass {
  background: rgba(20,24,40,0.7);
  backdrop-filter: blur(24px);
  border: 1px solid var(--border);
  border-radius: var(--r4);
  padding: 2rem;
  position: relative; overflow: hidden;
  animation: fadeUp .4s ease both;
}
.glass::before {
  content: ''; position: absolute; top: 0; left: 10%; right: 10%; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}
.glass-violet { border-color: rgba(124,58,237,0.2); }
.glass-cyan   { border-color: rgba(6,182,212,0.2); }
.glass-emerald { border-color: rgba(16,185,129,0.2); }

/* ── Section label ── */
.sec {
  font-family: 'DM Mono', monospace; font-size: .62rem;
  letter-spacing: .2em; text-transform: uppercase;
  color: var(--t4); display: flex; align-items: center; gap: .5rem;
  margin-bottom: .9rem;
}
.sec::after { content: ''; flex: 1; height: 1px; background: var(--border); }
.sec-violet { color: rgba(124,58,237,0.5); }
.sec-cyan   { color: rgba(6,182,212,0.45); }
.sec-emerald { color: rgba(16,185,129,0.45); }
.sec-amber  { color: rgba(245,158,11,0.45); }

/* ── Persona Cards ── */
.persona-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: .7rem; }
.persona-card {
  background: var(--surface2); border: 1px solid var(--border);
  border-radius: var(--r3); padding: 1.3rem 1rem;
  text-align: center; cursor: pointer;
  transition: all .22s cubic-bezier(.4,0,.2,1);
  position: relative; overflow: hidden;
}
.persona-card:hover { transform: translateY(-3px); border-color: var(--border3); }
.persona-card.active {
  border-color: rgba(124,58,237,0.45);
  background: rgba(124,58,237,0.07);
  box-shadow: 0 0 30px rgba(124,58,237,0.1);
}
.persona-card.active::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 2px; background: linear-gradient(90deg, var(--violet), var(--indigo), var(--cyan));
}
.persona-icon {
  font-size: 1.8rem; margin-bottom: .5rem;
  font-family: 'DM Sans', sans-serif; line-height: 1;
}
.persona-name { font-weight: 700; font-size: .9rem; color: var(--t1); }
.persona-role { font-family: 'DM Mono', monospace; font-size: .58rem; color: var(--t3); letter-spacing: .06em; margin-top: .2rem; }

/* ── Mode selector ── */
.mode-row { display: grid; grid-template-columns: repeat(3,1fr); gap: .6rem; margin: .5rem 0; }
.mode-card {
  border: 1px solid var(--border); border-radius: var(--r2);
  padding: .85rem 1rem; cursor: pointer; transition: all .2s ease;
  background: var(--surface2);
}
.mode-card.m-casual  { border-color: rgba(16,185,129,0.3); background: rgba(16,185,129,0.04); }
.mode-card.m-standard { border-color: rgba(99,102,241,0.3); background: rgba(99,102,241,0.04); }
.mode-card.m-intense  { border-color: rgba(244,63,94,0.3);  background: rgba(244,63,94,0.04); }
.mode-name  { font-weight: 600; font-size: .85rem; color: var(--t1); }
.mode-desc  { font-family: 'DM Mono', monospace; font-size: .6rem; color: var(--t3); margin-top: .25rem; line-height: 1.5; }
.mode-pill  { display: inline-block; font-family: 'DM Mono', monospace; font-size: .57rem; letter-spacing: .06em; border-radius: 99px; padding: .13rem .5rem; margin-top: .35rem; }
.pill-green  { background: rgba(16,185,129,0.1); color: rgba(16,185,129,0.8); border: 1px solid rgba(16,185,129,0.2); }
.pill-indigo { background: rgba(99,102,241,0.1); color: rgba(99,102,241,0.8); border: 1px solid rgba(99,102,241,0.2); }
.pill-rose   { background: rgba(244,63,94,0.1);  color: rgba(244,63,94,0.8);  border: 1px solid rgba(244,63,94,0.2); }

/* ── Resume profile card ── */
.resume-card {
  background: var(--surface3); border: 1px solid var(--border2);
  border-radius: var(--r3); padding: 1.4rem;
  position: relative; overflow: hidden; margin-top: .8rem;
}
.resume-card::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: 3px; background: linear-gradient(180deg, var(--violet), var(--cyan));
}
.rc-name  { font-weight: 700; font-size: 1.2rem; color: var(--t1); }
.rc-role  { font-family: 'DM Mono', monospace; font-size: .7rem; color: rgba(99,102,241,0.7); margin-top: .2rem; letter-spacing: .04em; }
.rc-stats { display: flex; gap: 1.5rem; margin-top: 1rem; padding-top: .8rem; border-top: 1px solid var(--border); }
.rc-num   { font-weight: 700; font-size: 1.2rem; color: var(--t1); }
.rc-lbl   { font-family: 'DM Mono', monospace; font-size: .56rem; color: var(--t4); letter-spacing: .1em; text-transform: uppercase; }

/* ── AI Avatar ── */
.ai-bar {
  display: flex; align-items: flex-start; gap: 1.2rem;
  padding: 1.4rem 1.6rem;
  background: var(--surface2); border: 1px solid var(--border2);
  border-radius: var(--r4); margin-bottom: 1.2rem;
  position: relative; overflow: hidden;
  animation: slideDown .35s ease both;
}
.ai-bar::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: 3px; background: linear-gradient(180deg, var(--violet), var(--indigo), var(--cyan));
}
.ai-avatar {
  width: 50px; height: 50px; border-radius: 50%; flex-shrink: 0;
  background: var(--surface3); border: 1.5px solid rgba(124,58,237,0.35);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem; font-family: 'DM Sans', sans-serif;
  box-shadow: 0 0 20px rgba(124,58,237,0.15);
}
.ai-avatar.speaking { animation: speakPulse 2s ease infinite; }
.ai-meta { flex-shrink: 0; }
.ai-name   { font-weight: 700; font-size: .9rem; color: var(--t1); }
.ai-status {
  font-family: 'DM Mono', monospace; font-size: .6rem;
  color: var(--emerald-l); letter-spacing: .06em;
  display: flex; align-items: center; gap: .3rem; margin-top: .2rem;
}
.status-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--emerald-l); animation: blink 2s infinite; }
.status-dot.thinking { background: var(--amber-l); }
.ai-speech {
  font-family: 'Playfair Display', serif; font-style: italic;
  font-size: .95rem; color: var(--t2); line-height: 1.7; flex: 1;
}
.speech-q { color: rgba(124,58,237,0.3); font-size: 1.2rem; vertical-align: -.2em; }

/* ── Reaction badge ── */
.reaction-strip {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .25rem .8rem; border-radius: 99px;
  font-family: 'DM Mono', monospace; font-size: .63rem;
  letter-spacing: .04em; animation: popIn .3s ease both;
  margin-bottom: .5rem;
}
.react-strong  { background: rgba(16,185,129,0.12); color: rgba(52,211,153,0.9); border: 1px solid rgba(16,185,129,0.25); }
.react-average { background: rgba(99,102,241,0.1); color: rgba(129,140,248,0.8); border: 1px solid rgba(99,102,241,0.2); }
.react-weak    { background: rgba(245,158,11,0.1); color: rgba(252,211,77,0.8); border: 1px solid rgba(245,158,11,0.2); }

/* ── Question Card ── */
.q-card {
  background: linear-gradient(135deg, rgba(124,58,237,0.04) 0%, rgba(99,102,241,0.03) 50%, rgba(6,182,212,0.03) 100%);
  border: 1px solid var(--border2);
  border-radius: var(--r5); padding: 2.2rem 2.5rem;
  margin: 1rem 0; position: relative; overflow: hidden;
  animation: fadeUp .35s ease both;
}
.q-card::before {
  content: ''; position: absolute; top: -1px; left: 8%; right: 8%;
  height: 2px; background: linear-gradient(90deg, transparent, var(--violet), var(--indigo), var(--cyan), transparent);
  opacity: .7;
}
.q-number { font-family: 'DM Mono', monospace; font-size: .6rem; letter-spacing: .2em; text-transform: uppercase; color: var(--t4); margin-bottom: .8rem; }
.q-text   { font-family: 'DM Sans', sans-serif; font-size: clamp(1rem, 2vw, 1.3rem); font-weight: 600; line-height: 1.45; color: var(--t1); }
.q-meta   { display: flex; align-items: center; gap: .45rem; flex-wrap: wrap; margin-top: .9rem; }

/* ── Badges ── */
.badge {
  display: inline-flex; align-items: center; gap: .25rem;
  font-family: 'DM Mono', monospace; font-size: .59rem;
  letter-spacing: .04em; border-radius: 99px; padding: .2rem .6rem;
  border: 1px solid;
}
.badge-rapport    { background: rgba(16,185,129,0.08); color: rgba(52,211,153,0.75); border-color: rgba(16,185,129,0.2); }
.badge-technical  { background: rgba(124,58,237,0.08); color: rgba(167,139,250,0.8); border-color: rgba(124,58,237,0.2); }
.badge-behavioral { background: rgba(6,182,212,0.08);  color: rgba(103,232,249,0.75); border-color: rgba(6,182,212,0.2); }
.badge-situational { background: rgba(245,158,11,0.08); color: rgba(252,211,77,0.75); border-color: rgba(245,158,11,0.2); }
.badge-ambition   { background: rgba(244,63,94,0.08);  color: rgba(251,113,133,0.75); border-color: rgba(244,63,94,0.2); }
.badge-comp { background: var(--surface3); color: var(--t3); border-color: var(--border); }
.badge-diff-e { background: rgba(16,185,129,0.06); color: rgba(52,211,153,0.7); border-color: rgba(16,185,129,0.18); }
.badge-diff-m { background: rgba(245,158,11,0.06); color: rgba(252,211,77,0.7); border-color: rgba(245,158,11,0.18); }
.badge-diff-h { background: rgba(244,63,94,0.06);  color: rgba(251,113,133,0.7); border-color: rgba(244,63,94,0.18); }

/* ── Live coaching ── */
.coach-bar {
  display: flex; align-items: flex-start; gap: .6rem;
  padding: .7rem 1rem; border-radius: var(--r2); border: 1px solid;
  font-family: 'DM Sans', sans-serif; font-size: .78rem;
  line-height: 1.55; margin-top: .5rem; transition: all .25s ease;
}
.coach-info    { color: rgba(99,102,241,0.7);  border-color: rgba(99,102,241,0.15); background: rgba(99,102,241,0.04); }
.coach-warn    { color: rgba(245,158,11,0.75); border-color: rgba(245,158,11,0.18); background: rgba(245,158,11,0.04); }
.coach-success { color: rgba(16,185,129,0.8);  border-color: rgba(16,185,129,0.18); background: rgba(16,185,129,0.04); }
.coach-icon { font-size: .95rem; flex-shrink: 0; margin-top: .05rem; }

/* ── Word meter ── */
.word-meter { display: flex; align-items: center; gap: .7rem; margin-top: .45rem; }
.wm-count  { font-family: 'DM Mono', monospace; font-size: .68rem; color: var(--t3); min-width: 55px; }
.wm-track  { flex: 1; height: 3px; background: var(--surface3); border-radius: 99px; overflow: hidden; }
.wm-fill   { height: 100%; border-radius: 99px; transition: width .3s ease, background .3s ease; }
.wm-status { font-family: 'DM Mono', monospace; font-size: .62rem; min-width: 55px; text-align: right; }

/* ── STAR grid ── */
.star-grid  { display: grid; grid-template-columns: repeat(4,1fr); gap: .35rem; }
.star-cell  { background: var(--surface3); border: 1px solid var(--border); border-radius: var(--r1); padding: .55rem; text-align: center; transition: all .2s; }
.star-cell.active { border-color: rgba(16,185,129,0.35); background: rgba(16,185,129,0.06); }
.star-label { font-family: 'DM Mono', monospace; font-size: .56rem; color: var(--t4); letter-spacing: .1em; text-transform: uppercase; margin-bottom: .2rem; }
.star-val   { font-weight: 700; font-size: 1rem; }
.star-y { color: var(--emerald-l); } .star-n { color: var(--t4); }

/* ── Waveform ── */
.wave-wrap { display: flex; align-items: center; justify-content: center; gap: 2px; height: 40px; }
.wave-bar  { width: 2.5px; border-radius: 99px; animation: waveDance var(--spd) ease-in-out infinite alternate; }

/* ── Feedback card ── */
.fb-card {
  background: var(--surface2); border: 1px solid var(--border2);
  border-radius: var(--r4); padding: 1.8rem;
  margin-top: 1rem; position: relative; overflow: hidden;
  animation: slideRight .3s ease both;
}
.fb-card::before {
  content: ''; position: absolute; left: 0; top: 0; bottom: 0;
  width: 3px; background: linear-gradient(180deg, var(--violet), var(--cyan));
}
.fb-score-area { display: flex; align-items: center; gap: 1.4rem; margin-bottom: 1.4rem; flex-wrap: wrap; }
.fb-ring { position: relative; width: 78px; height: 78px; flex-shrink: 0; }
.fb-verdict { font-weight: 700; font-size: 1.1rem; color: var(--t1); }
.fb-sub     { font-family: 'DM Mono', monospace; font-size: .65rem; color: var(--t3); margin-top: .12rem; }
.tone-chips { display: flex; flex-wrap: wrap; gap: .3rem; margin-top: .45rem; }
.tone-chip  { font-family: 'DM Mono', monospace; font-size: .59rem; letter-spacing: .04em; padding: .16rem .5rem; border-radius: 99px; border: 1px solid; }
.tc-pos  { background: rgba(16,185,129,0.08); color: rgba(52,211,153,0.75); border-color: rgba(16,185,129,0.2); }
.tc-neg  { background: rgba(244,63,94,0.07);  color: rgba(251,113,133,0.7); border-color: rgba(244,63,94,0.2); }
.tc-neu  { background: rgba(99,102,241,0.07); color: rgba(129,140,248,0.7); border-color: rgba(99,102,241,0.18); }
.fb-section { margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }
.fb-section:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.fb-label { font-family: 'DM Mono', monospace; font-size: .59rem; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .35rem; }
.fb-label-str { color: rgba(52,211,153,0.7); }
.fb-label-gap { color: rgba(244,63,94,0.7); }
.fb-label-sug { color: rgba(245,158,11,0.7); }
.fb-text { font-size: .88rem; color: var(--t2); line-height: 1.65; }

/* ── SVG ring ── */
.ring-svg { transform: rotate(-90deg); }

/* ── Results hero ── */
.result-hero {
  text-align: center; padding: 4.5rem 2rem;
  background: linear-gradient(145deg, var(--surface) 0%, var(--surface2) 100%);
  border: 1px solid var(--border2); border-radius: var(--r5); margin-bottom: 2rem;
  position: relative; overflow: hidden;
}
.result-hero::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse 60% 50% at 50% -10%, rgba(124,58,237,0.06), transparent);
}
.result-grade { font-family: 'DM Sans', sans-serif; font-size: clamp(5rem, 13vw, 10rem); font-weight: 700; line-height: .88; letter-spacing: -.05em; }
.grade-A { background: linear-gradient(135deg, #34d399, #06b6d4, #e0e7ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.grade-B { background: linear-gradient(135deg, #818cf8, #6366f1, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.grade-C { background: linear-gradient(135deg, #fcd34d, #f59e0b); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.grade-D { background: linear-gradient(135deg, #fb7185, #f43f5e); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.result-score-line { font-family: 'DM Mono', monospace; font-size: .68rem; color: var(--t3); letter-spacing: .18em; text-transform: uppercase; margin-top: .8rem; }
.result-tagline { font-family: 'Playfair Display', serif; font-style: italic; font-size: 1.15rem; color: var(--t2); margin-top: .4rem; }

/* ── Score list item ── */
.qbt-item { display: flex; gap: 1rem; align-items: flex-start; padding: 1rem 0; border-bottom: 1px solid var(--border); }
.qbt-item:last-child { border-bottom: none; }
.qbt-num  { font-family: 'DM Mono', monospace; font-size: .6rem; color: var(--t4); min-width: 22px; padding-top: .1rem; }
.qbt-body { flex: 1; }
.qbt-q    { font-weight: 600; font-size: .88rem; color: var(--t1); line-height: 1.4; margin-bottom: .35rem; }
.qbt-tags { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem; }
.score-pill { display: inline-flex; align-items: center; font-family: 'DM Mono', monospace; font-size: .63rem; padding: .16rem .5rem; border-radius: 99px; border: 1px solid; }

/* ── Insight tip ── */
.tip {
  background: rgba(99,102,241,0.03); border: 1px solid rgba(99,102,241,0.12);
  border-radius: var(--r2); padding: .7rem 1rem;
  font-family: 'DM Mono', monospace; font-size: .72rem;
  color: rgba(99,102,241,0.55); line-height: 1.6;
}
.tip-amber { color: rgba(245,158,11,0.6); border-color: rgba(245,158,11,0.15); background: rgba(245,158,11,0.03); }
.tip-violet { color: rgba(124,58,237,0.6); border-color: rgba(124,58,237,0.18); background: rgba(124,58,237,0.03); }

/* ── Skills ── */
.skills-row   { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .6rem; }
.skill-match  { font-family: 'DM Mono', monospace; font-size: .6rem; padding: .2rem .6rem; border-radius: var(--r1); background: rgba(16,185,129,0.08); color: rgba(52,211,153,0.75); border: 1px solid rgba(16,185,129,0.18); }
.skill-gap    { font-family: 'DM Mono', monospace; font-size: .6rem; padding: .2rem .6rem; border-radius: var(--r1); background: rgba(244,63,94,0.06);  color: rgba(251,113,133,0.65); border: 1px solid rgba(244,63,94,0.16); }
.skill-neutral { font-family: 'DM Mono', monospace; font-size: .6rem; padding: .2rem .6rem; border-radius: var(--r1); background: var(--surface3); color: var(--t3); border: 1px solid var(--border); }

/* ── Followup strip ── */
.followup-strip {
  display: inline-flex; align-items: center; gap: .4rem;
  font-family: 'DM Mono', monospace; font-size: .62rem; letter-spacing: .05em;
  padding: .25rem .75rem; border-radius: 99px;
  background: rgba(244,63,94,0.07); border: 1px solid rgba(244,63,94,0.18);
  color: rgba(251,113,133,0.8); margin-bottom: .7rem;
}

/* ── Benchmark ── */
.bench-bar {
  display: flex; align-items: center; gap: .7rem; margin-bottom: .45rem;
}
.bench-label { font-family: 'DM Mono', monospace; font-size: .62rem; color: var(--t4); min-width: 44px; }
.bench-track { flex: 1; height: 4px; background: var(--surface3); border-radius: 99px; overflow: hidden; }
.bench-fill  { height: 100%; border-radius: 99px; transition: width .5s ease; }
.bench-val   { font-family: 'DM Mono', monospace; font-size: .7rem; color: var(--t3); min-width: 28px; text-align: right; }

/* ── Export ── */
.export-row { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.2rem; background: rgba(245,158,11,0.03); border: 1px solid rgba(245,158,11,0.12); border-radius: var(--r3); margin-bottom: .5rem; }
.export-icon { font-size: 1.3rem; }
.export-title { font-weight: 600; font-size: .88rem; color: var(--t1); }
.export-desc  { font-family: 'DM Mono', monospace; font-size: .6rem; color: var(--t3); margin-top: .12rem; }

/* ── Animations ── */
@keyframes fadeUp   { from { opacity: 0; transform: translateY(16px); filter: blur(6px); } to { opacity: 1; transform: translateY(0); filter: blur(0); } }
@keyframes slideDown { from { opacity: 0; transform: translateY(-8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes slideRight { from { opacity: 0; transform: translateX(-8px); } to { opacity: 1; transform: translateX(0); } }
@keyframes popIn     { from { opacity: 0; transform: scale(.9); } to { opacity: 1; transform: scale(1); } }
@keyframes pulse     { 0%,100% { box-shadow: 0 0 0 0 rgba(124,58,237,.4); } 50% { box-shadow: 0 0 0 5px rgba(124,58,237,0); } }
@keyframes blink     { 0%,100% { opacity: 1; } 50% { opacity: .2; } }
@keyframes speakPulse { 0%,100% { box-shadow: 0 0 20px rgba(124,58,237,.12); } 50% { box-shadow: 0 0 35px rgba(124,58,237,.3); } }
@keyframes waveDance { from { height: 3px; } to { height: var(--maxh); } }
@keyframes gradShift { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
"""

def _inject_css():
    key = f"_css_{_CSS_VERSION}"
    if key not in st.session_state:
        st.markdown(f"<style>{DESIGN}</style>", unsafe_allow_html=True)
        st.session_state[key] = True

# ============================================================
# SESSION STATE
# ============================================================
_STATE_DEFAULTS = {
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
    "ai_summary": None, "camera_enabled": False, "show_hints": True,
    "_pending_followup": False,
    "interview_lang": "English",
    "practice_mode": False,
    "auto_calibrate": False,
    "sentiment_arc": [],
    "keyword_freq": {},
    "paused_state": None,
    "replay_idx": 0,
    "last_reaction": None,
}

def init_state():
    for k, v in _STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def get_llm():
    try:
        return ChatGroq(temperature=0.4, model_name="llama-3.3-70b-versatile",
                        api_key=st.secrets["GROQ_API_KEY"], max_retries=3, request_timeout=30)
    except Exception:
        return None

# ============================================================
# HELPERS
# ============================================================
@lru_cache(maxsize=256)
def score_color(s: float) -> str:
    return "#34d399" if s >= 7 else "#fcd34d" if s >= 5 else "#fb7185"

@lru_cache(maxsize=64)
def grade_letter(avg: float) -> str:
    if avg >= 8.5: return "A+"
    if avg >= 7.5: return "A"
    if avg >= 6.5: return "B+"
    if avg >= 5.5: return "B"
    if avg >= 4.5: return "C"
    return "D"

@lru_cache(maxsize=16)
def grade_css(g: str) -> str:
    return "grade-A" if g.startswith("A") else "grade-B" if g.startswith("B") else "grade-C" if g.startswith("C") else "grade-D"

@lru_cache(maxsize=16)
def grade_tagline(g: str) -> str:
    return {"A+":"Outstanding — a rare calibre of candidate.","A":"Excellent performance — strong hire signal.","B+":"Very good — above expectations in most areas.","B":"Solid candidate with clear strengths.","C":"Adequate but notable gaps remain.","D":"Significant development needed."}.get(g,"Interview complete.")

def _scores_avg(scores): return sum(s.get("score",0) for s in scores)/max(len(scores),1)

def compute_sentiment(answer):
    words = answer.lower().split()
    pos = sum(1 for w in words if w in _POSITIVE_SIGNALS)
    neg = sum(1 for w in words if w in _NEGATIVE_SIGNALS)
    total = pos+neg
    return 0.0 if not total else round((pos-neg)/total,2)

def update_keyword_freq(answer, freq):
    _stop = frozenset({"the","and","for","are","was","were","that","this","with","have","has","had","not","but","from","they","what","when","which","who","will","can","been","its","our","your","their","you","all","one","more","also","just","into","over","some","about","than","then","would","could","should","there"})
    for word in re.findall(r'\b[a-z]{3,}\b', answer.lower()):
        if word not in _stop and word not in FILLER_WORDS:
            freq[word] = freq.get(word,0)+1
    return freq

def calibrate_difficulty(scores, current_diff):
    if len(scores)<2: return current_diff
    recent = sum(s.get("score",5) for s in scores[-3:])/min(len(scores),3)
    return "hard" if recent>=7.5 else "medium" if recent>=5.5 else "easy"

def get_persona_reaction(persona_name, score):
    persona = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    if score >= 7.5:
        return "strong", random.choice(persona["reactions"]["strong"])
    elif score >= 5.5:
        return "average", random.choice(persona["reactions"]["average"])
    else:
        return "weak", random.choice(persona["reactions"]["weak"])

def get_benchmark(category):
    return ROLE_BENCHMARKS.get(category, {"p50":5.8,"p75":7.2,"label":"Candidate"})

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#4b5980"),
    xaxis=dict(gridcolor="#141828", zerolinecolor="#141828"),
    yaxis=dict(gridcolor="#141828", zerolinecolor="#141828"),
    margin=dict(t=30, b=30, l=12, r=12),
)

def ring_svg(score, size=80, stroke=5):
    r = (size/2)-stroke
    circ = 2*math.pi*r
    dash = min(score/10.0,1.0)*circ
    color = score_color(round(score,1))
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" class="ring-svg">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#1e253a" stroke-width="{stroke}"/>'
        f'<circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"'
        f' stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-linecap="round"'
        f' style="filter:drop-shadow(0 0 6px {color}88)"/>'
        f'</svg>'
    )

def waveform_html(n=28, color_a="var(--violet)", color_b="var(--cyan)"):
    bars = "".join(
        f'<div class="wave-bar" style="--spd:{0.3+0.5*abs(math.sin(i)):.2f}s;'
        f'--maxh:{10+int(28*abs(math.sin(i*1.3)))}px;'
        f'height:{3+int(10*abs(math.sin(i*0.7)))}px;'
        f'opacity:{0.4+0.45*abs(math.sin(i*0.9)):.2f};'
        f'background:{color_a if i%3!=2 else color_b};"></div>'
        for i in range(n)
    )
    return f'<div class="wave-wrap">{bars}</div>'

def tts_play(text):
    if not st.session_state.get("tts_enabled") or not text: return
    cache_key = f"_tts_{hash(text[:200])}"
    if cache_key in st.session_state: return
    try:
        buf = BytesIO()
        gTTS(text=text[:500], lang="en", slow=False).write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        st.markdown(f'<audio autoplay style="display:none"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
        st.session_state[cache_key] = True
    except: pass

def transcribe(audio_bytes):
    try:
        from groq import Groq
        gc = Groq(api_key=st.secrets["GROQ_API_KEY"])
        buf = io.BytesIO(audio_bytes); buf.name = "audio.wav"
        return gc.audio.transcriptions.create(model="whisper-large-v3-turbo", file=buf).text.strip()
    except Exception as e:
        st.warning(f"⚠️ Transcription failed: {e}"); return ""

def load_doc(f):
    ext = f.name.rsplit(".",1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(f.getvalue()); path = tmp.name
    try:
        if ext=="pdf": return "\n".join(d.page_content for d in PyPDFLoader(path).load())
        if ext in ("docx","doc"):
            docs = Docx2txtLoader(path).load()
            return docs[0].page_content if docs else ""
        return f.getvalue().decode("utf-8",errors="ignore")
    finally:
        if os.path.exists(path): os.remove(path)

def analyze_resume(resume_text, jd_text, role, llm):
    prompt = (
        f"Analyse this resume against the job description for: {role}\n\n"
        f"RESUME:\n{resume_text[:2500]}\n\nJOB DESCRIPTION:\n{jd_text[:1500]}\n\n"
        "Return ONLY this JSON:\n"
        '{"candidate_name":"<name>","current_role":"<title>","years_experience":"<e.g. 5 years>",'
        '"top_skills":["s1","s2","s3","s4","s5"],"matching_skills":["..."],"gap_skills":["..."],'
        '"education":"<degree>","companies":["c1","c2","c3"],"strengths":["s1","s2","s3"],'
        '"red_flags":[],"overall_fit_score":<0-10>,"fit_rationale":"<1-2 sentences>"}'
    )
    try:
        raw = llm.invoke(prompt).content.strip()
        raw = _RE_JSON_FENCE.sub("",raw).strip().rstrip("`").strip()
        return json.loads(raw)
    except:
        return {"candidate_name":"Candidate","current_role":"Professional","years_experience":"N/A","top_skills":[],"matching_skills":[],"gap_skills":[],"education":"N/A","companies":[],"strengths":[],"red_flags":[],"overall_fit_score":5.0,"fit_rationale":"Resume analysis unavailable."}

def gen_questions(jd, resume, role, n, llm, persona_name, mode, category, resume_profile=None, lang="English"):
    persona = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    comps   = COMPETENCY_FRAMEWORKS.get(category, COMPETENCY_FRAMEWORKS["Engineering"])
    mode_cfg = INTERVIEW_MODES.get(mode, INTERVIEW_MODES["Standard"])
    profile_ctx = ""
    if resume_profile:
        skills = ", ".join(resume_profile.get("top_skills",[])[:5])
        gaps   = ", ".join(resume_profile.get("gap_skills",[])[:3])
        profile_ctx = f"\nCANDIDATE: Skills: {skills} | Gaps to probe: {gaps}"
    mode_rules = {"Casual":"Supportive, accessible.","Standard":"Professional depth, balanced.","Intense":"Multi-part, edge cases, push-back."}.get(mode,"")
    prompt = (
        f"You are {persona['name']}, a {persona['style']} AI interviewer.\n"
        f"Mode: {mode} | Role: {role}\nCompetencies: {', '.join(comps)}{profile_ctx}\n\n"
        f"JD:\n{jd[:2000]}\n\nRESUME:\n{resume[:2000]}\n\n"
        f"Generate exactly {n} tailored questions. Q1-2: rapport, Q3-{max(4,n-3)}: technical/behavioral, Q{max(4,n-3)+1}-{n-1}: situational, Q{n}: ambition.\n"
        f"Mode rules: {mode_rules}\n"+(f"Questions in {lang}.\n" if lang!="English" else "")+
        "Return ONLY valid JSON:\n"
        '{"questions":["q1",...],"types":["rapport",...],"competencies":["Communication",...],"difficulties":["easy","medium","hard",...]}'
    )
    try:
        resp = llm.invoke(prompt)
        raw  = _RE_JSON_FENCE.sub("",resp.content.strip()).strip().rstrip("`").strip()
        d    = json.loads(raw)
        qs   = d.get("questions",[])[:n]; ts=d.get("types",["technical"]*n)[:n]
        cs   = d.get("competencies",["Technical Depth"]*n)[:n]; dfs=d.get("difficulties",["medium"]*n)[:n]
        while len(ts)<len(qs): ts.append("technical")
        while len(cs)<len(qs): cs.append("Technical Depth")
        while len(dfs)<len(qs): dfs.append("medium")
        return qs,ts,cs,dfs
    except:
        qs,ts,cs,dfs=[],[],[],[]
        for line in (resp.content.splitlines() if 'resp' in dir() else []):
            line=line.strip()
            if re.match(r'^\d+[.)\-]',line):
                cleaned=re.sub(r'^\d+[.)\-]\s*','',line).strip()
                if cleaned: qs.append(cleaned);ts.append("technical");cs.append("Technical Depth");dfs.append("medium")
        return qs[:n],ts[:n],cs[:n],dfs[:n]

def analyze_quality(answer):
    words = answer.lower().split(); wc=len(words); text=answer.lower()
    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    for mw in ("you know","sort of","kind of","i mean","so yeah"): filler_count+=text.count(mw)
    filler_pct = filler_count/max(wc,1)
    sentences = [s.strip() for s in re.split(r'[.!?]+',answer) if s.strip()]
    sc = max(len(sentences),1)
    star = {"Situation":bool(_RE_SITUATION.search(text)),"Task":bool(_RE_TASK.search(text)),"Action":bool(_RE_ACTION.search(text)),"Result":bool(_RE_RESULT.search(text))}
    star_score  = sum(star.values())
    specificity = sum([bool(_RE_NUMBERS.search(answer)),bool(_RE_PERCENT.search(answer)),bool(_RE_TIME.search(text))])
    verbosity   = "too_short" if wc<30 else "short" if wc<80 else "ideal" if wc<=250 else "long" if wc<=400 else "too_long"
    if   verbosity=="too_short":             hint=("warn","💡","Very brief — expand with context and a specific example.")
    elif verbosity=="short":                 hint=("warn","✍️","Add more detail — what was the measurable outcome?")
    elif filler_pct>0.07:                    hint=("warn","🎙️",f"High filler density ({filler_count}×). Speak more deliberately.")
    elif star_score==4 and 80<=wc<=250:      hint=("success","✨","Excellent! Full STAR coverage with ideal answer length.")
    elif verbosity=="too_long":              hint=("warn","✂️","Tighten up — focus on the most impactful details.")
    elif star_score>=3:                      hint=("success","🟢","Good structure — covering key STAR components.")
    elif specificity>=2:                     hint=("success","📊","Good use of specifics — that strengthens credibility.")
    else:                                    hint=("info","🎯","Ground your answer in a specific example with a measurable result.")
    return {"wc":wc,"filler_count":filler_count,"filler_pct":filler_pct,"star":star,"star_score":star_score,"verbosity":verbosity,"hint":hint,"specificity":specificity,"avg_sentence_len":wc/sc}

def evaluate(q, answer, role, q_type, competency, mode, persona_name, llm, context=None, lang="English"):
    persona  = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    mode_cfg = INTERVIEW_MODES.get(mode, INTERVIEW_MODES["Standard"])
    ctx_str  = ""
    if context: ctx_str = "CONTEXT:\n" + "\n".join(f"{m['role'].upper()}: {m['content'][:180]}" for m in context[-4:])
    tone_adj = "lenient and encouraging" if mode=="Casual" else "rigorous and exacting" if mode=="Intense" else "balanced and fair"
    lang_note = f"Respond with text fields in {lang}." if lang!="English" else ""
    prompt = (
        f"You are {persona['name']}, a {persona['style']} AI interviewer evaluating a {role} candidate.\n"
        f"Mode: {mode} | Q-type: {q_type} | Competency: {competency}\n{ctx_str}\n\n"
        f"QUESTION: {q}\nANSWER: {answer}\n\nBe {tone_adj}. {lang_note}\n\n"
        f"Return ONLY valid JSON:\n"
        f'{{"score":<0-10>,"competency_score":<0-10>,"verdict":"<Exceptional|Strong|Solid|Average|Weak>",'
        f'"strength":"<1 sentence>","weakness":"<1 sentence>","suggestion":"<1 sentence>",'
        f'"star_feedback":"<1 sentence or empty>","tone_signals":["<3 signals>"],'
        f'"needs_followup":<true|false>,"followup_question":"<question or empty>",'
        f'"interviewer_reaction":"<warm 1-sentence human reaction>","ideal_hint":"<1-2 sentences>"}}'
    )
    _default = {"score":5.0,"competency_score":5.0,"verdict":"Average","strength":"Answer provided.","weakness":"Could not fully evaluate.","suggestion":"Use STAR with a specific example.","star_feedback":"","tone_signals":["Thoughtful"],"needs_followup":False,"followup_question":"","interviewer_reaction":"Thanks for sharing that.","ideal_hint":"Include a specific example with a measurable outcome."}
    try:
        raw = llm.invoke(prompt).content.strip()
        raw = _RE_JSON_FENCE.sub("",raw).strip().rstrip("`").strip()
        r   = json.loads(raw)
        r["score"]            = min(10.0,max(0.0,float(r.get("score",5))))
        r["competency_score"] = min(10.0,max(0.0,float(r.get("competency_score",r["score"]))))
        return r
    except:
        return _default

def gen_summary(feedback_list, role, name, avg_score, persona_name, mode, llm):
    persona = PERSONAS.get(persona_name, PERSONAS["Ketu"])
    qa_pairs = "\n\n".join(
        f"Q{i+1} [{item.get('type','?')} · {item.get('competency','?')}]: {item['q']}\n"
        f"Answer: {item['a'][:220]}…\nScore: {item['eval']['score']}/10 — {item['eval']['verdict']}"
        for i,item in enumerate(feedback_list)
    )
    prompt = (
        f"You are {persona['name']}, a {persona['style']} interviewer writing a post-interview report.\n"
        f"Candidate: {name or 'the candidate'} | Role: {role} | Mode: {mode} | Overall: {avg_score:.1f}/10\n\n"
        f"INTERVIEW DATA:\n{qa_pairs}\n\n"
        "Write a structured assessment with exactly these 4 sections:\n\n"
        "**OVERALL IMPRESSION**\n2-3 sentences on performance, calibre, and role fit.\n\n"
        "**KEY STRENGTHS**\n2-3 sentences referencing specific answers.\n\n"
        "**DEVELOPMENT AREAS**\n2-3 sentences on gaps.\n\n"
        "**HIRING RECOMMENDATION**\n1-2 sentences: Strong Hire / Hire / Hold / No Hire.\n\n"
        f"Prose only. No bullets. Professional but human. Write as {persona['name']}, first person."
    )
    return llm.invoke(prompt).content.strip()

def build_json(state):
    sc_list = state.get("scores",[]); avg=_scores_avg(sc_list)
    return json.dumps({"meta":{"candidate":state.get("candidate_name","Anonymous"),"role":state.get("role_title",""),"mode":state.get("interview_mode","Standard"),"persona":state.get("persona","Ketu"),"date":datetime.now().isoformat(),"version":"3.0"},"summary":{"avg_score":round(avg,2),"grade":grade_letter(round(avg,1)),"total_questions":len(sc_list)},"resume_profile":state.get("resume_profile"),"qa_transcript":[{"num":i+1,"question":item["q"],"type":item.get("type",""),"competency":item.get("competency",""),"difficulty":item.get("difficulty",""),"answer":item["a"],"score":item["eval"].get("score",0),"verdict":item["eval"].get("verdict",""),"strength":item["eval"].get("strength",""),"weakness":item["eval"].get("weakness",""),"suggestion":item["eval"].get("suggestion",""),"tone":item["eval"].get("tone_signals",[]),"time_sec":item.get("time",0),"word_count":item.get("qa",{}).get("wc",0),"filler_words":item.get("qa",{}).get("filler_count",0),"star_score":item.get("qa",{}).get("star_score",0)} for i,item in enumerate(state.get("feedback_list",[]))],"competency_scores":{k:round(sum(v)/len(v),2) for k,v in state.get("competency_scores",{}).items() if v},"communication_stats":{"total_words":sum(state.get("word_counts",[])),"avg_words_per_answer":sum(state.get("word_counts",[]))//max(len(state.get("word_counts",[])),1),"total_filler_words":sum(state.get("filler_counts",[]))},"sentiment_arc":state.get("sentiment_arc",[]),"top_keywords":sorted(state.get("keyword_freq",{}).items(),key=lambda x:x[1],reverse=True)[:20],"ai_assessment":state.get("ai_summary","")},indent=2)

def build_csv(state):
    rows=[{"Q#":i+1,"Question":item["q"],"Type":item.get("type",""),"Competency":item.get("competency",""),"Difficulty":item.get("difficulty",""),"Answer":item["a"][:200]+"…","Score":item["eval"].get("score",0),"Verdict":item["eval"].get("verdict",""),"Strength":item["eval"].get("strength",""),"Gap":item["eval"].get("weakness",""),"Suggestion":item["eval"].get("suggestion",""),"Words":item.get("qa",{}).get("wc",0),"Fillers":item.get("qa",{}).get("filler_count",0),"STAR Score":item.get("qa",{}).get("star_score",0)} for i,item in enumerate(state.get("feedback_list",[]))]
    return pd.DataFrame(rows).to_csv(index=False)

# ============================================================
# CAMERA PANEL
# ============================================================
_CAMERA_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:transparent;font-family:'DM Sans','Helvetica Neue',sans-serif}
#root{background:#0c0f1a;border:1px solid rgba(255,255,255,0.06);border-radius:16px;overflow:hidden}
#topbar{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;background:rgba(0,0,0,0.5);border-bottom:1px solid rgba(255,255,255,0.05)}
.dot-live{width:7px;height:7px;border-radius:50%;background:#f43f5e;animation:blink 1.1s infinite;display:inline-block;margin-right:5px}
#live-lbl{font-size:9.5px;color:#f43f5e;letter-spacing:.15em;font-family:'DM Mono','Courier New',monospace}
#mode-pill{font-size:9px;letter-spacing:.06em;padding:2px 9px;border-radius:99px;background:rgba(99,102,241,0.1);color:rgba(129,140,248,0.7);border:1px solid rgba(99,102,241,0.18);font-family:'DM Mono',monospace}
#fps-lbl{font-size:9px;color:rgba(99,102,241,0.3);font-family:'DM Mono',monospace}
#vidwrap{position:relative;background:#080a12}
video{width:100%;display:block;transform:scaleX(-1)}
canvas#overlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
#eye-badge{position:absolute;top:8px;left:8px;background:rgba(8,10,18,0.82);border:1px solid rgba(99,102,241,0.2);border-radius:8px;padding:4px 8px;display:flex;flex-direction:column;gap:2px}
#eye-score{font-size:15px;font-weight:700;color:#818cf8;line-height:1;font-family:'DM Sans',sans-serif}
#eye-lbl{font-size:7.5px;color:rgba(99,102,241,0.35);letter-spacing:.12em;text-transform:uppercase;font-family:'DM Mono',monospace}
#conf-ring{position:absolute;top:8px;right:8px;width:54px;height:54px}
#conf-ring svg{width:100%;height:100%}
#conf-center{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
#conf-num{font-size:13px;font-weight:700;color:#f1f5ff;line-height:1;font-family:'DM Sans',sans-serif}
#conf-lbl{font-size:7px;color:rgba(99,102,241,0.4);letter-spacing:.1em;text-transform:uppercase;margin-top:1px;font-family:'DM Mono',monospace}
#expr-badge{position:absolute;bottom:44px;left:8px;background:rgba(8,10,18,0.82);border:1px solid rgba(124,58,237,0.22);border-radius:8px;padding:4px 9px;display:flex;align-items:center;gap:5px}
#expr-icon{font-size:14px}#expr-text{font-size:9px;color:rgba(167,139,250,0.75);letter-spacing:.05em;font-family:'DM Mono',monospace}
#posture-badge{position:absolute;bottom:44px;right:8px;background:rgba(8,10,18,0.82);border:1px solid rgba(16,185,129,0.2);border-radius:8px;padding:4px 9px;display:flex;align-items:center;gap:5px}
#posture-icon{font-size:12px}#posture-text{font-size:9px;color:rgba(52,211,153,0.65);letter-spacing:.04em;font-family:'DM Mono',monospace}
#scan{position:absolute;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.15),transparent);animation:scan 4s linear infinite;pointer-events:none}
@keyframes scan{0%{top:0}100%{top:100%}}
#spark-wrap{background:rgba(0,0,0,0.45);border-top:1px solid rgba(255,255,255,0.04);padding:5px 10px 4px;display:flex;align-items:center;gap:8px}
#spark-lbl{font-size:8.5px;color:rgba(99,102,241,0.28);letter-spacing:.14em;min-width:60px;font-family:'DM Mono',monospace}
canvas#sparkline{flex:1;height:22px}
#coach-strip{background:rgba(99,102,241,0.04);border-top:1px solid rgba(99,102,241,0.08);padding:7px 11px;display:flex;align-items:center;gap:7px}
#coach-icon{font-size:12px;flex-shrink:0}#coach-text{font-size:9.5px;color:rgba(99,102,241,0.45);line-height:1.45;letter-spacing:.02em;font-family:'DM Sans',sans-serif}
#stats-row{display:grid;grid-template-columns:repeat(4,1fr);background:#080a12;border-top:1px solid rgba(255,255,255,0.04)}
.stat-cell{padding:5px 0;text-align:center;border-right:1px solid rgba(255,255,255,0.04)}.stat-cell:last-child{border-right:none}
.snum{font-size:12px;font-weight:700;color:#818cf8;font-family:'DM Sans',sans-serif}.slbl{font-size:7.5px;color:#1e253a;letter-spacing:.1em;text-transform:uppercase;margin-top:1px;font-family:'DM Mono',monospace}
#snap-btn{display:block;width:calc(100% - 16px);margin:6px 8px;background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.22);border-radius:7px;color:rgba(167,139,250,0.7);font-family:'DM Mono',monospace;font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;padding:5px 0;cursor:pointer;transition:all .2s}
#snap-btn:hover{background:rgba(124,58,237,0.18);color:rgba(200,180,255,.9)}
#nocam{display:none;padding:2rem 1rem;text-align:center;font-size:11px;color:rgba(99,102,241,0.25);line-height:2;background:#080a12;font-family:'DM Mono',monospace}
#snap-preview{display:none;padding:6px 8px;background:#080a12;border-top:1px solid rgba(255,255,255,0.04)}
#snap-preview img{width:100%;border-radius:7px;border:1px solid rgba(124,58,237,0.2)}
#snap-preview p{font-size:8px;color:rgba(124,58,237,0.4);text-align:center;margin-top:3px;font-family:'DM Mono',monospace}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.15}}
</style></head><body>
<div id="root">
  <div id="topbar">
    <div><span class="dot-live"></span><span id="live-lbl">LIVE</span></div>
    <span id="mode-pill">Loading…</span>
    <span id="fps-lbl">-- fps</span>
  </div>
  <div id="vidwrap">
    <video id="vid" autoplay playsinline muted></video>
    <canvas id="overlay"></canvas>
    <div id="eye-badge"><div id="eye-score">--</div><div id="eye-lbl">Eye Contact</div></div>
    <div id="conf-ring">
      <svg viewBox="0 0 54 54" fill="none">
        <circle cx="27" cy="27" r="22" stroke="#141828" stroke-width="4.5"/>
        <circle id="conf-arc" cx="27" cy="27" r="22" stroke="#818cf8" stroke-width="4.5"
          stroke-linecap="round" stroke-dasharray="0 138.2"
          style="transform-origin:center;transform:rotate(-90deg);transition:stroke-dasharray .5s ease,stroke .4s ease"/>
      </svg>
      <div id="conf-center"><div id="conf-num">--</div><div id="conf-lbl">Conf</div></div>
    </div>
    <div id="expr-badge"><span id="expr-icon">😐</span><span id="expr-text">Neutral</span></div>
    <div id="posture-badge"><span id="posture-icon">🟢</span><span id="posture-text">Upright</span></div>
    <div id="scan"></div>
    <div id="nocam">📷<br>Camera access required.<br>Allow permissions to enable presence analysis.</div>
  </div>
  <div id="spark-wrap"><span id="spark-lbl">CONFIDENCE</span><canvas id="sparkline"></canvas></div>
  <div id="coach-strip"><span id="coach-icon">👁️</span><span id="coach-text">Initialising presence analysis…</span></div>
  <div id="stats-row">
    <div class="stat-cell"><div class="snum" id="stat-eye">--</div><div class="slbl">Avg Eye</div></div>
    <div class="stat-cell"><div class="snum" id="stat-conf">--</div><div class="slbl">Avg Conf</div></div>
    <div class="stat-cell"><div class="snum" id="stat-expr">--</div><div class="slbl">Smile %</div></div>
    <div class="stat-cell"><div class="snum" id="stat-frames">0</div><div class="slbl">Frames</div></div>
  </div>
  <button id="snap-btn">📸 Capture Snapshot</button>
  <div id="snap-preview"><img id="snap-img" src="" alt="Snapshot"><p id="snap-time"></p></div>
</div>
<canvas id="hidden-canvas" style="display:none"></canvas>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0/dist/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface@0.0.7/dist/blazeface.min.js"></script>
<script>
const vid=document.getElementById('vid'),overlay=document.getElementById('overlay');
const ctx2d=overlay.getContext('2d');
const modePill=document.getElementById('mode-pill'),fpsLbl=document.getElementById('fps-lbl');
const eyeScoreEl=document.getElementById('eye-score'),confNumEl=document.getElementById('conf-num');
const confArc=document.getElementById('conf-arc');
const exprIcon=document.getElementById('expr-icon'),exprText=document.getElementById('expr-text');
const postureIc=document.getElementById('posture-icon'),postureT=document.getElementById('posture-text');
const coachIcon=document.getElementById('coach-icon'),coachText=document.getElementById('coach-text');
const sparkCv=document.getElementById('sparkline'),sparkCtx=sparkCv.getContext('2d');
const statEye=document.getElementById('stat-eye'),statConf=document.getElementById('stat-conf');
const statExpr=document.getElementById('stat-expr'),statFrames=document.getElementById('stat-frames');
const nocam=document.getElementById('nocam');
const snapBtn=document.getElementById('snap-btn'),snapPrev=document.getElementById('snap-preview');
const snapImg=document.getElementById('snap-img'),snapTime=document.getElementById('snap-time');
const hiddenCv=document.getElementById('hidden-canvas');
let model=null,running=false;
let lastFpsTime=performance.now(),frameCount=0,fps=0;
const DETECT_INTERVAL=1000/15,HIST=60;
let lastDetect=0,eyeHist=[],confHist=[];
let smileFrames=0,totalFrames=0,emaEye=50,emaConf=50,coachTimer=0;
const EMA_A=0.2;
const COACHING=[
  {t:(e,c)=>e<35,i:'👁️',m:'Maintain eye contact — look directly into the camera lens.'},
  {t:(e,c)=>e<60,i:'🎯',m:'Keep your gaze centred — imagine talking to the interviewer, not the screen.'},
  {t:(e,c)=>c<45,i:'💪',m:'Sit upright and roll your shoulders back to project confidence.'},
  {t:(e,c)=>c<70,i:'📐',m:'Keep your head level — tilting can signal uncertainty.'},
  {t:(e,c)=>e>=60&&c>=70&&c<88,i:'✨',m:'Good presence! Breathe steadily and pause before answering.'},
  {t:(e,c)=>e>=60&&c>=88,i:'🏆',m:'Excellent presence — composed, confident, and engaged.'},
];
function loadModelWhenIdle(){
  const doLoad=async()=>{
    try{modePill.textContent='Loading AI…';model=await blazeface.load({maxFaces:1,scoreThreshold:0.65});modePill.textContent='BlazeFace · Active';modePill.style.color='rgba(52,211,153,0.7)';}catch(e){modePill.textContent='Vision unavailable';}
  };
  if('requestIdleCallback' in window)requestIdleCallback(doLoad,{timeout:3000});else setTimeout(doLoad,500);
}
async function startCam(){
  try{const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:320},height:{ideal:240}},audio:false});vid.srcObject=s;vid.onloadedmetadata=()=>{overlay.width=vid.videoWidth||320;overlay.height=vid.videoHeight||240;hiddenCv.width=overlay.width;hiddenCv.height=overlay.height;running=true;requestAnimationFrame(loop);};nocam.style.display='none';}
  catch(err){vid.style.display='none';nocam.style.display='block';modePill.textContent='No Camera · Demo';runFallback();}
}
async function loop(ts){
  if(!running)return;
  frameCount++;if(ts-lastFpsTime>=1000){fps=frameCount;frameCount=0;lastFpsTime=ts;fpsLbl.textContent=fps+' fps';}
  ctx2d.clearRect(0,0,overlay.width,overlay.height);
  if(model&&vid.readyState===4&&ts-lastDetect>=DETECT_INTERVAL){lastDetect=ts;let preds=[];try{preds=await model.estimateFaces(vid,false);}catch(e){}if(preds.length>0)drawFace(preds[0]);else{drawNoFace();updateMetrics({eye:0,conf:20,expr:'none',posture:'off'});}}else if(!model){drawScanOverlay();fakeTick();}
  updateSparkline();updateStats();requestAnimationFrame(loop);
}
function drawFace(face){
  const [x1,y1]=face.topLeft,[x2,y2]=face.bottomRight;const w=x2-x1,h=y2-y1;const mx1=overlay.width-x2,mx2=overlay.width-x1;const modelConf=Math.round((face.probability?.[0]??0.82)*100);const clen=Math.min(w,h)*.2,cr=8;
  ctx2d.strokeStyle='rgba(129,140,248,0.6)';ctx2d.lineWidth=1.5;
  [[mx1,y1,1,1],[mx2,y1,-1,1],[mx1,y2,1,-1],[mx2,y2,-1,-1]].forEach(([bx,by,sx,sy])=>{ctx2d.beginPath();ctx2d.moveTo(bx+sx*clen,by);ctx2d.lineTo(bx+sx*cr,by);ctx2d.arcTo(bx,by,bx,by+sy*cr,cr);ctx2d.lineTo(bx,by+sy*clen);ctx2d.stroke();});
  if(face.landmarks){const lc=['#818cf8','#818cf8','#fcd34d','#f97316','#94a3b8','#94a3b8'];face.landmarks.forEach((lm,i)=>{const lx=overlay.width-lm[0];ctx2d.beginPath();ctx2d.arc(lx,lm[1],2.5,0,Math.PI*2);ctx2d.fillStyle=lc[i]||'#818cf8';ctx2d.fill();});
  if(face.landmarks.length>=2){const re=face.landmarks[0],le=face.landmarks[1];const rex=overlay.width-re[0],lex2=overlay.width-le[0];ctx2d.beginPath();ctx2d.moveTo(rex,re[1]);ctx2d.lineTo(lex2,le[1]);ctx2d.strokeStyle='rgba(129,140,248,0.1)';ctx2d.lineWidth=1;ctx2d.stroke();}
  const eye=computeEye(face);const tilt=computeTilt(face);const expr=computeExpr(face,h);const posture=tilt<8?'upright':tilt<18?'slight':'leaning';updateMetrics({eye,conf:Math.round(modelConf*.55+eye*.45),expr,posture});}else{updateMetrics({eye:60,conf:modelConf,expr:'neutral',posture:'upright'});}
}
function computeEye(face){if(!face.landmarks||face.landmarks.length<2)return 50;const re=face.landmarks[0],le=face.landmarks[1];const ex=(overlay.width-re[0]+overlay.width-le[0])/2,ey=(re[1]+le[1])/2;const dx=Math.abs(ex-overlay.width/2)/(overlay.width*.5),dy=Math.abs(ey-overlay.height*.42)/(overlay.height*.5);return Math.round(Math.max(0,Math.min(100,(1-Math.sqrt(dx*dx+dy*dy)*1.4)*100)));}
function computeTilt(face){if(!face.landmarks||face.landmarks.length<2)return 0;const re=face.landmarks[0],le=face.landmarks[1];return Math.abs(Math.atan2(le[1]-re[1],le[0]-re[0])*180/Math.PI);}
function computeExpr(face,faceH){if(!face.landmarks||face.landmarks.length<4)return 'neutral';const ratio=(face.landmarks[3][1]-face.landmarks[2][1])/Math.max(faceH*.5,1);return ratio>.55?'smile':ratio<.30?'tense':'neutral';}
function drawNoFace(){ctx2d.strokeStyle='rgba(99,102,241,0.08)';ctx2d.lineWidth=1;ctx2d.setLineDash([4,8]);ctx2d.strokeRect(overlay.width*.25,overlay.height*.15,overlay.width*.5,overlay.height*.7);ctx2d.setLineDash([]);ctx2d.fillStyle='rgba(99,102,241,0.07)';ctx2d.font='10px DM Mono,monospace';ctx2d.textAlign='center';ctx2d.fillText('Position face in frame',overlay.width/2,overlay.height*.92);}
function drawScanOverlay(){ctx2d.fillStyle='rgba(99,102,241,0.03)';ctx2d.fillRect(0,0,overlay.width||320,overlay.height||240);ctx2d.fillStyle='rgba(99,102,241,0.12)';ctx2d.font='10px DM Mono,monospace';ctx2d.textAlign='center';ctx2d.fillText('Loading AI model…',(overlay.width||320)/2,(overlay.height||240)/2);}
const EXPR_INFO={smile:{icon:'😊',text:'Warm',color:'rgba(52,211,153,0.65)'},neutral:{icon:'😐',text:'Neutral',color:'rgba(129,140,248,0.5)'},tense:{icon:'😬',text:'Tense',color:'rgba(252,211,77,0.65)'},none:{icon:'🔍',text:'Scanning',color:'rgba(75,89,128,0.5)'}};
const POSTURE_INFO={upright:{icon:'🟢',text:'Upright'},slight:{icon:'🟡',text:'Slight tilt'},leaning:{icon:'🟠',text:'Leaning'},off:{icon:'⚫',text:'Off-frame'}};
function updateMetrics({eye,conf,expr,posture}){totalFrames++;if(expr==='smile')smileFrames++;emaEye=emaEye*(1-EMA_A)+eye*EMA_A;emaConf=emaConf*(1-EMA_A)+conf*EMA_A;const e=Math.round(emaEye),c=Math.round(emaConf);eyeHist.push(e);if(eyeHist.length>HIST)eyeHist.shift();confHist.push(c);if(confHist.length>HIST)confHist.shift();const ec=e>=65?'#34d399':e>=40?'#fcd34d':'#fb7185';eyeScoreEl.textContent=e+'%';eyeScoreEl.style.color=ec;const CIRC=138.2,dash=(c/100)*CIRC;const cc=c>=65?'#34d399':c>=40?'#fcd34d':'#fb7185';confNumEl.textContent=c;confNumEl.style.color=cc;confArc.setAttribute('stroke-dasharray',`${dash} ${CIRC}`);confArc.setAttribute('stroke',cc);const ei=EXPR_INFO[expr]||EXPR_INFO.neutral;exprIcon.textContent=ei.icon;exprText.textContent=ei.text;exprText.style.color=ei.color;const pi=POSTURE_INFO[posture]||POSTURE_INFO.upright;postureIc.textContent=pi.icon;postureT.textContent=pi.text;coachTimer++;if(coachTimer>fps*6||coachTimer===1){coachTimer=0;const match=COACHING.find(c2=>c2.t(e,c));if(match){coachIcon.textContent=match.i;coachText.textContent=match.m;}}}
function updateSparkline(){const sw=sparkCv.width=sparkCv.offsetWidth||180,sh=sparkCv.height=sparkCv.offsetHeight||22;sparkCtx.clearRect(0,0,sw,sh);if(confHist.length<2)return;const step=sw/(HIST-1);sparkCtx.beginPath();confHist.forEach((v,i)=>{const x=i*step,y=sh-(v/100)*sh;i===0?sparkCtx.moveTo(x,y):sparkCtx.lineTo(x,y);});sparkCtx.strokeStyle='rgba(129,140,248,0.45)';sparkCtx.lineWidth=1.5;sparkCtx.stroke();sparkCtx.lineTo((confHist.length-1)*step,sh);sparkCtx.lineTo(0,sh);sparkCtx.closePath();sparkCtx.fillStyle='rgba(99,102,241,0.05)';sparkCtx.fill();}
function updateStats(){if(!eyeHist.length)return;const ae=Math.round(eyeHist.reduce((a,b)=>a+b,0)/eyeHist.length);const ac=Math.round(confHist.reduce((a,b)=>a+b,0)/confHist.length);const sp=totalFrames>0?Math.round((smileFrames/totalFrames)*100):0;statEye.textContent=ae+'%';statEye.style.color=ae>=65?'#34d399':ae>=40?'#fcd34d':'#fb7185';statConf.textContent=ac;statConf.style.color=ac>=65?'#34d399':ac>=40?'#fcd34d':'#fb7185';statExpr.textContent=sp+'%';statFrames.textContent=totalFrames;}
let _ft=0;
function runFallback(){let t=0;setInterval(()=>{t++;const e=40+Math.round(Math.sin(t*.15)*22+Math.random()*8);const c=52+Math.round(Math.cos(t*.1)*18+Math.random()*6);updateMetrics({eye:e,conf:c,expr:t%22<3?'smile':'neutral',posture:Math.abs(e-50)>15?'slight':'upright'});updateSparkline();updateStats();},200);}
function fakeTick(){_ft++;const e=52+Math.round(Math.sin(_ft*.09)*14);updateMetrics({eye:e,conf:62,expr:'neutral',posture:'upright'});}
snapBtn.addEventListener('click',()=>{const hc=hiddenCv.getContext('2d');hc.save();hc.scale(-1,1);hc.drawImage(vid,-hiddenCv.width,0);hc.restore();const now=new Date().toLocaleTimeString();hc.fillStyle='rgba(0,0,0,0.5)';hc.fillRect(0,hiddenCv.height-22,hiddenCv.width,22);hc.fillStyle='rgba(129,140,248,0.7)';hc.font='9px DM Mono,monospace';hc.fillText('KETU AI v3 · '+now,8,hiddenCv.height-8);snapImg.src=hiddenCv.toDataURL('image/png');snapTime.textContent='Captured at '+now;snapPrev.style.display='block';});
loadModelWhenIdle();startCam();
</script></body></html>"""

def camera_panel():
    import streamlit.components.v1 as components
    components.html(_CAMERA_HTML, height=540, scrolling=False)

# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="font-family:DM Sans,sans-serif;font-weight:700;font-size:1.6rem;'
            'background:linear-gradient(135deg,#a78bfa,#818cf8,#67e8f9);-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-.04em;margin-bottom:.1rem">KETU AI'
            '<span style="font-family:DM Mono,monospace;font-size:.65rem;-webkit-text-fill-color:#2a3355;'
            'font-weight:400;letter-spacing:.15em;margin-left:.4rem">v3.0</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:.6rem;color:#2a3355;letter-spacing:.16em;text-transform:uppercase;margin-bottom:1rem">Vivid Edition</div>', unsafe_allow_html=True)
        st.markdown("---")

        screen  = st.session_state.screen
        persona = PERSONAS.get(st.session_state.persona, PERSONAS["Ketu"])

        if screen == "interview":
            idx = st.session_state.current
            n   = len(st.session_state.questions)
            st.progress(idx/max(n,1))
            scores = st.session_state.scores
            c1,c2 = st.columns(2)
            c1.metric("Question", f"{idx}/{n}")
            if scores: c2.metric("Grade", grade_letter(round(_scores_avg(scores),1)))
            st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:.7rem;color:#2a3355;margin:.5rem 0">Interviewer: {persona["avatar"]} {persona["name"]}</div><div style="font-family:DM Mono,monospace;font-size:.7rem;color:#2a3355;margin-bottom:.8rem">Mode: {st.session_state.interview_mode}</div>', unsafe_allow_html=True)
            st.markdown("---")
            if st.button("⏹ End Interview", use_container_width=True):
                st.session_state.screen="results"; st.rerun()
            st.session_state.camera_enabled = st.toggle("📷 AI Presence Monitor", value=st.session_state.camera_enabled)
            if st.session_state.camera_enabled: camera_panel()

        elif screen == "results":
            scores = st.session_state.scores
            if scores:
                avg = _scores_avg(scores)
                st.success(f"Interview complete · {grade_letter(round(avg,1))}")
                st.metric("Final Score", f"{avg:.1f}/10")
                st.markdown("---")
                if st.button("🔄 New Interview", use_container_width=True):
                    for k in list(st.session_state.keys()): del st.session_state[k]
                    st.rerun()

        st.markdown("---")
        features = ["4 interviewer personas","3 interview modes","Resume deep analysis","STAR tracking","Filler word analysis","Real-time coaching","Sentiment arc","Keyword heatmap","Competency radar","Role benchmark","Auto difficulty calibration","Practice mode","9 language support","Pause & resume","AI presence monitor","CSV + JSON export","Voice input (Whisper)"]
        for f in features:
            st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:.62rem;color:#1e253a;padding:.13rem 0">· {f}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.caption(datetime.now().strftime("%H:%M · %d %b %Y"))

# ============================================================
# SCREEN — SETUP
# ============================================================
def screen_setup():
    st.markdown("""
    <div class="hero">
      <div style="display:flex;justify-content:center">
        <div class="hero-eyebrow"><span class="hero-pulse"></span>Adaptive · Multi-Persona · AI Presence Analysis</div>
      </div>
      <div class="hero-title">KETU AI</div>
      <div class="hero-italic">next-generation interview intelligence</div>
      <p class="hero-desc">Elite AI interviewer with adaptive follow-ups, resume intelligence, STAR tracking, competency mapping, live camera presence, sentiment analysis, and multilingual support.</p>
      <div class="hero-tags">
        <span class="hero-tag">4 Personas</span>
        <span class="hero-tag">8 Competency Frameworks</span>
        <span class="hero-tag">Live STAR Tracking</span>
        <span class="hero-tag">Resume Intelligence</span>
        <span class="hero-tag">Voice Input</span>
        <span class="hero-tag">📷 AI Presence</span>
        <span class="hero-tag">🌐 9 Languages</span>
        <span class="hero-tag">🎭 Practice Mode</span>
        <span class="hero-tag">📈 Benchmarks</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    llm = get_llm()
    if llm is None:
        st.error("⚠️ `GROQ_API_KEY` not found. Add it to `.streamlit/secrets.toml`.")
        return

    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="glass glass-violet">', unsafe_allow_html=True)
        st.markdown('<div class="sec sec-violet">🎭 Choose Your Interviewer</div>', unsafe_allow_html=True)

        p_cols = st.columns(4)
        for i, (pn, pd_) in enumerate(PERSONAS.items()):
            with p_cols[i]:
                active = st.session_state.persona == pn
                st.markdown(
                    f'<div class="persona-card {"active" if active else ""}" style="{"border-color:"+pd_["color"]+"44;background:"+pd_["color"]+"0a" if active else ""}">'
                    f'<div class="persona-icon" style="color:{pd_["color"]}">{pd_["avatar"]}</div>'
                    f'<div class="persona-name">{pn}</div>'
                    f'<div class="persona-role">{pd_["title"]}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("✓ Active" if active else "Select", key=f"p_{pn}", use_container_width=True):
                    st.session_state.persona = pn; st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec">📋 Job Details</div>', unsafe_allow_html=True)
        st.session_state.candidate_name = st.text_input("Your Name (optional)", placeholder="e.g. Arjun Mehta", value=st.session_state.candidate_name)
        st.session_state.role_title     = st.text_input("Role / Job Title *", placeholder="e.g. Senior Backend Engineer", value=st.session_state.role_title)

        c1,c2 = st.columns(2)
        with c1:
            cat = st.selectbox("Role Category", list(COMPETENCY_FRAMEWORKS.keys()), index=list(COMPETENCY_FRAMEWORKS.keys()).index(st.session_state.category_tag))
            st.session_state.category_tag = cat
        with c2:
            mode = st.selectbox("Interview Mode", list(INTERVIEW_MODES.keys()), index=list(INTERVIEW_MODES.keys()).index(st.session_state.interview_mode))
            st.session_state.interview_mode = mode

        active_mode = st.session_state.interview_mode
        st.markdown(
            '<div class="mode-row">' +
            "".join(
                f'<div class="mode-card {"m-"+m.lower() if active_mode==m else ""}">'
                f'<div class="mode-name">{INTERVIEW_MODES[m]["emoji"]} {m}</div>'
                f'<div class="mode-desc">{INTERVIEW_MODES[m]["desc"]}</div>'
                f'<span class="mode-pill {"pill-green" if m=="Casual" else "pill-indigo" if m=="Standard" else "pill-rose"}">'
                f'{INTERVIEW_MODES[m]["pressure"].upper()} PRESSURE</span></div>'
                for m in INTERVIEW_MODES
            ) + '</div>',
            unsafe_allow_html=True,
        )

        st.session_state.jd_text = st.text_area("Job Description *", height=220, placeholder="Paste the full job description here…", value=st.session_state.jd_text)

        st.markdown('<div class="sec" style="margin-top:1rem">⚙️ Settings</div>', unsafe_allow_html=True)
        c3,c4,c5,c6 = st.columns(4)
        with c3: st.session_state.num_questions = st.slider("Questions", 4, 15, st.session_state.num_questions)
        with c4: st.session_state.tts_enabled   = st.toggle("🔊 Voice TTS",  value=st.session_state.tts_enabled)
        with c5: st.session_state.show_hints     = st.toggle("💡 Hints",      value=st.session_state.show_hints)
        with c6: st.session_state.camera_enabled = st.toggle("📷 Camera",     value=st.session_state.camera_enabled)

        st.markdown('<div class="sec" style="margin-top:.6rem">🔬 Advanced</div>', unsafe_allow_html=True)
        fa1,fa2,fa3,fa4 = st.columns(4)
        with fa1: st.session_state.practice_mode  = st.toggle("🎭 Practice",    value=st.session_state.get("practice_mode",False), help="Hide scores during interview")
        with fa2: st.session_state.auto_calibrate = st.toggle("🎯 Auto-Calibrate", value=st.session_state.get("auto_calibrate",False), help="Adjust difficulty based on rolling score")
        with fa3:
            lang = st.selectbox("🌐 Language", list(LANGUAGES.keys()), index=list(LANGUAGES.keys()).index(st.session_state.get("interview_lang","English")))
            st.session_state.interview_lang = lang
        with fa4:
            st.markdown('<div style="font-family:DM Mono,monospace;font-size:.6rem;color:var(--t4);line-height:1.5;margin-top:.3rem">Practice hides scores live.<br>Auto-Calibrate adjusts difficulty.</div>', unsafe_allow_html=True)

        comps = COMPETENCY_FRAMEWORKS.get(cat,[])
        comp_html = "".join(f'<span class="skill-neutral">{c}</span>' for c in comps)
        st.markdown(f'<div class="sec" style="margin-top:.8rem">📊 Competencies for {cat}</div><div class="skills-row">{comp_html}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if st.session_state.camera_enabled:
            st.markdown('<div class="glass" style="padding:1rem;margin-bottom:.8rem">', unsafe_allow_html=True)
            st.markdown('<div class="sec sec-violet">📷 Presence Monitor — Preview</div>', unsafe_allow_html=True)
            camera_panel()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass glass-violet">', unsafe_allow_html=True)
        st.markdown('<div class="sec sec-violet">📄 Resume Upload & Analysis</div>', unsafe_allow_html=True)

        resume_file = st.file_uploader("Upload Resume", type=["pdf","docx","doc","txt"], label_visibility="collapsed")
        if resume_file:
            with st.spinner("🔍 Analysing resume…"):
                text = load_doc(resume_file)
                st.session_state.resume_text = text
                if st.session_state.jd_text.strip():
                    profile = analyze_resume(text, st.session_state.jd_text, st.session_state.role_title or "this role", llm)
                    st.session_state.resume_profile = profile
                    if profile.get("candidate_name") and profile["candidate_name"]!="Candidate" and not st.session_state.candidate_name:
                        st.session_state.candidate_name = profile["candidate_name"]

            profile   = st.session_state.resume_profile or {}
            fit_score = profile.get("overall_fit_score",0)
            fit_color = score_color(round(fit_score,1))
            match_html = "".join(f'<span class="skill-match">✓ {s}</span>' for s in profile.get("matching_skills",[])[:4])
            gap_html   = "".join(f'<span class="skill-gap">✗ {s}</span>'   for s in profile.get("gap_skills",[])[:3])
            skill_html = "".join(f'<span class="skill-neutral">{s}</span>' for s in profile.get("top_skills",[])[:5])
            companies  = " · ".join(profile.get("companies",[])[:3]) or "N/A"
            st.markdown(f"""
            <div class="resume-card">
              <div class="rc-name">{profile.get('candidate_name','Candidate')}</div>
              <div class="rc-role">{profile.get('current_role','Professional')} · {profile.get('years_experience','N/A')} · {profile.get('education','N/A')}</div>
              <div style="font-family:'DM Mono',monospace;font-size:.63rem;color:var(--t3);margin-top:.5rem">Companies: {companies}</div>
              <div class="skills-row" style="margin-top:.7rem">{skill_html}</div>
              {f'<div class="skills-row" style="margin-top:.4rem">{match_html}{gap_html}</div>' if match_html or gap_html else ''}
              <div class="rc-stats">
                <div><div class="rc-num" style="color:{fit_color}">{fit_score:.1f}/10</div><div class="rc-lbl">Role Fit</div></div>
                <div><div class="rc-num">{len(profile.get('matching_skills',[]))}</div><div class="rc-lbl">Matches</div></div>
                <div><div class="rc-num">{len(profile.get('gap_skills',[]))}</div><div class="rc-lbl">Gaps</div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if profile.get("fit_rationale"):
                st.markdown(f'<div class="tip tip-violet" style="margin-top:.6rem">🎯 {profile["fit_rationale"]}</div>', unsafe_allow_html=True)
            with st.expander("📋 Resume text preview"):
                st.text(text[:900]+"…")
        else:
            st.markdown("""
            <div style="border:2px dashed rgba(124,58,237,0.2);border-radius:20px;padding:2rem 1.5rem;
              text-align:center;background:rgba(124,58,237,0.02);margin-top:.5rem">
              <div style="font-size:2rem;margin-bottom:.6rem">📄</div>
              <div style="font-weight:600;font-size:.9rem;color:var(--t1)">Drop your resume here</div>
              <div style="font-family:'DM Mono',monospace;font-size:.63rem;color:var(--t3);margin-top:.3rem">PDF · DOCX · TXT · Auto-analysed against JD</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""<div class="tip tip-violet" style="margin-top:.8rem">
        ✦ KETU AI v3.0 features live camera intelligence using BlazeFace — tracking your eye contact,
        facial expressions, and posture in real-time alongside adaptive answer analysis.
        </div><br>""", unsafe_allow_html=True)

        persona = PERSONAS.get(st.session_state.persona, PERSONAS["Ketu"])
        btn_style = f"background:linear-gradient(135deg,{persona['color']},{persona['accent']});color:#fff;border:none"
        if st.button(f"{persona['avatar']}  Begin Interview with {persona['name']}", use_container_width=True):
            if not st.session_state.jd_text.strip():
                st.error("Please paste a job description.")
            elif not st.session_state.resume_text.strip():
                st.error("Please upload a resume.")
            elif not st.session_state.role_title.strip():
                st.error("Please enter the role / job title.")
            else:
                with st.spinner(f"{persona['avatar']} {persona['name']} is reviewing your profile and crafting questions…"):
                    qs,ts,cs,dfs = gen_questions(
                        st.session_state.jd_text, st.session_state.resume_text,
                        st.session_state.role_title, st.session_state.num_questions,
                        llm, st.session_state.persona, st.session_state.interview_mode,
                        st.session_state.category_tag, st.session_state.resume_profile,
                        lang=st.session_state.get("interview_lang","English"),
                    )
                if not qs:
                    st.error("Could not generate questions. Check your API key."); return
                greeting = random.choice(persona["greetings"])
                st.session_state.update({
                    "questions":qs,"q_types":ts,"q_competencies":cs,"q_difficulties":dfs,
                    "current":0,"scores":[],"feedback_list":[],
                    "transcript":[],"competency_scores":{},"filler_counts":[],"word_counts":[],
                    "ai_summary":None,"session_start":time.time(),"q_start":time.time(),
                    "submitted":False,"ketu_message":greeting,
                    "is_followup":False,"followup_count":0,"_pending_followup":False,
                    "sentiment_arc":[],"keyword_freq":{},"paused_state":None,"replay_idx":0,
                    "last_reaction":None,"screen":"interview",
                })
                st.rerun()

# ============================================================
# SCREEN — INTERVIEW
# ============================================================
def screen_interview():
    llm       = get_llm()
    ss        = st.session_state
    idx       = ss.current
    questions = ss.questions
    n         = len(questions)

    if idx >= n:
        ss.screen = "results"; st.rerun()

    q_types = ss.q_types; q_comps = ss.q_competencies; q_diffs = ss.q_difficulties
    persona  = PERSONAS.get(ss.persona, PERSONAS["Ketu"])
    mode     = ss.interview_mode; mode_cfg = INTERVIEW_MODES[mode]

    q          = questions[idx]
    q_type     = q_types[idx]  if idx < len(q_types)  else "technical"
    competency = q_comps[idx]  if idx < len(q_comps)  else "Technical Depth"
    difficulty = q_diffs[idx]  if idx < len(q_diffs)  else "medium"
    q_info     = QUESTION_TYPES.get(q_type, ("❓","badge-technical",q_type.title()))
    diff_cls   = {"easy":"badge-diff-e","medium":"badge-diff-m","hard":"badge-diff-h"}.get(difficulty,"badge-diff-m")

    elapsed   = int(time.time()-(ss.session_start or time.time()))
    mins,secs = divmod(elapsed,60)
    avg_so_far = _scores_avg(ss.scores) if ss.scores else 0.0

    # Progress bar row
    tb1,tb2,tb3,tb4,tb5,tb6 = st.columns([4,1,1,1,1,1])
    with tb1:
        st.progress(idx/n)
        st.caption(f"Q{idx+1}/{n} · {mins:02d}:{secs:02d} · {mode} · {persona['name']}")
    with tb2: st.metric("Avg",    f"{avg_so_far:.1f}")
    with tb3: st.metric("Done",   f"{len(ss.scores)}/{n}")
    with tb4: st.metric("Words",  f"{sum(ss.word_counts)}")
    with tb5: st.metric("Fillers",f"{sum(ss.filler_counts)}")
    with tb6:
        if st.button("⏹ End", help="End interview"): ss.screen="results"; st.rerun()

    st.markdown("---")

    # ── AI Avatar ─────────────────────────────────────────────
    msg      = ss.ketu_message or ""
    speaking = "speaking" if msg and not ss.submitted else ""
    st.markdown(f"""
    <div class="ai-bar">
      <div class="ai-avatar {speaking}" style="border-color:{persona['color']}55;box-shadow:0 0 20px {persona['glow']};font-size:1.4rem;color:{persona['color']}">{persona['avatar']}</div>
      <div class="ai-meta">
        <div class="ai-name" style="color:{persona['accent']}">{persona['name']}</div>
        <div class="ai-status"><span class="status-dot {"thinking" if ss.submitted else ""}"></span>{persona['title']}</div>
      </div>
      <div class="ai-speech"><span class="speech-q">"</span>{msg or "Ready for your answer…"}<span class="speech-q">"</span></div>
    </div>""", unsafe_allow_html=True)
    if msg: tts_play(msg)

    # Reaction strip from previous answer
    if ss.get("last_reaction"):
        r_level, r_text = ss.last_reaction
        r_cls = {"strong":"react-strong","average":"react-average","weak":"react-weak"}.get(r_level,"react-average")
        r_emoji = {"strong":"✨","average":"→","weak":"💭"}.get(r_level,"→")
        st.markdown(f'<div class="reaction-strip {r_cls}">{r_emoji} {r_text}</div>', unsafe_allow_html=True)

    if ss.is_followup:
        st.markdown('<div class="followup-strip">🔄 Follow-up — Probing deeper on your previous answer</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="q-card">
      <div class="q-number">Question {idx+1} of {n}</div>
      <p class="q-text">{q}</p>
      <div class="q-meta">
        <span class="badge {q_info[1]}">{q_info[0]} {q_info[2]}</span>
        <span class="badge badge-comp">📊 {competency}</span>
        <span class="badge {diff_cls}">{difficulty.upper()}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    if not ss.submitted:
        # Voice input
        if HAS_AUDIO_RECORDER:
            st.markdown('<div class="sec">🎙️ Voice Answer</div>', unsafe_allow_html=True)
            audio_bytes = audio_recorder(text="", icon_size="2x", key=f"rec_{idx}")
            if audio_bytes and f"tr_{idx}" not in ss:
                st.markdown(waveform_html(), unsafe_allow_html=True)
                with st.spinner("Transcribing…"):
                    text_tr = transcribe(audio_bytes)
                    if text_tr: ss[f"ans_{idx}"]=text_tr; ss[f"tr_{idx}"]=True; st.rerun()

        st.markdown('<div class="sec">✍️ Your Answer</div>', unsafe_allow_html=True)
        if f"tr_{idx}" in ss:
            st.info(f"🎙️ Transcribed: *{ss.get(f'ans_{idx}','')}*")

        ans = st.text_area(
            "Your response", value=ss.get(f"ans_{idx}",""),
            key=f"in_{idx}", height=175,
            placeholder="Type your answer here, or use the voice recorder above…",
            label_visibility="collapsed",
        )

        # Live coaching
        if ans.strip():
            qa = analyze_quality(ans)
            wc = qa["wc"]
            pct = min(wc/250,1.0)
            mc  = score_color(wc/25) if 80<=wc<=250 else "#fcd34d" if wc<80 else "#fb7185"
            status = "Ideal ✓" if 80<=wc<=250 else "Too short" if wc<80 else "Too long"
            st.markdown(f"""
            <div class="word-meter">
              <span class="wm-count">{wc} words</span>
              <div class="wm-track"><div class="wm-fill" style="width:{pct*100:.0f}%;background:{mc}"></div></div>
              <span class="wm-status" style="color:{mc}">{status}</span>
            </div>""", unsafe_allow_html=True)

            htype,hicon,htext = qa["hint"]
            hint_cls = {"warn":"coach-warn","success":"coach-success","info":"coach-info"}.get(htype,"coach-info")
            st.markdown(f'<div class="coach-bar {hint_cls}"><span class="coach-icon">{hicon}</span>{htext}</div>', unsafe_allow_html=True)

            if q_type in ("behavioral","situational") and wc>30:
                star_cells = "".join(
                    f'<div class="star-cell {"active" if v else ""}">'
                    f'<div class="star-label">{k}</div>'
                    f'<div class="star-val {"star-y" if v else "star-n"}">{"✓" if v else "○"}</div></div>'
                    for k,v in qa["star"].items()
                )
                st.markdown(f'<div style="margin-top:.6rem"><div class="sec" style="margin-bottom:.35rem">⭐ STAR Coverage</div><div class="star-grid">{star_cells}</div></div>', unsafe_allow_html=True)

        elif ss.show_hints:
            tips = {"technical":"⚙ Mention specific tools, architectures, and measurable outcomes.","behavioral":"◎ Use STAR: Situation · Task · Action · Result.","rapport":"💬 Be authentic — this is about knowing you.","situational":"◈ Walk through your thinking step-by-step. Trade-offs matter.","ambition":"⬆ Connect your goals to what excites you about this role."}
            st.markdown(f'<div class="tip">{tips.get(q_type,"💡 Take your time and be specific.")}</div>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns([3,1,1])
        with c1: submit = st.button("✓  Submit Answer", use_container_width=True)
        with c2: skip   = st.button("Skip →",           use_container_width=True)
        with c3: hint   = st.button("💡 Hint",           use_container_width=True)

        bt1,bt2 = st.columns(2)
        with bt1:
            if st.button("📋 Insert Answer Template", use_container_width=True):
                ss[f"ans_{idx}"] = STAR_TEMPLATES.get(q_type, STAR_TEMPLATES["behavioral"]); st.rerun()
        with bt2:
            if st.button("⏸ Pause & Save Session", use_container_width=True):
                snap = {k:v for k,v in ss.items() if not k.startswith("_css")}
                dl_data = json.dumps(snap, indent=2, default=str)
                st.download_button("⬇️ Download Pause File", data=dl_data, file_name=f"ketu_pause_{datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json")

        if hint:
            star_note = "For behavioral: Situation → Task → Action → Result." if q_type in ("behavioral","situational") else ""
            st.markdown(f'<div class="coach-bar coach-info"><span class="coach-icon">🎯</span>For a <b>{q_type}</b> question on <b>{competency}</b>: focus on specifics, quantified outcomes, and what YOU personally did. {star_note}</div>', unsafe_allow_html=True)

        if skip:
            ss.transcript.append({"role":"user","content":"[Skipped]","q":q})
            ss.update({"current":idx+1,"submitted":False,"is_followup":False,"ketu_message":random.choice(persona["transitions"]),"q_start":time.time(),"last_reaction":None})
            st.rerun()

        if submit:
            if not ans.strip():
                st.warning("Please provide an answer before submitting.")
            else:
                qa = analyze_quality(ans)
                ss.filler_counts.append(qa["filler_count"])
                ss.word_counts.append(qa["wc"])
                with st.spinner(random.choice(persona["thinking"])):
                    ev = evaluate(q, ans, ss.role_title, q_type, competency, mode, ss.persona, llm, ss.transcript[-6:], lang=ss.get("interview_lang","English"))
                    ev["_qa"] = qa

                ss.transcript.append({"role":"user","content":ans,"q":q})
                ss.transcript.append({"role":persona["name"],"content":ev.get("interviewer_reaction","")})
                ss.sentiment_arc.append(compute_sentiment(ans))
                ss.keyword_freq = update_keyword_freq(ans, ss.keyword_freq)

                r_level, r_text = get_persona_reaction(ss.persona, ev.get("score",5))

                if not ss.is_followup:
                    ss.scores.append(ev)
                    if ss.auto_calibrate and idx+1<len(ss.q_difficulties):
                        ss.q_difficulties[idx+1] = calibrate_difficulty(ss.scores, ss.q_difficulties[idx+1])
                    ss.feedback_list.append({"q":q,"a":ans,"eval":ev,"type":q_type,"competency":competency,"difficulty":difficulty,"time":int(time.time()-(ss.q_start or time.time())),"qa":qa})
                    ss.competency_scores.setdefault(competency,[]).append(ev.get("competency_score",ev["score"]))
                else:
                    if ss.scores:
                        prev = ss.scores[-1]["score"]
                        ss.scores[-1]["score"] = min(10.0,(prev+ev["score"])/2+0.5)

                pending = (ev.get("needs_followup",False) and ev.get("followup_question","") and ss.followup_count<mode_cfg["max_followups"] and not ss.is_followup)
                ss.update({"current_feedback":ev,"submitted":True,"_pending_followup":pending,"last_reaction":(r_level,r_text)})
                st.rerun()

    else:
        f         = ss.current_feedback
        sc        = f.get("score",5.0)
        practice  = ss.get("practice_mode",False)
        sc_color  = score_color(round(sc,1)) if not practice else "#3d5580"
        reaction  = f.get("interviewer_reaction","")
        qa_local  = f.get("_qa",{})
        tones     = f.get("tone_signals",[])

        if reaction:
            r_level = ss.get("last_reaction",("average",""))[0]
            st.markdown(f"""
            <div class="ai-bar">
              <div class="ai-avatar" style="color:{persona['color']};border-color:{persona['color']}55">{persona['avatar']}</div>
              <div class="ai-meta">
                <div class="ai-name" style="color:{persona['accent']}">{persona['name']}</div>
                <div class="ai-status"><span class="status-dot thinking"></span>Reviewing</div>
              </div>
              <div class="ai-speech"><span class="speech-q">"</span>{reaction}<span class="speech-q">"</span></div>
            </div>""", unsafe_allow_html=True)
            tts_play(reaction)

        tone_html = "".join(
            f'<span class="tone-chip {"tc-pos" if t in POSITIVE_TONE else "tc-neg" if t in NEGATIVE_TONE else "tc-neu"}">{t}</span>'
            for t in tones
        )
        st.markdown(f"""
        <div class="fb-card">
          <div class="fb-score-area">
            <div class="fb-ring">
              {ring_svg(sc) if not practice else ring_svg(0)}
              <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                font-weight:700;font-size:1.35rem;color:{sc_color}">
                {"?" if practice else f"{sc:.1f}"}
              </div>
            </div>
            <div>
              <div class="fb-verdict">{"Recorded ✓" if practice else f.get("verdict","Average")}</div>
              <div class="fb-sub">{competency} · {"Score hidden in practice mode" if practice else "/10"}</div>
              <div class="tone-chips">{tone_html}</div>
            </div>
          </div>
          <div class="fb-section"><div class="fb-label fb-label-str">✓ Strength</div><div class="fb-text">{f.get('strength','—')}</div></div>
          <div class="fb-section"><div class="fb-label fb-label-gap">✗ Gap</div><div class="fb-text">{f.get('weakness','—')}</div></div>
          <div class="fb-section"><div class="fb-label fb-label-sug">→ Suggestion</div><div class="fb-text">{f.get('suggestion','—')}</div></div>
          {f'<div class="fb-section"><div class="fb-label" style="color:rgba(52,211,153,0.6)">⭐ STAR</div><div class="fb-text">{f.get("star_feedback","")}</div></div>' if f.get("star_feedback") and q_type in ("behavioral","situational") else ''}
        </div>""", unsafe_allow_html=True)

        if f.get("ideal_hint"):
            with st.expander("💡 What a strong answer looks like"):
                st.markdown(f'<div class="tip tip-violet">{f["ideal_hint"]}</div>', unsafe_allow_html=True)

        if qa_local:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Words",       f'{qa_local.get("wc",0)}')
            c2.metric("Fillers",     f'{qa_local.get("filler_count",0)}')
            c3.metric("STAR",        f'{qa_local.get("star_score",0)}/4')
            c4.metric("Specificity", f'{qa_local.get("specificity",0)}/3')

        pending = ss._pending_followup
        fq      = f.get("followup_question","")
        if pending and fq:
            st.markdown(f'<div class="tip tip-amber" style="margin-top:.5rem">🔍 {persona["name"]} wants to explore this further…</div>', unsafe_allow_html=True)
            fc1,fc2 = st.columns(2)
            with fc1:
                if st.button("🔄 Answer Follow-up", use_container_width=True):
                    ss.questions.insert(idx+1,fq); ss.q_types.insert(idx+1,q_type)
                    ss.q_competencies.insert(idx+1,competency); ss.q_difficulties.insert(idx+1,"hard")
                    ss.update({"current":idx+1,"submitted":False,"is_followup":True,"followup_count":ss.followup_count+1,"_pending_followup":False,"ketu_message":f"Good. Let me push on this: {fq}","q_start":time.time()})
                    st.rerun()
            with fc2:
                if st.button("Skip Follow-up →", use_container_width=True):
                    ss.update({"current":idx+1,"submitted":False,"is_followup":False,"_pending_followup":False,"ketu_message":random.choice(persona["transitions"]),"q_start":time.time(),"last_reaction":None})
                    st.rerun()
        else:
            label = "Finish Interview →" if idx+1>=len(questions) else f"Next Question → Q{idx+2}"
            if st.button(label, use_container_width=True):
                ss.update({"current":idx+1,"submitted":False,"is_followup":False,"_pending_followup":False,"ketu_message":random.choice(persona["transitions"]),"q_start":time.time(),"last_reaction":None})
                st.rerun()

# ============================================================
# SCREEN — RESULTS
# ============================================================
def screen_results():
    llm           = get_llm()
    ss            = st.session_state
    scores        = ss.scores
    feedback_list = ss.feedback_list
    persona       = PERSONAS.get(ss.persona, PERSONAS["Ketu"])

    if not scores:
        st.warning("No answers were recorded.")
        if st.button("Start Over"):
            for k in list(ss.keys()): del ss[k]
            st.rerun()
        return

    avg     = _scores_avg(scores)
    grade   = grade_letter(round(avg,1))
    g_cls   = grade_css(grade)
    name    = ss.candidate_name or "Candidate"
    role    = ss.role_title
    elapsed = int(time.time()-(ss.session_start or time.time()))
    mins    = elapsed//60
    n_total = len(ss.questions)
    mode    = ss.interview_mode

    total_words   = sum(ss.word_counts)   if ss.word_counts   else 0
    total_fillers = sum(ss.filler_counts) if ss.filler_counts else 0
    avg_words     = total_words//max(len(ss.word_counts),1)
    star_scores   = [item.get("qa",{}).get("star_score",0) for item in feedback_list if item.get("qa")]
    avg_star      = sum(star_scores)/max(len(star_scores),1)
    filler_pct    = (total_fillers/max(total_words,1))*100

    st.markdown(f"""
    <div class="result-hero">
      <div style="display:flex;justify-content:center;margin-bottom:1rem">
        <div class="hero-eyebrow"><span class="hero-pulse"></span>{name} · {role} · {mode} · {persona['name']}{' · 🎭 Practice' if ss.get('practice_mode') else ''}</div>
      </div>
      <div class="result-grade {g_cls}">{grade}</div>
      <div class="result-score-line">Final Score · {avg:.1f} / 10</div>
      <div class="result-tagline">{grade_tagline(grade)}</div>
    </div>""", unsafe_allow_html=True)

    if ss.get("practice_mode"):
        st.success("🎭 Practice Mode complete — your scores are now revealed.")

    m1,m2,m3,m4,m5,m6,m7 = st.columns(7)
    m1.metric("Score",    f"{avg:.1f}/10")
    m2.metric("Answered", f"{len(scores)}/{n_total}")
    m3.metric("Duration", f"{mins}m")
    m4.metric("Avg Words",f"{avg_words}")
    m5.metric("Fillers",  f"{total_fillers}")
    m6.metric("STAR Avg", f"{avg_star:.1f}/4")
    m7.metric("Filler %", f"{filler_pct:.1f}%")

    st.markdown("---")

    tab1,tab2,tab3,tab4,tab5,tab6,tab7 = st.tabs(["📊 Analytics","📋 Breakdown","🤖 AI Assessment","📄 Resume","🎭 Replay","📈 Insights","⬇️ Export"])

    comp_agg = {k:sum(v)/len(v) for k,v in ss.competency_scores.items() if v}

    with tab1:
        col_l,col_r = st.columns([1.2,0.8],gap="large")
        with col_l:
            if len(scores)>=2:
                st.markdown('<div class="sec">📈 Score Timeline</div>', unsafe_allow_html=True)
                vals = [s.get("score",0) for s in scores]
                fig_line = go.Figure(go.Scatter(
                    x=[f"Q{i+1}" for i in range(len(scores))], y=vals, mode="lines+markers",
                    line=dict(color="#818cf8",width=2.5,shape="spline"),
                    marker=dict(size=9,color=vals,colorscale=[[0,"#fb7185"],[0.5,"#fcd34d"],[1,"#34d399"]],line=dict(color="#080a12",width=2.5)),
                    fill="tozeroy", fillcolor="rgba(99,102,241,0.05)",
                ))
                fig_line.add_hline(y=avg,line_dash="dot",line_color="rgba(129,140,248,0.35)",annotation_text=f"avg {avg:.1f}",annotation_font_color="#818cf8",annotation_font_size=10)
                fig_line.update_layout(**PLOTLY_BASE,height=240,showlegend=False,yaxis=dict(gridcolor="#141828",zerolinecolor="#141828",range=[0,10.5]))
                st.plotly_chart(fig_line,use_container_width=True,config={"displayModeBar":False})

            if ss.word_counts and len(ss.word_counts)>=2:
                st.markdown('<div class="sec">📝 Words per Answer</div>', unsafe_allow_html=True)
                wc_v = ss.word_counts
                fig_wc = go.Figure(go.Bar(
                    x=[f"Q{i+1}" for i in range(len(wc_v))],y=wc_v,
                    marker_color=["#34d399" if 80<=w<=250 else "#fcd34d" if w<80 else "#fb7185" for w in wc_v],
                    marker_line_width=0,text=wc_v,textposition="outside",textfont=dict(size=10,color="#4b5980"),
                ))
                fig_wc.add_hline(y=80,line_dash="dot",line_color="rgba(52,211,153,0.3)")
                fig_wc.add_hline(y=250,line_dash="dot",line_color="rgba(244,63,94,0.3)")
                fig_wc.update_layout(**PLOTLY_BASE,height=180,showlegend=False)
                st.plotly_chart(fig_wc,use_container_width=True,config={"displayModeBar":False})

            if comp_agg:
                st.markdown('<div class="sec">🏆 Competency Scores</div>', unsafe_allow_html=True)
                sorted_c = sorted(comp_agg.items(),key=lambda x:x[1])
                fig_comp = go.Figure(go.Bar(
                    x=[v for _,v in sorted_c],y=[k for k,_ in sorted_c],orientation="h",
                    marker_color=["#34d399" if v>=7 else "#fcd34d" if v>=5 else "#fb7185" for _,v in sorted_c],
                    marker_line_width=0,text=[f"{v:.1f}" for _,v in sorted_c],textposition="outside",textfont=dict(size=10,color="#94a3c8"),
                ))
                fig_comp.update_layout(**PLOTLY_BASE,height=max(180,len(sorted_c)*36),showlegend=False,xaxis=dict(range=[0,11]))
                st.plotly_chart(fig_comp,use_container_width=True,config={"displayModeBar":False})

        with col_r:
            if len(comp_agg)>=3:
                st.markdown('<div class="sec">🕸️ Competency Radar</div>', unsafe_allow_html=True)
                cats_r=list(comp_agg.keys()); vals_r=list(comp_agg.values())
                fig_rad = go.Figure(go.Scatterpolar(
                    r=vals_r+[vals_r[0]],theta=cats_r+[cats_r[0]],fill="toself",
                    fillcolor="rgba(99,102,241,0.06)",line=dict(color="#818cf8",width=2),marker=dict(color="#818cf8",size=6),
                ))
                fig_rad.update_layout(**PLOTLY_BASE,polar=dict(bgcolor="rgba(0,0,0,0)",angularaxis=dict(color="#1e253a",gridcolor="#141828",tickfont=dict(size=9,family="DM Mono")),radialaxis=dict(range=[0,10],color="#1e253a",gridcolor="#141828")),height=300)
                st.plotly_chart(fig_rad,use_container_width=True,config={"displayModeBar":False})

            st.markdown('<div class="sec">📊 Score Distribution</div>', unsafe_allow_html=True)
            bins={"0-4":0,"5-6":0,"7-8":0,"9-10":0}
            for s in scores:
                v=s.get("score",0)
                if v<=4: bins["0-4"]+=1
                elif v<=6: bins["5-6"]+=1
                elif v<=8: bins["7-8"]+=1
                else: bins["9-10"]+=1
            fig_dist = go.Figure(go.Bar(
                x=list(bins.keys()),y=list(bins.values()),
                marker_color=["#fb7185","#fcd34d","#818cf8","#34d399"],marker_line_width=0,
                text=list(bins.values()),textposition="outside",textfont=dict(size=11,color="#4b5980"),
            ))
            fig_dist.update_layout(**PLOTLY_BASE,height=200,showlegend=False)
            st.plotly_chart(fig_dist,use_container_width=True,config={"displayModeBar":False})

            type_counts = Counter(item.get("type","technical") for item in feedback_list)
            if type_counts:
                st.markdown('<div class="sec">🏷️ Question Types</div>', unsafe_allow_html=True)
                type_colors={"technical":"#7c3aed","behavioral":"#818cf8","situational":"#fcd34d","rapport":"#34d399","ambition":"#f97316"}
                fig_type = go.Figure(go.Pie(
                    labels=list(type_counts.keys()),values=list(type_counts.values()),
                    hole=0.6,marker_colors=[type_colors.get(k,"#94a3c8") for k in type_counts],
                    textfont=dict(family="DM Mono",size=9),
                ))
                fig_type.update_layout(**PLOTLY_BASE,height=200,showlegend=True,legend=dict(font=dict(family="DM Mono",size=9,color="#4b5980"),bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(fig_type,use_container_width=True,config={"displayModeBar":False})

            fp_color = "#34d399" if filler_pct<3 else "#fcd34d" if filler_pct<6 else "#fb7185"
            fp_label = "Excellent" if filler_pct<3 else "Acceptable" if filler_pct<6 else "Needs work"
            st.markdown(f"""
            <div class="glass" style="padding:1.2rem;margin-top:.5rem">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem">
                <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:var(--t4)">FILLER DENSITY</div>
                <div style="font-weight:700;font-size:.88rem;color:{fp_color}">{filler_pct:.1f}% · {fp_label}</div>
              </div>
              <div style="height:3px;background:var(--surface3);border-radius:99px;overflow:hidden">
                <div style="height:100%;width:{min(filler_pct/10*100,100):.0f}%;background:{fp_color};border-radius:99px"></div>
              </div>
              <div style="font-family:'DM Mono',monospace;font-size:.6rem;color:var(--t4);margin-top:.4rem">
                {total_fillers} fillers · {total_words} total words · avg {avg_words}/answer
              </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass" style="padding:1.5rem">', unsafe_allow_html=True)
        for i,item in enumerate(feedback_list):
            sc_i    = item["eval"].get("score",0)
            sc_c_i  = score_color(round(sc_i,1))
            verdict = item["eval"].get("verdict","—")
            qt      = item.get("type","technical")
            qinfo   = QUESTION_TYPES.get(qt,("❓","badge-technical",qt.title()))
            comp    = item.get("competency","—")
            diff    = item.get("difficulty","medium")
            diff_c  = {"easy":"badge-diff-e","medium":"badge-diff-m","hard":"badge-diff-h"}.get(diff,"badge-diff-m")
            wc_i    = item.get("qa",{}).get("wc",0)
            fc_i    = item.get("qa",{}).get("filler_count",0)
            star_i  = item.get("qa",{}).get("star_score",0)
            t_secs  = item.get("time",0)
            st.markdown(f"""
            <div class="qbt-item">
              <div class="qbt-num">Q{i+1}</div>
              <div class="qbt-body">
                <div class="qbt-q">{item['q'][:95]}{'…' if len(item['q'])>95 else ''}</div>
                <div class="qbt-tags">
                  <span class="score-pill" style="color:{sc_c_i};border-color:{sc_c_i}44;background:rgba(0,0,0,0.2)">{sc_i:.1f}/10 · {verdict}</span>
                  <span class="badge {qinfo[1]}">{qinfo[0]} {qinfo[2]}</span>
                  <span class="badge badge-comp">{comp}</span>
                  <span class="badge {diff_c}">{diff.upper()}</span>
                  {'<span style="font-family:DM Mono,monospace;font-size:.6rem;color:var(--t4)">⏱ '+str(t_secs)+'s · '+str(wc_i)+'w · '+str(fc_i)+' fillers · ⭐'+str(star_i)+'/4</span>' if wc_i else ''}
                </div>
              </div>
            </div>""", unsafe_allow_html=True)
            with st.expander(f"Full feedback — Q{i+1}: {item['q'][:55]}…"):
                st.markdown(f"**Answer:** {item['a']}")
                tones_i = item["eval"].get("tone_signals",[])
                if tones_i:
                    chips = "".join(f'<span class="tone-chip {"tc-pos" if t in POSITIVE_TONE else "tc-neg" if t in NEGATIVE_TONE else "tc-neu"}">{t}</span>' for t in tones_i)
                    st.markdown(f'<div class="tone-chips">{chips}</div>', unsafe_allow_html=True)
                ca,cb = st.columns(2)
                with ca:
                    st.success(f"**Strength:** {item['eval'].get('strength','—')}")
                    st.info(f"**Suggestion:** {item['eval'].get('suggestion','—')}")
                with cb:
                    st.error(f"**Gap:** {item['eval'].get('weakness','—')}")
                    if item["eval"].get("ideal_hint"): st.markdown(f'<div class="tip tip-violet">💡 {item["eval"]["ideal_hint"]}</div>', unsafe_allow_html=True)
                qa_d = item.get("qa",{})
                if qa_d.get("star") and item.get("type") in ("behavioral","situational"):
                    cells = "".join(f'<div class="star-cell {"active" if v else ""}"><div class="star-label">{k}</div><div class="star-val {"star-y" if v else "star-n"}">{"✓" if v else "○"}</div></div>' for k,v in qa_d["star"].items())
                    st.markdown(f'<div style="max-width:300px;margin-top:.5rem"><div class="star-grid">{cells}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        if not ss.ai_summary:
            with st.spinner(f"✍️ {persona['name']} is writing your assessment…"):
                ss.ai_summary = gen_summary(feedback_list, role, name, avg, ss.persona, mode, llm)
        st.markdown(f"""
        <div class="glass glass-violet">
          <div class="sec sec-violet">{persona['avatar']} {persona['name']}'s Assessment</div>
          <div style="font-family:'Playfair Display',serif;font-style:italic;font-size:.97rem;color:var(--t2);line-height:1.9">
            {ss.ai_summary.replace(chr(10),'<br>')}
          </div>
        </div>""", unsafe_allow_html=True)

    with tab4:
        profile = ss.resume_profile
        if profile:
            fit=profile.get("overall_fit_score",0); fc_=score_color(round(fit,1))
            st.markdown(f"""
            <div class="glass glass-violet">
              <div class="sec sec-violet">📄 Resume Intelligence</div>
              <div class="rc-name">{profile.get('candidate_name','Candidate')}</div>
              <div class="rc-role">{profile.get('current_role','Professional')} · {profile.get('years_experience','N/A')}</div>
              <div style="font-family:'DM Mono',monospace;font-size:.68rem;color:var(--t3);margin:.5rem 0">Education: {profile.get('education','N/A')}</div>
              <div style="font-family:'DM Mono',monospace;font-size:.68rem;color:var(--t3);margin-bottom:.8rem">Companies: {' · '.join(profile.get('companies',[])[:4]) or 'N/A'}</div>""", unsafe_allow_html=True)
            cr,cl = st.columns(2)
            with cr:
                st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:.6rem;color:var(--t4);letter-spacing:.14em;text-transform:uppercase;margin-bottom:.4rem">Role Fit</div><div style="font-weight:700;font-size:2.5rem;color:{fc_}">{fit:.1f}/10</div><div style="font-family:DM Mono,monospace;font-size:.68rem;color:var(--t3);margin-top:.3rem">{profile.get("fit_rationale","")}</div>', unsafe_allow_html=True)
            with cl:
                if profile.get("strengths"):
                    st.markdown('<div style="font-family:DM Mono,monospace;font-size:.6rem;color:rgba(52,211,153,0.6);letter-spacing:.14em;text-transform:uppercase;margin-bottom:.4rem">Strengths</div>', unsafe_allow_html=True)
                    for s in profile["strengths"][:3]: st.markdown(f'<div style="font-size:.82rem;color:var(--t2);padding:.2rem 0">✓ {s}</div>', unsafe_allow_html=True)
            if profile.get("matching_skills"):
                st.markdown('<div style="font-family:DM Mono,monospace;font-size:.6rem;color:rgba(52,211,153,0.6);letter-spacing:.14em;text-transform:uppercase;margin:1rem 0 .4rem">Matches</div>', unsafe_allow_html=True)
                st.markdown('<div class="skills-row">'+"".join(f'<span class="skill-match">✓ {s}</span>' for s in profile["matching_skills"])+'</div>', unsafe_allow_html=True)
            if profile.get("gap_skills"):
                st.markdown('<div style="font-family:DM Mono,monospace;font-size:.6rem;color:rgba(244,63,94,0.6);letter-spacing:.14em;text-transform:uppercase;margin:.8rem 0 .4rem">Gaps</div>', unsafe_allow_html=True)
                st.markdown('<div class="skills-row">'+"".join(f'<span class="skill-gap">✗ {s}</span>' for s in profile["gap_skills"])+'</div>', unsafe_allow_html=True)
            if profile.get("red_flags"): st.warning("⚠️ Potential concerns: " + " · ".join(profile["red_flags"]))
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Resume profile not available — paste the JD before uploading resume.")

    with tab5:
        st.markdown('<div class="sec sec-violet">🎭 Interview Replay</div>', unsafe_allow_html=True)
        if not feedback_list:
            st.info("No answers to replay yet.")
        else:
            total_r = len(feedback_list)
            ridx = st.slider("Jump to question", 1, total_r, min(ss.replay_idx+1,total_r), key="replay_slider")-1
            ss.replay_idx = ridx
            item  = feedback_list[ridx]
            sc_r  = item["eval"].get("score",0)
            sc_c_r = score_color(round(sc_r,1))
            qa_r  = item.get("qa",{})
            st.markdown(f"""
            <div class="q-card">
              <div class="q-number">Question {ridx+1} of {total_r} · {item.get('type','').title()} · {item.get('difficulty','').upper()}</div>
              <p class="q-text">{item['q']}</p>
              <div class="q-meta"><span class="badge badge-comp">📊 {item.get('competency','')}</span></div>
            </div>""", unsafe_allow_html=True)
            st.markdown('<div class="sec" style="margin-top:.5rem">Your Answer</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="background:var(--surface2);border:1px solid var(--border2);border-radius:var(--r3);padding:1rem 1.2rem;font-size:.9rem;color:var(--t2);line-height:1.7">{item["a"]}</div>', unsafe_allow_html=True)
            rc1,rc2,rc3,rc4 = st.columns(4)
            rc1.metric("Score",f"{sc_r:.1f}/10"); rc2.metric("Words",qa_r.get("wc",0)); rc3.metric("STAR",f'{qa_r.get("star_score",0)}/4'); rc4.metric("Fillers",qa_r.get("filler_count",0))
            st.markdown(f"""
            <div class="fb-card" style="margin-top:.8rem">
              <div class="fb-section"><div class="fb-label fb-label-str">✓ Strength</div><div class="fb-text">{item['eval'].get('strength','—')}</div></div>
              <div class="fb-section"><div class="fb-label fb-label-gap">✗ Gap</div><div class="fb-text">{item['eval'].get('weakness','—')}</div></div>
              <div class="fb-section"><div class="fb-label fb-label-sug">→ Suggestion</div><div class="fb-text">{item['eval'].get('suggestion','—')}</div></div>
            </div>""", unsafe_allow_html=True)
            rn1,rn2 = st.columns(2)
            with rn1:
                if st.button("← Previous",use_container_width=True,disabled=ridx==0): ss.replay_idx=max(0,ridx-1); st.rerun()
            with rn2:
                if st.button("Next →",use_container_width=True,disabled=ridx>=total_r-1): ss.replay_idx=min(total_r-1,ridx+1); st.rerun()

    with tab6:
        ins_l,ins_r = st.columns([1,1],gap="large")
        with ins_l:
            if ss.sentiment_arc and len(ss.sentiment_arc)>=2:
                st.markdown('<div class="sec">💬 Sentiment Arc</div>', unsafe_allow_html=True)
                arc_vals = ss.sentiment_arc
                arc_cols = ["#34d399" if v>0.1 else "#fb7185" if v<-0.1 else "#fcd34d" for v in arc_vals]
                fig_arc = go.Figure(go.Bar(x=[f"Q{i+1}" for i in range(len(arc_vals))],y=arc_vals,marker_color=arc_cols,marker_line_width=0,text=[f"{v:+.2f}" for v in arc_vals],textposition="outside",textfont=dict(size=9,color="#4b5980")))
                fig_arc.add_hline(y=0,line_color="rgba(255,255,255,0.06)")
                fig_arc.update_layout(**PLOTLY_BASE,height=200,showlegend=False,yaxis=dict(gridcolor="#141828",zerolinecolor="#141828",range=[-1.1,1.1]))
                st.plotly_chart(fig_arc,use_container_width=True,config={"displayModeBar":False})
                avg_sent = sum(arc_vals)/len(arc_vals)
                sent_label = "Positive — you conveyed energy and achievement." if avg_sent>0.1 else "Neutral — consider more positive framing." if avg_sent>-0.1 else "Negative-leaning — reframe challenges as growth."
                st.markdown(f'<div class="tip">📊 Avg sentiment: <b>{avg_sent:+.2f}</b> — {sent_label}</div>', unsafe_allow_html=True)

            st.markdown('<div class="sec" style="margin-top:1rem">🏆 Role Benchmark</div>', unsafe_allow_html=True)
            bm = get_benchmark(ss.category_tag); bm_p50,bm_p75 = bm["p50"],bm["p75"]
            bm_label = "Above P75 🏆" if avg>bm_p75 else "Above P50 ✓" if avg>bm_p50 else "Below P50 — needs work"
            bm_color = "#34d399" if avg>bm_p75 else "#fcd34d" if avg>bm_p50 else "#fb7185"
            st.markdown(f"""
            <div class="glass" style="padding:1.2rem">
              <div style="font-weight:700;font-size:.85rem;color:{bm_color};margin-bottom:.6rem">{bm_label}</div>
              <div style="font-family:'DM Mono',monospace;font-size:.63rem;color:var(--t3);margin-bottom:.8rem">Role: {bm['label']} · Your score: {avg:.1f}/10</div>
              <div class="bench-bar"><span class="bench-label">YOU</span><div class="bench-track"><div class="bench-fill" style="width:{min(avg/10*100,100):.0f}%;background:{bm_color}"></div></div><span class="bench-val" style="color:{bm_color}">{avg:.1f}</span></div>
              <div class="bench-bar"><span class="bench-label">P75</span><div class="bench-track"><div class="bench-fill" style="width:{bm_p75/10*100:.0f}%;background:rgba(129,140,248,0.4)"></div></div><span class="bench-val">{bm_p75}</span></div>
              <div class="bench-bar"><span class="bench-label">P50</span><div class="bench-track"><div class="bench-fill" style="width:{bm_p50/10*100:.0f}%;background:rgba(94,94,94,0.3)"></div></div><span class="bench-val">{bm_p50}</span></div>
            </div>""", unsafe_allow_html=True)

        with ins_r:
            kf = ss.keyword_freq
            if kf:
                st.markdown('<div class="sec">🔤 Keyword Frequency</div>', unsafe_allow_html=True)
                top_kw = sorted(kf.items(),key=lambda x:x[1],reverse=True)[:20]
                max_cnt = top_kw[0][1] if top_kw else 1
                kw_html = ""
                for word,cnt in top_kw:
                    intensity = cnt/max_cnt
                    bg = f"rgba(99,102,241,{0.05+intensity*0.2:.2f})"
                    border = f"rgba(99,102,241,{0.12+intensity*0.3:.2f})"
                    size = 0.6+intensity*0.25
                    kw_html += (f'<span style="font-family:DM Mono,monospace;font-size:{size:.2f}rem;background:{bg};border:1px solid {border};border-radius:4px;padding:.2rem .5rem;color:rgba(129,140,248,{0.4+intensity*0.55:.2f});display:inline-block;margin:.2rem">{word} <sup style="font-size:.55rem;opacity:.6">{cnt}</sup></span>')
                st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:.1rem;margin-top:.2rem">{kw_html}</div>', unsafe_allow_html=True)
                overused = [w for w,c in top_kw[:5] if c>=3]
                if overused: st.markdown(f'<div class="coach-bar coach-warn" style="margin-top:.8rem"><span class="coach-icon">⚠️</span>Frequently repeated: <b>{", ".join(overused)}</b>. Vary vocabulary for stronger impact.</div>', unsafe_allow_html=True)

            if ss.auto_calibrate and ss.q_difficulties:
                st.markdown('<div class="sec" style="margin-top:1rem">🎯 Auto-Calibration Log</div>', unsafe_allow_html=True)
                diff_colors={"easy":"#34d399","medium":"#fcd34d","hard":"#fb7185"}
                diff_html="".join(f'<span style="font-family:DM Mono,monospace;font-size:.59rem;color:{diff_colors.get(d,"#94a3c8")};background:{diff_colors.get(d,"#94a3c8")}18;border:1px solid {diff_colors.get(d,"#94a3c8")}33;border-radius:99px;padding:.14rem .5rem;margin:.14rem">Q{i+1}: {d.upper()}</span>' for i,d in enumerate(ss.q_difficulties[:len(feedback_list)]))
                st.markdown(f'<div style="display:flex;flex-wrap:wrap;gap:.1rem">{diff_html}</div>', unsafe_allow_html=True)

            st.markdown('<div class="sec" style="margin-top:1rem">⏸ Resume Session</div>', unsafe_allow_html=True)
            resume_file_pause = st.file_uploader("Upload a saved pause file (.json)", type=["json"], key="resume_upload", label_visibility="collapsed")
            if resume_file_pause:
                try:
                    saved=json.loads(resume_file_pause.read().decode())
                    st.session_state.update(saved); st.session_state.screen="interview"; st.session_state.submitted=False
                    st.success("✅ Session restored!"); st.rerun()
                except Exception as e: st.error(f"Could not restore session: {e}")

    with tab7:
        st.markdown('<div class="sec">⬇️ Download Your Report</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        with c1: st.download_button("📦 JSON Report", data=build_json(st.session_state), file_name=f"ketu_v3_{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json", use_container_width=True)
        with c2: st.download_button("📊 CSV Export", data=build_csv(st.session_state), file_name=f"ketu_v3_{name.replace(' ','_')}.csv", mime="text/csv", use_container_width=True)
        with c3:
            if st.button("🔄 New Interview", use_container_width=True):
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.rerun()
        with c4:
            if st.button("📋 Same Role", use_container_width=True):
                r,j,t,c_=ss.resume_text,ss.jd_text,ss.role_title,ss.category_tag
                for k in list(ss.keys()): del ss[k]
                st.session_state.update({"resume_text":r,"jd_text":j,"role_title":t,"category_tag":c_})
                st.rerun()
        with c5:
            if st.button("🔥 Intense Mode", use_container_width=True):
                r,j,t,c_=ss.resume_text,ss.jd_text,ss.role_title,ss.category_tag
                for k in list(ss.keys()): del ss[k]
                st.session_state.update({"resume_text":r,"jd_text":j,"role_title":t,"category_tag":c_,"interview_mode":"Intense"})
                st.rerun()

        st.markdown("""
        <div class="export-row"><div class="export-icon">📦</div><div><div class="export-title">Full Interview Report (JSON)</div><div class="export-desc">Per-question scores · AI assessment · competency breakdown · STAR analysis · communication stats · resume profile.</div></div></div>
        <div class="export-row"><div class="export-icon">📊</div><div><div class="export-title">Tabular Export (CSV)</div><div class="export-desc">Question-by-question breakdown — track progress across sessions.</div></div></div>""", unsafe_allow_html=True)

# ============================================================
# ROUTER
# ============================================================
_inject_css()
render_sidebar()

_screen = st.session_state.screen
if   _screen == "setup":     screen_setup()
elif _screen == "interview": screen_interview()
elif _screen == "results":   screen_results()
