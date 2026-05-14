"""
TalentLens — Streamlit frontend (v6.1)
All AI/RAG/screening logic lives in backend.py
"""

import re
import time
import json
import streamlit as st
import pandas as pd
from pathlib import Path

import backend

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentLens", page_icon="🔍", layout="wide")

# ─── THEME / CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ══ SIDEBAR — pastel blue in light, deep navy in dark ══ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dbeafe 0%, #eff6ff 100%) !important;
        border-right: 1px solid rgba(147,197,253,0.55) !important;
    }
    [data-testid="stSidebar"] * {
        color: #1e3a8a !important;
        -webkit-text-fill-color: #1e3a8a !important;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1e40af !important;
        -webkit-text-fill-color: #1e40af !important;
    }

    /* Inputs */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] input[type="password"],
    [data-testid="stSidebar"] input[type="text"] {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
        caret-color: #0f172a !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stTextInput input {
        background: #ffffff !important;
        border: 1px solid #93c5fd !important;
        color: #0f172a !important;
        border-radius: 10px !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.45) inset !important;
    }
    [data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: #bfdbfe !important;
        color: #1e3a8a !important;
        border: 1px solid #93c5fd !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
        transition: all 0.18s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #93c5fd !important;
        color: #172554 !important;
    }
    [data-testid="stSidebar"] .stDivider { opacity: 0.4 !important; }

    /* ══ UPLOAD BLOCKS — pastel pink like v1 ══ */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        background: #fdf2f8 !important;
        border: 1px solid #f5c2d7 !important;
        border-radius: 14px !important;
        padding: 10px !important;
        box-shadow: 0 6px 18px rgba(244,114,182,0.08) !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: #fff7fb !important;
        border-radius: 12px !important;
        border: 1px dashed #e9a8c3 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: #fde2e7 !important;
        color: #9d174d !important;
        border: 1px solid #f8b4c8 !important;
        border-radius: 10px !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button:hover {
        background: #fbcfe8 !important;
        color: #831843 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] *,
    [data-testid="stSidebar"] [class*="uploadedFile"],
    [data-testid="stSidebar"] [class*="fileName"],
    [data-testid="stSidebar"] [class*="fileSize"] {
        color: #831843 !important;
        -webkit-text-fill-color: #831843 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] li,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
        background: rgba(255,255,255,0.72) !important;
        border: 1px solid rgba(244,114,182,0.18) !important;
        border-radius: 10px !important;
    }

    /* File icons — fix black logos */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] svg,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] img,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFileIcon"] {
        filter: brightness(0) saturate(100%) invert(17%) sepia(87%) saturate(1711%) hue-rotate(307deg) brightness(90%) contrast(98%) !important;
        opacity: 1 !important;
    }

    /* ══ TABS ══ */
    div[data-baseweb="tab-list"] { gap: 0.4rem; border-bottom: 2px solid rgba(59,130,246,0.35); }
    button[data-baseweb="tab"] { padding: 0.55rem 1.3rem; border-radius: 12px 12px 0 0 !important; font-weight: 500; }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #3b82f6 !important; color: #1d4ed8 !important; font-weight: 700 !important;
    }

    /* ══ CARDS ══ */
    .card {
        background: rgba(255,255,255,0.55); border: 1px solid rgba(148,163,184,0.18);
        border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.9rem;
        backdrop-filter: blur(6px);
    }
    .card-hero {
        background: linear-gradient(135deg,rgba(219,234,254,0.8),rgba(253,242,248,0.72));
        border: 1px solid rgba(59,130,246,0.25); border-radius: 14px;
        padding: 1.4rem 1.6rem; margin-bottom: 1rem;
    }

    /* ══ BADGES ══ */
    .best-badge {
        background: linear-gradient(90deg,#3b82f6,#1d4ed8); color: #fff;
        font-size: 11px; font-weight: 700; padding: 4px 14px; border-radius: 20px;
        display: inline-block; margin-bottom: 8px; letter-spacing: 0.03em;
    }
    .verdict { font-size: 12px; padding: 4px 13px; border-radius: 20px; display: inline-block; margin: 4px 2px; font-weight: 600; }
    .v-strong  { background: #dbeafe; color: #1e40af; border: 1px solid rgba(59,130,246,0.22); }
    .v-good    { background: #dcfce7; color: #166534; border: 1px solid rgba(22,163,74,0.22); }
    .v-partial { background: #fef3c7; color: #92400e; border: 1px solid rgba(217,119,6,0.22); }
    .v-weak    { background: #fee2e2; color: #991b1b; border: 1px solid rgba(220,38,38,0.22); }

    /* ══ RECOMMENDATION BADGES ══ */
    .rec-advance { background:rgba(22,163,74,0.14);  color:#15803d; border:1px solid rgba(22,163,74,0.28);  font-size:11px; padding:3px 12px; border-radius:20px; font-weight:700; display:inline-block; }
    .rec-hold    { background:rgba(217,119,6,0.14);  color:#b45309; border:1px solid rgba(217,119,6,0.28);  font-size:11px; padding:3px 12px; border-radius:20px; font-weight:700; display:inline-block; }
    .rec-reject  { background:rgba(220,38,38,0.14);  color:#b91c1c; border:1px solid rgba(220,38,38,0.28);  font-size:11px; padding:3px 12px; border-radius:20px; font-weight:700; display:inline-block; }

    /* ══ TAGS ══ */
    .tag { font-size:11px; padding:3px 11px; border-radius:20px; margin:2px; display:inline-block; font-weight:500; }
    .tag-s { background:#dbeafe; color:#1e40af; border:1px solid rgba(59,130,246,0.18); }
    .tag-g { background:#fee2e2; color:#991b1b; border:1px solid rgba(220,38,38,0.18); }

    /* ══ SCORE ══ */
    .score-big   { font-size:2.8rem; font-weight:800; line-height:1; text-align:center; }
    .score-label { font-size:0.85rem; text-align:center; opacity:0.55; margin-top:2px; }

    /* ══ FILE PILLS ══ */
    .file-pill {
        display:flex; align-items:center; gap:10px;
        background:rgba(219,234,254,0.34); border:1px solid rgba(147,197,253,0.42);
        border-radius:8px; padding:7px 13px; margin-bottom:5px; font-size:13px;
    }
    .file-pill-name { flex:1; font-weight:500; }
    .file-pill-size { font-size:11px; opacity:0.6; }

    /* ══ STATUS DOTS ══ */
    .status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
    .dot-green { background:#4ade80; box-shadow:0 0 6px #4ade8099; }
    .dot-red   { background:#f87171; }
    .dot-amber { background:#fbbf24; }

    /* ══ INFO BOXES ══ */
    .info-box    { background:rgba(219,234,254,0.38);  border-left:4px solid #3b82f6; border-radius:0 8px 8px 0; padding:10px 14px; font-size:13px; color:#1e40af; margin-bottom:12px; }
    .warn-box    { background:rgba(254,243,199,0.5);   border-left:4px solid #d97706; border-radius:0 8px 8px 0; padding:10px 14px; font-size:13px; color:#92400e; margin-bottom:12px; }
    .success-box { background:rgba(220,252,231,0.55);  border-left:4px solid #16a34a; border-radius:0 8px 8px 0; padding:10px 14px; font-size:13px; color:#166534; margin-bottom:12px; }

    /* ══ METRICS ══ */
    [data-testid="stMetric"] {
        background:rgba(219,234,254,0.22) !important; border:1px solid rgba(147,197,253,0.38) !important;
        border-radius:12px !important; padding:0.9rem 1rem !important;
    }
    [data-testid="stMetricValue"] { font-weight:700 !important; }

    /* ══ BUTTONS ══ */
    .stButton > button[kind="primary"] {
        background:linear-gradient(90deg,#3b82f6,#1d4ed8) !important;
        color:#fff !important; border:none !important; border-radius:10px !important;
        font-weight:700 !important; transition:opacity 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover { opacity:0.88 !important; }
    .stButton > button { border-radius:8px; font-weight:500; transition:all 0.15s; }
    .reset-btn > button { background:#fee2e2 !important; color:#991b1b !important; border:1px solid #fca5a5 !important; }
    .reset-btn > button:hover { background:#fecaca !important; }

    /* ══ MATRIX TABLE ══ */
    .matrix-table { width:100%; border-collapse:collapse; font-size:13px; }
    .matrix-table th { background:rgba(219,234,254,0.72); color:#1e40af; padding:9px 16px; text-align:left; font-weight:600; border-bottom:2px solid rgba(59,130,246,0.25); }
    .matrix-table td { padding:8px 16px; border-bottom:1px solid rgba(148,163,184,0.12); vertical-align:middle; }
    .matrix-table tr:hover td { background:rgba(219,234,254,0.14); }
    .matrix-score { font-weight:700; font-size:14px; }

    /* ══ FILTER STRIP ══ */
    .filter-strip {
        background:rgba(255,255,255,0.46); border:1px solid rgba(148,163,184,0.12);
        border-radius:12px; padding:12px 16px; margin-bottom:1rem;
        backdrop-filter: blur(5px);
    }

    /* ══ DARK MODE ══ */
    @media (prefers-color-scheme: dark) {
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
            border-right: 1px solid rgba(59,130,246,0.28) !important;
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
            -webkit-text-fill-color: #e2e8f0 !important;
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] input[type="password"],
        [data-testid="stSidebar"] input[type="text"] {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
            caret-color: #ffffff !important;
        }
        [data-testid="stSidebar"] .stTextInput input {
            background: #0f172a !important;
            border: 1px solid rgba(59,130,246,0.5) !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] .stTextInput input::placeholder { color: #94a3b8 !important; }

        [data-testid="stSidebar"] [data-testid="stFileUploader"] {
            background: rgba(244,114,182,0.1) !important;
            border: 1px solid rgba(244,114,182,0.28) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
            background: rgba(30,41,59,0.76) !important;
            border: 1px dashed rgba(244,114,182,0.36) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] *,
        [data-testid="stSidebar"] [class*="uploadedFile"],
        [data-testid="stSidebar"] [class*="fileName"],
        [data-testid="stSidebar"] [class*="fileSize"] {
            color: #fbcfe8 !important;
            -webkit-text-fill-color: #fbcfe8 !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] li,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
            background: rgba(30,41,59,0.92) !important;
            border: 1px solid rgba(244,114,182,0.12) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background: rgba(244,114,182,0.18) !important;
            color: #fbcfe8 !important;
            border: 1px solid rgba(244,114,182,0.34) !important;
        }
        [data-testid="stSidebar"] [data-testid="stFileUploader"] svg,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] img,
        [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileUploaderFileIcon"] {
            filter: brightness(0) invert(1) sepia(1) saturate(3) hue-rotate(300deg) brightness(1.55) !important;
        }

        .card {
            background: rgba(15,23,42,0.5); border: 1px solid rgba(148,163,184,0.18);
        }
        .card-hero {
            background: linear-gradient(135deg,rgba(30,58,138,0.28),rgba(131,24,67,0.16));
            border: 1px solid rgba(96,165,250,0.28);
        }
        .v-strong  { background: #1e3a8a; color: #bfdbfe; }
        .v-good    { background: #14532d; color: #bbf7d0; }
        .v-partial { background: #713f12; color: #fde68a; }
        .v-weak    { background: #7f1d1d; color: #fecaca; }
        .tag-s { background: #1e3a8a; color: #bfdbfe; }
        .tag-g { background: #7f1d1d; color: #fecaca; }
        .info-box    { color: #93c5fd; background: rgba(30,58,138,0.22); }
        .warn-box    { color: #fcd34d; background: rgba(180,83,9,0.18); }
        .success-box { color: #86efac; background: rgba(20,83,45,0.22); }
        [data-testid="stMetric"] {
            background: rgba(30,58,138,0.12) !important; border: 1px solid rgba(96,165,250,0.22) !important;
        }
        .matrix-table th { background: rgba(30,58,138,0.4); color:#bfdbfe; }
        .matrix-table td { border-bottom:1px solid rgba(255,255,255,0.06); }
        .matrix-table tr:hover td { background: rgba(255,255,255,0.03); }
        .filter-strip { background: rgba(15,23,42,0.42); border:1px solid rgba(148,163,184,0.12); }
        .rec-advance { color:#86efac; }
        .rec-hold { color:#fcd34d; }
        .rec-reject { color:#fca5a5; }
    }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
REQUEST_DELAY = 1

# ─── UI HELPERS ───────────────────────────────────────────────────────────────
def fmt_name(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ")

def strip_think(text: str) -> str:
    """Remove Qwen thinking blocks without destroying markdown formatting."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def shorten_text(text: str, limit: int = 420) -> str:
    """Make a short card summary while keeping the full report untouched."""
    cleaned = re.sub(r"\s+", " ", strip_think(text)).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + "..."


def clean_ai_report(text: str) -> str:
    """Prepare the full AI report for Streamlit markdown display.

    The LLM sometimes returns markdown headings/tables, while some previous UI
    displays flattened the report into one paragraph. This function keeps real
    newlines when present and adds spacing before common headings so the full
    evaluation is readable in the Results tab.
    """
    cleaned = strip_think(text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    # Add spacing before markdown headings if the model placed them inline.
    headings = [
        "Candidate Summary", "Must-Have Skills Match", "Nice-to-Have Skills Match",
        "Experience Assessment", "Key Gaps", "Suitability Score", "Scoring rationale",
        "Recommendation", "Matching Skills", "Missing / Weak Areas", "Missing Areas", "Weak Areas",
    ]
    for h in headings:
        cleaned = re.sub(rf"\s*(\*\*{re.escape(h)}\*\*)", r"\n\n\1", cleaned)
        cleaned = re.sub(rf"(?<!\n)\b({re.escape(h)}\s*:)", r"\n\n\1", cleaned)
    # If table rows got lightly flattened, at least separate row markers for readability.
    cleaned = re.sub(r"\s+(\|\s*-{3,})", r"\n\1", cleaned)
    cleaned = re.sub(r"(\|)\s+(\|\s*[A-Za-z*])", r"\1\n\2", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned

def verdict_label(score):
    if score is None: return "Unknown",      "v-weak"
    if score >= 8:    return "Strong Match",  "v-strong"
    if score >= 6:    return "Good Match",    "v-good"
    if score >= 4:    return "Partial Match", "v-partial"
    return "Weak Match", "v-weak"

def score_color(score):
    if score is None: return "#94a3b8"
    if score >= 8:    return "#60a5fa"
    if score >= 6:    return "#4ade80"
    if score >= 4:    return "#fbbf24"
    return "#f87171"

def extract_recommendation(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    m = re.search(r"\*?\*?Recommendation\*?\*?\s*[:\-–]?\s*(Advance|Hold|Reject)", text, re.IGNORECASE)
    return m.group(1).capitalize() if m else ""

def rec_badge(rec: str) -> str:
    cls  = {"advance": "rec-advance", "hold": "rec-hold", "reject": "rec-reject"}.get(rec.lower(), "")
    icon = {"advance": "✅", "hold": "⏸️", "reject": "❌"}.get(rec.lower(), "")
    return f'<span class="{cls}">{icon} {rec}</span>' if cls else ""

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
_defaults = {
    "groq_key":      "",
    "jd_files":      {},
    "cv_files":      {},
    "raw_jd_texts":  {},
    "raw_cv_texts":  {},
    "results":       {},
    "chat_history":  [],
    "analysis_done": False,
    "files_loaded":  False,
    "anonymise":     False,
    "shortlist":     [],       # list of cv_name strings ticked for emailing
    "uploader_key":  0,        # incrementing this forces file uploaders to visually reset
    "logged_in":     False,
    "username":      "",
    "role":          "",
    "hr_feedback":   {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── ROLE-BASED ACCESS CONTROL ────────────────────────────────────────────────
USER_DB_PATH = Path("users.json")
DEFAULT_USERS = {
    "hr_admin": {"password": "hr123", "role": "HR"},
    "dept_user": {"password": "dept123", "role": "DEPARTMENT"},
}


def load_users() -> dict:
    """Load demo/prototype user accounts from users.json if available."""
    if USER_DB_PATH.exists():
        try:
            data = json.loads(USER_DB_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return data
        except Exception:
            pass
    return DEFAULT_USERS.copy()


def save_users(users: dict) -> None:
    """Save prototype accounts. For production, use DB + hashed passwords."""
    USER_DB_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


USERS = load_users()

def is_hr() -> bool:
    return st.session_state.get("role") == "HR"

def is_department() -> bool:
    return st.session_state.get("role") == "DEPARTMENT"

def display_cv_name(filename: str) -> str:
    """HR sees real filenames; department users see anonymised labels."""
    if is_hr():
        return fmt_name(filename)
    all_names = list(st.session_state.get("cv_files", {}).keys())
    if filename in all_names:
        return f"Candidate {all_names.index(filename) + 1}"
    m = re.search(r"candidate[_\s-]*(\d+)", str(filename), re.IGNORECASE)
    if m:
        return f"Candidate {m.group(1)}"
    return "Candidate"

def anonymised_cv_texts_for_chat():
    if is_hr():
        return st.session_state.raw_cv_texts
    return {
        display_cv_name(name): backend.anonymise_text(text)
        for name, text in st.session_state.raw_cv_texts.items()
    }

if not st.session_state.logged_in:
    st.title("🔐 TalentLens Login")
    st.caption("Role-based access control for fair CV screening")
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="hr_admin or dept_user")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
    if submitted:
        user = USERS.get(username.strip())
        if user and password == user["password"]:
            st.session_state.logged_in = True
            st.session_state.username = username.strip()
            st.session_state.role = user["role"]
            if user["role"] == "DEPARTMENT":
                st.session_state.anonymise = True
            st.rerun()
        else:
            st.error("Invalid username or password")
    st.info("Demo accounts: HR = hr_admin / hr123 · Department = dept_user / dept123")
    st.stop()

def reset_all():
    for k in ["jd_files", "cv_files", "raw_jd_texts", "raw_cv_texts", "results", "chat_history"]:
        st.session_state[k] = {} if k != "chat_history" else []
    st.session_state["analysis_done"] = False
    st.session_state["files_loaded"]  = False
    st.session_state["uploader_key"] += 1
    st.rerun()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 TalentLens")
    st.caption("Qwen3-32B via Groq · FAISS RAG")
    st.markdown(f"**Logged in:** `{st.session_state.username}`")
    role_label = "HR / Admin" if is_hr() else "Department User"
    st.caption(f"Role: {role_label}")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()
    st.divider()

    st.markdown("### 🔑 Groq API Key")
    key_val = st.text_input(
        "API Key", type="password",
        value=st.session_state.groq_key,
        label_visibility="collapsed",
        placeholder="Enter your Groq API key…"
    )
    col_save, col_status = st.columns([2, 1])
    with col_save:
        if st.button("💾 Save Key", use_container_width=True):
            st.session_state.groq_key = key_val.strip()
            st.success("Saved!")
    with col_status:
        dot   = "dot-green" if st.session_state.groq_key else "dot-red"
        label = "OK" if st.session_state.groq_key else "Not set"
        st.markdown(f'<span class="status-dot {dot}"></span>{label}', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📁 Upload Files")

    jd_uploads = st.file_uploader(
        "Job Descriptions (PDF / TXT / DOCX)",
        type=["pdf", "txt", "docx"], accept_multiple_files=True,
        key=f"jd_uploader_{st.session_state.uploader_key}",
    )
    cv_uploads = st.file_uploader(
        "Candidate CVs (PDF / TXT / DOCX)",
        type=["pdf", "txt", "docx"], accept_multiple_files=True,
        key=f"cv_uploader_{st.session_state.uploader_key}",
    )

    st.divider()

    if st.button("⚙️ Load & Preprocess", use_container_width=True, type="primary"):
        if not jd_uploads or not cv_uploads:
            st.warning("⚠️ Upload at least one JD and one CV first.")
        else:
            # ── Input validation ──
            val_warnings = backend.validate_uploads(jd_uploads, cv_uploads)
            if val_warnings:
                for w in val_warnings:
                    st.warning(f"⚠️ {w}")
                st.stop()
            with st.spinner("Reading and preprocessing files…"):
                loaded = backend.load_and_preprocess_files(jd_uploads, cv_uploads)
            if loaded["errors"]:
                for e in loaded["errors"]:
                    st.error(e)
            else:
                st.session_state.update({
                    "jd_files":     loaded["jd_files"],
                    "raw_jd_texts": loaded["raw_jd_texts"],
                    "cv_files":     loaded["cv_files"],
                    "raw_cv_texts": loaded["raw_cv_texts"],
                    "results":      {},
                    "analysis_done": False,
                    "files_loaded": True,
                })
                st.success(f"✅ {len(loaded['jd_files'])} JD(s) · {len(loaded['cv_files'])} CV(s) loaded")

    if st.session_state.files_loaded:
        n_jd = len(st.session_state.jd_files)
        n_cv = len(st.session_state.cv_files)
        st.markdown(f'<div style="font-size:12px;margin-top:4px"><span class="status-dot dot-green"></span>Loaded: {n_jd} JD · {n_cv} CVs</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-size:12px;margin-top:4px"><span class="status-dot dot-amber"></span>No files loaded yet</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### ⚙️ Options")
    if is_department():
        st.session_state.anonymise = True
        st.markdown('<div class="info-box">🕵️ Department mode — anonymisation is forced ON. Full candidate details are hidden.</div>', unsafe_allow_html=True)
    else:
        st.session_state.anonymise = st.toggle(
            "🕵️ Anonymise CVs",
            value=st.session_state.anonymise,
            help="Strips names, emails, universities and pronouns from CVs before screening to reduce bias.",
        )
        if st.session_state.anonymise:
            st.markdown('<div class="info-box">🕵️ Anonymisation ON — names, emails, universities and pronouns will be hidden from the AI.</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🗑️ Reset")
    st.caption("Fully wipes all loaded data and results.")
    with st.container():
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Reset All Data", use_container_width=True):
            reset_all()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.caption(f"TalentLens v6 · FAISS RAG · Qwen3-32B · backend {backend.VERSION}")

# ─── TABS ─────────────────────────────────────────────────────────────────────
if is_hr():
    tab_run, tab_res, tab_matrix, tab_chat, tab_users = st.tabs(
        ["🚀 Run Screening", "📊 Results", "🗂️ Compare Matrix", "💬 Chat", "👥 User Management"]
    )
else:
    tab_run, tab_res, tab_matrix, tab_chat = st.tabs(
        ["🚀 Run Screening", "📊 Results", "🗂️ Compare Matrix", "💬 Chat"]
    )
    tab_users = None

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN SCREENING
# ══════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.markdown("## CV Screening")

    if not st.session_state.groq_key:
        st.markdown('<div class="warn-box">👈 <strong>Step 1:</strong> Add your Groq API key in the sidebar.</div>', unsafe_allow_html=True)
    elif not st.session_state.files_loaded:
        st.markdown('<div class="info-box">👈 <strong>Step 2:</strong> Upload files then click <strong>Load &amp; Preprocess</strong>.</div>', unsafe_allow_html=True)
    else:
        col_jd, col_cv = st.columns(2)
        with col_jd:
            st.markdown("**📄 Job Descriptions**")
            for n in st.session_state.jd_files:
                st.markdown(f'<div class="file-pill"><span>📄</span><span class="file-pill-name">{display_cv_name(n)}</span><span class="file-pill-size">{len(st.session_state.raw_jd_texts[n]):,} chars</span></div>', unsafe_allow_html=True)
        with col_cv:
            st.markdown("**👤 Candidate CVs**")
            for n in st.session_state.cv_files:
                st.markdown(f'<div class="file-pill"><span>👤</span><span class="file-pill-name">{display_cv_name(n)}</span><span class="file-pill-size">{len(st.session_state.raw_cv_texts[n]):,} chars</span></div>', unsafe_allow_html=True)

        st.divider()

        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            use_rag = st.checkbox("✅ Use RAG (recommended)", value=True,
                                  help="Retrieves most relevant CV sections before scoring.")
        with col_opt2:
            delay = st.number_input("Delay between calls (s)", min_value=0, max_value=5, value=REQUEST_DELAY, step=1,
                                    help="Increase if you hit 429 errors.")

        if st.session_state.anonymise:
            st.markdown('<div class="info-box">🕵️ <strong>Anonymisation is ON</strong> — candidate names, emails, universities and pronouns will be stripped before the AI sees the CV.</div>', unsafe_allow_html=True)

            # Show anonymised CV preview for testing
            if st.session_state.get("cv_files"):
                show_anon_preview = st.checkbox(
                    "🔍 Show anonymised CV preview",
                    value=False,
                    help="Shows the anonymised CV text that will be sent to the AI. Use this to verify PII has been removed.",
                )
                if show_anon_preview:
                    st.caption("Preview of anonymised CV text. Page view is easier to inspect and can be selected/copied; LLM view shows the compact processed text used for screening.")
                    for name in st.session_state.cv_files:
                        raw = st.session_state.raw_cv_texts.get(name, "")
                        anon_page = backend.anonymise_text(raw).strip() if raw else "(no raw text available)"
                        anon_llm = backend.preprocess_text(anon_page) if raw else "(no raw text available)"

                        with st.expander(f"🕵️ Preview: {fmt_name(name)}"):
                            page_tab, llm_tab = st.tabs(["📄 Page view", "🤖 LLM view"])

                            with page_tab:
                                st.text_area(
                                    "Anonymised CV page view",
                                    value=anon_page[:6000],
                                    height=520,
                                    disabled=False,
                                    label_visibility="collapsed",
                                )

                            with llm_tab:
                                st.text_area(
                                    "Compact text sent to LLM after preprocessing",
                                    value=anon_llm[:6000],
                                    height=300,
                                    disabled=False,
                                    label_visibility="collapsed",
                                )

        total_pairs = len(st.session_state.jd_files) * len(st.session_state.cv_files)
        st.markdown(f'<div class="info-box">Ready to screen <strong>{len(st.session_state.cv_files)} candidate(s)</strong> against <strong>{len(st.session_state.jd_files)} job description(s)</strong> — {total_pairs} API call(s) total.</div>', unsafe_allow_html=True)

        if st.button("▶️ Run Screening", type="primary", use_container_width=True):
            prog = st.progress(0, text="Starting…")

            def progress_cb(step, total, jd_name, cv_name):
                prog.progress(step / total, text=f"({step}/{total}) Scoring {fmt_name(cv_name)} for {fmt_name(jd_name)}…")

            try:
                screen_cv_files = st.session_state.cv_files
                screen_raw_cv_texts = st.session_state.raw_cv_texts
                screen_anonymise = st.session_state.anonymise

                # Department users should not pass real filenames/details into the LLM.
                # Use generic candidate labels and force anonymisation.
                if is_department():
                    alias_map = {name: f"Candidate_{i+1}" for i, name in enumerate(st.session_state.cv_files.keys())}
                    screen_cv_files = {alias_map[name]: text for name, text in st.session_state.cv_files.items()}
                    screen_raw_cv_texts = {alias_map[name]: st.session_state.raw_cv_texts[name] for name in st.session_state.cv_files if name in st.session_state.raw_cv_texts}
                    screen_anonymise = True

                all_results = backend.screen_candidates(
                    jd_files=st.session_state.jd_files,
                    cv_files=screen_cv_files,
                    groq_key=st.session_state.groq_key,
                    use_rag=use_rag,
                    anonymise=screen_anonymise,
                    delay=int(delay),
                    progress_callback=progress_cb,
                    raw_cv_files=screen_raw_cv_texts,
                )
                prog.empty()
                st.session_state.results       = all_results
                st.session_state.analysis_done = True
                first_jd = list(all_results.keys())[0]
                best     = all_results[first_jd][0]
                st.markdown(f'<div class="success-box">✅ Screening complete! Top candidate: <strong>{display_cv_name(best["cv_name"])}</strong> — Score: <strong>{best["score"]}/10</strong>. Open <strong>Results</strong> or <strong>Compare Matrix</strong> tab.</div>', unsafe_allow_html=True)
            except Exception as e:
                prog.empty()
                st.error(f"❌ Screening failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_res:
    st.markdown("## 📊 Results")

    if not st.session_state.analysis_done:
        st.markdown('<div class="warn-box">⚠️ No results yet — run screening first.</div>', unsafe_allow_html=True)
    else:
        res_dict = st.session_state.results
        jd_names = list(res_dict.keys())
        sel_jd   = st.selectbox("Select Job Description", jd_names, format_func=fmt_name) if len(jd_names) > 1 else jd_names[0]
        jd_res   = res_dict[sel_jd]
        scores   = [r["score"] for r in jd_res if r["score"] is not None]

        # ── FILTER STRIP ──
        st.markdown('<div class="filter-strip">', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns([2, 2, 2])
        with fc1:
            min_score = st.slider("Min Score", 0.0, 10.0, 0.0, 0.5, key="filter_score")
        with fc2:
            filter_verdict = st.selectbox("Verdict", ["All", "Strong Match", "Good Match", "Partial Match", "Weak Match"], key="filter_verdict")
        with fc3:
            filter_rec = st.selectbox("Recommendation", ["All", "Advance", "Hold", "Reject"], key="filter_rec")
        st.markdown('</div>', unsafe_allow_html=True)

        # Apply filters
        filtered = []
        for r in jd_res:
            if r["score"] is not None and r["score"] < min_score:
                continue
            if filter_verdict != "All" and verdict_label(r["score"])[0] != filter_verdict:
                continue
            if filter_rec != "All" and extract_recommendation(r.get("output", "")).lower() != filter_rec.lower():
                continue
            filtered.append(r)

        if not filtered:
            st.info("No candidates match the current filters.")
        else:
            best    = filtered[0]
            verdict, vcls = verdict_label(best["score"])
            color   = score_color(best["score"])
            rec     = extract_recommendation(best.get("output", ""))
            st.markdown(
                f'<div class="card-hero">'
                f'<span class="best-badge">⭐ Best Match</span>&nbsp;&nbsp;{rec_badge(rec)}<br>'
                f'<span style="font-size:1.4rem;font-weight:800">{display_cv_name(best["cv_name"])}</span>'
                f'&nbsp;<span style="opacity:0.5;font-size:0.9rem">Score: <span style="color:{color};font-weight:700">{best["score"]}/10</span></span><br>'
                f'<span class="verdict {vcls}">{verdict}</span>'
                f'</div>', unsafe_allow_html=True
            )
            if best.get("summary"):
                st.info(shorten_text(best["summary"], 420))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("CVs Shown",     len(filtered))
            m2.metric("Best Score",    f"{max(scores):.1f}/10" if scores else "—")
            m3.metric("Avg Score",     f"{sum(scores)/len(scores):.1f}/10" if scores else "—")
            m4.metric("Top Candidate", display_cv_name(jd_res[0]["cv_name"]) if jd_res else "—")

            st.divider()
            st.markdown("**Score Comparison**")
            chart_data = pd.DataFrame({
                "Candidate": [display_cv_name(r["cv_name"]) for r in filtered],
                "Score /10": [r["score"] if r["score"] is not None else 0 for r in filtered],
            }).set_index("Candidate")
            st.bar_chart(chart_data, color="#3b82f6")

            st.divider()
            st.markdown(f"### Candidate Cards")

            for i, r in enumerate(filtered):
                verdict, vcls = verdict_label(r["score"])
                color         = score_color(r["score"])
                rec           = extract_recommendation(r.get("output", ""))
                rank_icon     = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"#{i+1}"))
                score_str     = f"{r['score']}/10" if r["score"] is not None else "N/A"
                mode_badge    = f'<span style="font-size:11px;opacity:0.45;padding:2px 8px;border-radius:10px;border:1px solid rgba(148,163,184,0.25)">{r.get("mode","")}</span>'

                with st.expander(f"{rank_icon}  {display_cv_name(r['cv_name'])}  —  {score_str}", expanded=(i == 0)):
                    left, right = st.columns([3, 1])

                    with left:
                        if i == 0:
                            st.markdown('<span class="best-badge">⭐ Best Match</span>', unsafe_allow_html=True)
                        hdr = f'<span class="verdict {vcls}">{verdict}</span> {mode_badge}'
                        if rec:
                            hdr += f'&nbsp;{rec_badge(rec)}'
                        st.markdown(hdr, unsafe_allow_html=True)
                        if r.get("summary"):
                            st.markdown(f"**Summary:** {r['summary']}")
                        if r.get("skills"):
                            tags = [s.strip() for s in re.split(r"[,;•\-\n]+", r["skills"]) if s.strip()][:8]
                            st.markdown("**Matching Skills:**")
                            st.markdown(" ".join(f'<span class="tag tag-s">{t}</span>' for t in tags), unsafe_allow_html=True)
                        if r.get("gaps"):
                            gtags = [g.strip() for g in re.split(r"[,;•\-\n]+", r["gaps"]) if g.strip()][:5]
                            st.markdown("**Gaps / Weak Areas:**")
                            st.markdown(" ".join(f'<span class="tag tag-g">{g}</span>' for g in gtags), unsafe_allow_html=True)

                    with right:
                        st.markdown(f'<div class="score-big" style="color:{color}">{r["score"] if r["score"] is not None else "—"}</div><div class="score-label">/10</div>', unsafe_allow_html=True)
                        if r["score"] is not None:
                            st.progress(float(r["score"]) / 10.0)

                    with st.expander("📋 Full AI Evaluation"):
                        clean = clean_ai_report(r.get("output", ""))
                        st.markdown(clean)
                        st.download_button(
                            "⬇️ Download this report",
                            clean,
                            file_name=f"talentlens_{Path(r['cv_name']).stem}_report.txt",
                            mime="text/plain",
                            key=f"download_report_{sel_jd}_{r['cv_name']}",
                        )

                    if r.get("rag_evidence"):
                        with st.expander("🔎 Explainability: Retrieved RAG Evidence"):
                            st.caption("These are the exact resume chunks retrieved by FAISS and sent to the LLM for this candidate.")
                            for chunk_i, chunk in enumerate(r.get("rag_evidence", []), start=1):
                                st.markdown(f"**Chunk {chunk_i}**")
                                st.text_area("", chunk, height=120, key=f"rag_{sel_jd}_{r['cv_name']}_{chunk_i}", label_visibility="collapsed")

                    if is_hr():
                        with st.expander("📝 HR Feedback / Score Correction"):
                            fb_key = f"{sel_jd}::{r['cv_name']}"
                            saved = st.session_state.hr_feedback.get(fb_key, {})
                            corrected_score = st.slider(
                                "Corrected score", 0.0, 10.0,
                                float(saved.get("corrected_score", r["score"] if r["score"] is not None else 0.0)),
                                0.5, key=f"score_{fb_key}"
                            )
                            corrected_rec = st.selectbox(
                                "Corrected recommendation", ["Advance", "Hold", "Reject"],
                                index=["Advance", "Hold", "Reject"].index(saved.get("corrected_recommendation", extract_recommendation(r.get("output", "")) or "Hold")),
                                key=f"rec_{fb_key}"
                            )
                            note = st.text_area("HR feedback note", value=saved.get("feedback_note", ""), key=f"note_{fb_key}")
                            if st.button("Save HR feedback", key=f"save_{fb_key}"):
                                st.session_state.hr_feedback[fb_key] = {
                                    "job_description": sel_jd,
                                    "candidate": r["cv_name"],
                                    "original_score": r["score"],
                                    "corrected_score": corrected_score,
                                    "corrected_recommendation": corrected_rec,
                                    "feedback_note": note,
                                }
                                st.success("HR feedback saved.")

            if is_hr() and st.session_state.hr_feedback:
                fb_df = pd.DataFrame(list(st.session_state.hr_feedback.values()))
                st.download_button("⬇️ Download HR Feedback CSV", fb_df.to_csv(index=False),
                                   file_name="talentlens_hr_feedback.csv", mime="text/csv", use_container_width=True)

            st.divider()
            rows = [{
                "Rank": i+1, "Candidate": display_cv_name(r["cv_name"]), "File": r["cv_name"],
                "Score /10": r["score"], "Verdict": verdict_label(r["score"])[0],
                "Recommendation": extract_recommendation(r.get("output", "")),
                "Mode": r.get("mode",""), "Summary": (r.get("summary") or "")[:200],
                "Skills": (r.get("skills") or "")[:200], "Gaps": (r.get("gaps") or "")[:200],
            } for i, r in enumerate(filtered)]
            st.download_button("⬇️ Download Results CSV", pd.DataFrame(rows).to_csv(index=False),
                               file_name=f"talentlens_{Path(sel_jd).stem}.csv", mime="text/csv", use_container_width=True)

            # ── EMAIL SHORTLISTING ────────────────────────────────────────────
            st.divider()
            if not is_hr():
                st.info("Email shortlisting is only available to HR users because it may reveal/contact candidates.")
            else:
                st.markdown("### ✉️ Email Shortlisting")
                st.caption("Tick candidates to shortlist, then generate a personalised interview invitation email for each.")

                job_title    = st.text_input("Job Title", placeholder="e.g. Senior Hardware Engineer")
                company_name = st.text_input("Company Name", placeholder="e.g. Acme Semiconductors")
                extra_notes  = st.text_area("Extra notes (optional)", placeholder="e.g. Remote role, 2 interview rounds, start date Q3", height=68)

                shortlisted = []
                st.markdown("**Select candidates to shortlist:**")
                for r in filtered:
                    checked = st.checkbox(
                        f"{display_cv_name(r['cv_name'])}  —  {r['score']}/10",
                        key=f"sl_{r['cv_name']}",
                    )
                    if checked:
                        shortlisted.append(r["cv_name"])

                if shortlisted:
                    st.markdown(f'<div class="info-box">✉️ {len(shortlisted)} candidate(s) selected for shortlisting.</div>', unsafe_allow_html=True)
                    if st.button("✉️ Generate Invitation Emails", type="primary", use_container_width=True):
                        if not job_title.strip():
                            st.warning("Please enter a job title first.")
                        else:
                            for cv_name in shortlisted:
                                with st.spinner(f"Generating email for {display_cv_name(cv_name)}…"):
                                    try:
                                        email_text = backend.generate_shortlist_email(
                                            groq_key=st.session_state.groq_key,
                                            candidate_name=display_cv_name(cv_name),
                                            job_title=job_title.strip(),
                                            company_name=company_name.strip() or "our company",
                                            extra_notes=extra_notes.strip(),
                                        )
                                        lines = email_text.split("\n")
                                        subject = ""
                                        body_lines = []
                                        for line in lines:
                                            if line.lower().startswith("subject:"):
                                                subject = line[8:].strip()
                                            else:
                                                body_lines.append(line)
                                        body = "\n".join(body_lines).strip()

                                        with st.expander(f"✉️ Email for {display_cv_name(cv_name)}", expanded=True):
                                            if subject:
                                                st.markdown(f"**Subject:** {subject}")
                                            st.text_area("Email body (copy this)", body, height=220, key=f"email_{cv_name}")
                                    except Exception as e:
                                        st.error(f"Failed to generate email for {display_cv_name(cv_name)}: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB — USER MANAGEMENT (HR ONLY)
# ══════════════════════════════════════════════════════════════════════════════
if tab_users is not None:
    with tab_users:
        st.markdown("## 👥 User Management")
        st.caption("Prototype account setup for role-based access control. HR/Admin users can create HR or Department accounts from the website.")

        users = load_users()

        st.markdown("### Existing Accounts")
        user_rows = [
            {"Username": username, "Role": info.get("role", "")}
            for username, info in users.items()
        ]
        st.dataframe(pd.DataFrame(user_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("### Create New Account")
        with st.form("create_account_form"):
            new_username = st.text_input("New username", placeholder="e.g. hiring_manager_1")
            new_password = st.text_input("Temporary password", type="password")
            new_role = st.selectbox("Role", ["HR", "DEPARTMENT"], format_func=lambda x: "HR / Admin" if x == "HR" else "Department User")
            create_submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

        if create_submitted:
            uname = new_username.strip()
            if not uname:
                st.error("Username cannot be empty.")
            elif not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", uname):
                st.error("Username must be 3-32 characters and only use letters, numbers, dot, underscore or dash.")
            elif uname in users:
                st.error("That username already exists.")
            elif len(new_password) < 4:
                st.error("Password must be at least 4 characters for this prototype.")
            else:
                users[uname] = {"password": new_password, "role": new_role}
                save_users(users)
                st.success(f"Created {new_role} account: {uname}")
                st.rerun()

        st.divider()
        st.markdown("### Manage Accounts")
        editable_users = list(users.keys())
        selected_user = st.selectbox("Select user", editable_users, key="manage_user_select")
        if selected_user:
            c1, c2 = st.columns(2)
            with c1:
                reset_password = st.text_input("New password", type="password", key="reset_password_input")
                if st.button("Reset Password", use_container_width=True):
                    if len(reset_password) < 4:
                        st.error("Password must be at least 4 characters.")
                    else:
                        users[selected_user]["password"] = reset_password
                        save_users(users)
                        st.success(f"Password reset for {selected_user}.")
                        st.rerun()
            with c2:
                new_role_for_user = st.selectbox(
                    "Change role", ["HR", "DEPARTMENT"],
                    index=0 if users[selected_user].get("role") == "HR" else 1,
                    key="role_change_select",
                    format_func=lambda x: "HR / Admin" if x == "HR" else "Department User",
                )
                if st.button("Update Role", use_container_width=True):
                    users[selected_user]["role"] = new_role_for_user
                    save_users(users)
                    st.success(f"Updated {selected_user} to {new_role_for_user}.")
                    st.rerun()

            if selected_user == st.session_state.username:
                st.info("You cannot delete the account you are currently logged in with.")
            else:
                if st.button("🗑️ Delete Selected Account", use_container_width=True):
                    users.pop(selected_user, None)
                    save_users(users)
                    st.warning(f"Deleted account: {selected_user}")
                    st.rerun()

        st.markdown(
            '<div class="warn-box">Prototype note: accounts are stored in <code>users.json</code>. '
            'For a real production system, use a database with hashed passwords and proper authentication.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
with tab_matrix:
    st.markdown("## 🗂️ Candidate × JD Score Matrix")
    st.caption("All candidates scored across all job descriptions at a glance.")

    if not st.session_state.analysis_done:
        st.markdown('<div class="warn-box">⚠️ Run screening first to see the comparison matrix.</div>', unsafe_allow_html=True)
    else:
        res_dict     = st.session_state.results
        jd_cols      = list(res_dict.keys())
        all_cv_names = []
        for rows in res_dict.values():
            for r in rows:
                if r["cv_name"] not in all_cv_names:
                    all_cv_names.append(r["cv_name"])

        lookup = {cv: {} for cv in all_cv_names}
        for jd_name, rows in res_dict.items():
            for r in rows:
                lookup[r["cv_name"]][jd_name] = r

        def avg_score(cv):
            sc = [lookup[cv][j]["score"] for j in jd_cols if lookup[cv].get(j) and lookup[cv][j]["score"] is not None]
            return sum(sc) / len(sc) if sc else -1

        sorted_cvs = sorted(all_cv_names, key=avg_score, reverse=True)

        header_cells = "".join(f"<th>{fmt_name(j)}</th>" for j in jd_cols) + "<th>Avg</th>"

        rows_html = ""
        for cv in sorted_cvs:
            sc_list = [lookup[cv][j]["score"] for j in jd_cols if lookup[cv].get(j) and lookup[cv][j]["score"] is not None]
            avg_val = sum(sc_list) / len(sc_list) if sc_list else None
            avg_str = f"{avg_val:.1f}" if avg_val is not None else "—"
            avg_c   = score_color(avg_val)

            row = f'<td><strong>{display_cv_name(cv)}</strong></td>'
            for jd in jd_cols:
                r = lookup[cv].get(jd)
                if r and r["score"] is not None:
                    s    = r["score"]
                    c    = score_color(s)
                    rec  = extract_recommendation(r.get("output", ""))
                    icon = {"advance": "✅", "hold": "⏸️", "reject": "❌"}.get(rec.lower(), "")
                    row += f'<td><span class="matrix-score" style="color:{c}">{s}</span><span style="font-size:10px;opacity:0.45">/10</span> {icon}</td>'
                else:
                    row += '<td style="opacity:0.3">—</td>'
            row += f'<td><span class="matrix-score" style="color:{avg_c}">{avg_str}</span></td>'
            rows_html += f"<tr>{row}</tr>"

        st.markdown(f"""
        <table class="matrix-table">
            <thead><tr><th>Candidate</th>{header_cells}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.divider()

        matrix_rows = []
        for cv in sorted_cvs:
            sc_list  = [lookup[cv][j]["score"] for j in jd_cols if lookup[cv].get(j) and lookup[cv][j]["score"] is not None]
            row_data = {"Candidate": display_cv_name(cv)}
            for jd in jd_cols:
                r = lookup[cv].get(jd)
                row_data[fmt_name(jd)] = r["score"] if r and r["score"] is not None else ""
            row_data["Average"] = f"{sum(sc_list)/len(sc_list):.1f}" if sc_list else ""
            matrix_rows.append(row_data)

        st.download_button("⬇️ Download Matrix CSV", pd.DataFrame(matrix_rows).to_csv(index=False),
                           file_name="talentlens_matrix.csv", mime="text/csv", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("## 💬 Ask About Candidates")
    st.caption("Ask anything — CV details, compare candidates, generate interview questions, etc.")
    st.caption("Chat now uses compact score summaries only, so it should be faster and less likely to freeze on Render free tier.")

    if not st.session_state.groq_key:
        st.markdown('<div class="warn-box">👈 Set your Groq API key in the sidebar first.</div>', unsafe_allow_html=True)
    else:
        def local_best_candidate_answer(question: str):
            qlow = question.lower().strip()
            if "best candidate" not in qlow or not st.session_state.results:
                return None
            all_rows = []
            for jd_name, rows in st.session_state.results.items():
                for row in rows:
                    if row.get("score") is not None:
                        all_rows.append((jd_name, row))
            if not all_rows:
                return None
            jd_name, best = max(all_rows, key=lambda x: x[1].get("score") or 0)
            name = display_cv_name(best.get("cv_name", "Candidate"))
            score = best.get("score")
            summary = shorten_text(best.get("summary") or best.get("output", ""), 350)
            gaps = shorten_text(best.get("gaps") or "No major gaps extracted.", 180)
            return (
                f"**Best candidate:** {name} for **{fmt_name(jd_name)}** with a score of **{score}/10**.\n\n"
                f"**Why:** {summary}\n\n"
                f"**Main gaps to note:** {gaps}"
            )

        def run_chat_llm(question: str):
            quick_local = local_best_candidate_answer(question)
            if quick_local:
                st.session_state.chat_history.append({"role": "assistant", "content": quick_local})
                return
            context = backend.build_chat_context(
                raw_jd_texts=st.session_state.raw_jd_texts,
                raw_cv_texts={},
                results=st.session_state.results,
            )
            recent_history = st.session_state.chat_history[-4:]
            try:
                reply = backend.ask_about_candidates(
                    groq_key=st.session_state.groq_key,
                    context=context,
                    question=question,
                    chat_history=recent_history,
                )
                reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL | re.IGNORECASE).strip()
            except Exception as e:
                err = str(e)
                if "413" in err or "rate_limit" in err.lower() or "tokens" in err.lower():
                    reply = "⚠️ Request too large for the free API tier. Try a more specific question, or clear the chat history."
                else:
                    reply = f"❌ Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        if not st.session_state.files_loaded:
            st.markdown('<div class="info-box">ℹ️ Upload files and run screening first for context-aware answers.</div>', unsafe_allow_html=True)

        st.markdown("**💡 Quick prompts:**")
        quick_prompts = [
            "Who is the best candidate and why?",
            "Compare the top 2 candidates side by side.",
            "Generate 5 interview questions for the top candidate.",
            "What skills are missing across all candidates?",
            "Which candidate would need the least training?",
            "Summarise all candidates in a table.",
        ]
        cols = st.columns(3)
        for idx, q in enumerate(quick_prompts):
            with cols[idx % 3]:
                if st.button(q, use_container_width=True, key=f"qp_{idx}"):
                    st.session_state["_queued_prompt"] = q

        # Process any queued prompt (from quick buttons or elsewhere)
        if st.session_state.get("_queued_prompt"):
            q = st.session_state.pop("_queued_prompt")
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    run_chat_llm(q)
                st.markdown(st.session_state.chat_history[-1]["content"])

        if st.session_state.chat_history:
            st.divider()

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_msg = st.chat_input("Ask about any candidate, skill gap, or comparison…")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    run_chat_llm(user_msg)
                st.markdown(st.session_state.chat_history[-1]["content"])

        if st.session_state.chat_history:
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()
