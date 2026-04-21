"""
TalentLens — Streamlit frontend
All AI/RAG/screening logic lives in backend.py
This file only handles the UI.
"""

import re
import time
import streamlit as st
import pandas as pd
from pathlib import Path

import backend  # ← the backend module does all the heavy lifting

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalentLens", page_icon="🔍", layout="wide")

# ─── THEME ────────────────────────────────────────────────────────────────────
# Streamlit has its own dark/light theme toggle (Settings gear ⚙ → Theme).
# The CSS below is designed to look good in BOTH modes by using CSS variables
# and only overriding the things Streamlit gets wrong.
st.markdown("""
<style>
    /* ── Sidebar — always a soft pastel blue regardless of theme ── */
    [data-testid="stSidebar"] {
        background: #dbeafe !important;
    }
    [data-testid="stSidebar"] * {
        color: #1e3a8a !important;
    }
    [data-testid="stSidebar"] .stTextInput input {
        background: #ffffff !important;
        border: 1px solid #93c5fd !important;
        color: #1e293b !important;
        border-radius: 10px;
    }
    [data-testid="stSidebar"] .stTextInput input::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] .stTextInput > div {
        background: transparent !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #bfdbfe !important;
        color: #1e3a8a !important;
        border: 1px solid #93c5fd !important;
        border-radius: 10px;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #93c5fd !important;
    }
    [data-testid="stSidebar"] .element-container { margin-bottom: 0.55rem !important; }
    [data-testid="stSidebar"] .stDivider { margin: 0.9rem 0 !important; }

    /* ── Upload blocks ── */
    [data-testid="stFileUploader"] {
        background: #fdf2f8 !important;
        border: 1px solid #f5c2d7 !important;
        border-radius: 14px !important;
        padding: 10px !important;
    }
    [data-testid="stFileUploader"] section {
        background: #fff7fb !important;
        border-radius: 12px !important;
        border: 1px dashed #e9a8c3 !important;
    }
    [data-testid="stFileUploader"] button {
        background: #fde2e7 !important;
        color: #9d174d !important;
        border: 1px solid #f8b4c8 !important;
        border-radius: 10px !important;
    }
    [data-testid="stFileUploader"] button:hover {
        background: #fbcfe8 !important;
        color: #831843 !important;
    }

    /* ── Tabs ── */
    div[data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 2px solid #93c5fd;
    }
    button[data-baseweb="tab"] {
        height: auto; white-space: nowrap;
        padding: 0.6rem 1.2rem;
        border-radius: 10px 10px 0 0 !important;
        font-weight: 500;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #3b82f6 !important;
        color: #1d4ed8 !important;
        font-weight: 700 !important;
    }

    /* ── Cards — use semi-transparent so they work in both light and dark ── */
    .card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(147,197,253,0.4);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }
    .card-top { border-left: 4px solid #3b82f6; }

    /* ── Badges ── */
    .best-badge {
        background: #3b82f6; color: #fff; font-size: 11px; font-weight: 600;
        padding: 3px 12px; border-radius: 20px;
        display: inline-block; margin-bottom: 8px;
    }
    .verdict {
        font-size: 12px; padding: 3px 12px; border-radius: 20px;
        display: inline-block; margin: 4px 0; font-weight: 500;
    }
    .v-strong  { background: #dbeafe; color: #1e40af; }
    .v-good    { background: #dcfce7; color: #166534; }
    .v-partial { background: #fef9c3; color: #854d0e; }
    .v-weak    { background: #fee2e2; color: #991b1b; }

    /* ── Tags ── */
    .tag {
        font-size: 11px; padding: 3px 10px; border-radius: 20px;
        margin: 2px; display: inline-block;
    }
    .tag-s { background: #dbeafe; color: #1e40af; }
    .tag-g { background: #fee2e2; color: #991b1b; }

    /* ── Score display ── */
    .score-big  { font-size: 2.4rem; font-weight: 700; line-height: 1; text-align: center; }
    .score-label { font-size: 0.95rem; text-align: center; opacity: 0.6; }

    /* ── File pill ── */
    .file-pill {
        display: flex; align-items: center; gap: 10px;
        background: rgba(219,234,254,0.3);
        border: 1px solid rgba(147,197,253,0.5);
        border-radius: 8px; padding: 6px 12px;
        margin-bottom: 6px; font-size: 13px;
    }
    .file-pill-icon { font-size: 16px; }
    .file-pill-name { flex: 1; font-weight: 500; }
    .file-pill-size { font-size: 11px; opacity: 0.6; }

    /* ── Status dots ── */
    .status-dot {
        display: inline-block; width: 8px; height: 8px;
        border-radius: 50%; margin-right: 6px;
    }
    .dot-green { background: #4ade80; }
    .dot-red   { background: #f87171; }
    .dot-amber { background: #fbbf24; }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(219,234,254,0.2) !important;
        border: 1px solid rgba(147,197,253,0.4) !important;
        border-radius: 10px !important;
        padding: 0.8rem 1rem !important;
    }

    /* ── Main action buttons ── */
    .stButton > button[kind="primary"] {
        background: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2563eb !important;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.15s;
    }

    /* ── Reset button ── */
    .reset-btn > button {
        background: #fee2e2 !important;
        color: #991b1b !important;
        border: 1px solid #fca5a5 !important;
    }
    .reset-btn > button:hover {
        background: #fecaca !important;
    }

    /* ── Info / warn / success boxes — semi-transparent so dark mode works ── */
    .info-box {
        background: rgba(219,234,254,0.35);
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px; font-size: 13px;
        color: #1e40af; margin-bottom: 12px;
    }
    .warn-box {
        background: rgba(254,243,199,0.45);
        border-left: 4px solid #d97706;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px; font-size: 13px;
        color: #92400e; margin-bottom: 12px;
    }
    .success-box {
        background: rgba(220,252,231,0.45);
        border-left: 4px solid #16a34a;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px; font-size: 13px;
        color: #166534; margin-bottom: 12px;
    }

    /* ── Dark mode overrides — fix boxes/cards that go invisible ── */
    @media (prefers-color-scheme: dark) {
        .info-box    { color: #93c5fd !important; }
        .warn-box    { color: #fcd34d !important; }
        .success-box { color: #86efac !important; }
        .v-strong  { background: #1e3a8a; color: #bfdbfe; }
        .v-good    { background: #14532d; color: #bbf7d0; }
        .v-partial { background: #713f12; color: #fef08a; }
        .v-weak    { background: #7f1d1d; color: #fecaca; }
        .tag-s { background: #1e3a8a; color: #bfdbfe; }
        .tag-g { background: #7f1d1d; color: #fecaca; }
        .best-badge { background: #1d4ed8; }
    }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
REQUEST_DELAY = 1

# ─── UI HELPERS (display only — no business logic) ────────────────────────────
def fmt_name(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ")

def verdict_label(score):
    if score is None: return "Unknown",      "v-weak"
    if score >= 8:    return "Strong Match",  "v-strong"
    if score >= 6:    return "Good Match",    "v-good"
    if score >= 4:    return "Partial Match", "v-partial"
    return "Weak Match", "v-weak"

def score_color(score):
    if score is None: return "#9a9895"
    if score >= 8:    return "#1d4ed8"
    if score >= 6:    return "#16a34a"
    if score >= 4:    return "#d97706"
    return "#dc2626"

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
_defaults = {
    "groq_key":      "",
    "jd_files":      {},   # name → preprocessed text
    "cv_files":      {},   # name → preprocessed text
    "raw_jd_texts":  {},   # name → raw text
    "raw_cv_texts":  {},   # name → raw text
    "results":       {},
    "chat_history":  [],
    "analysis_done": False,
    "files_loaded":  False,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reset_all():
    for k in ["jd_files", "cv_files", "raw_jd_texts", "raw_cv_texts", "results", "chat_history"]:
        st.session_state[k] = {} if k != "chat_history" else []
    st.session_state["analysis_done"] = False
    st.session_state["files_loaded"]  = False
    st.rerun()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 TalentLens")
    st.caption("Qwen3-32B via Groq · FAISS RAG")
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
        if st.session_state.groq_key:
            st.markdown('<span class="status-dot dot-green"></span>OK', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-dot dot-red"></span>Not set', unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📁 Upload Files")

    jd_uploads = st.file_uploader(
        "Job Descriptions (PDF / TXT / DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload one or more job description files."
    )
    cv_uploads = st.file_uploader(
        "Candidate CVs (PDF / TXT / DOCX)",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload all candidate CV files you want to screen."
    )

    st.divider()

    # ── Load & Preprocess — calls backend ──
    if st.button("⚙️ Load & Preprocess", use_container_width=True, type="primary"):
        if not jd_uploads or not cv_uploads:
            st.warning("⚠️ Upload at least one JD and one CV first.")
        else:
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
                n_jd = len(loaded["jd_files"])
                n_cv = len(loaded["cv_files"])
                st.success(f"✅ {n_jd} JD(s) · {n_cv} CV(s) loaded")

    # Status indicator
    if st.session_state.files_loaded:
        n_jd = len(st.session_state.jd_files)
        n_cv = len(st.session_state.cv_files)
        st.markdown(
            f'<div style="font-size:12px;margin-top:4px">'
            f'<span class="status-dot dot-green"></span>'
            f'Loaded: {n_jd} JD · {n_cv} CVs</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="font-size:12px;margin-top:4px">'
            '<span class="status-dot dot-amber"></span>'
            'No files loaded yet</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown("### 🗑️ Reset")
    st.caption(
        "Pressing X on a file only removes it from the uploader widget — "
        "it does **not** clear loaded data. Use this to fully wipe everything."
    )
    with st.container():
        st.markdown('<div class="reset-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Reset All Data", use_container_width=True):
            reset_all()
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("TalentLens v4 · backend.py · FAISS RAG · Qwen3-32B")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_run, tab_res, tab_chat = st.tabs(["🚀 Run Screening", "📊 Results", "💬 Chat"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUN SCREENING
# ══════════════════════════════════════════════════════════════════════════════
with tab_run:
    st.markdown("## CV Screening")

    if not st.session_state.groq_key:
        st.markdown(
            '<div class="warn-box">👈 <strong>Step 1:</strong> Add your Groq API key in the sidebar.</div>',
            unsafe_allow_html=True
        )
    elif not st.session_state.files_loaded:
        st.markdown(
            '<div class="info-box">👈 <strong>Step 2:</strong> Upload your JD and CV files, '
            'then click <strong>Load &amp; Preprocess</strong>.</div>',
            unsafe_allow_html=True
        )
    else:
        # File summary pills
        col_jd, col_cv = st.columns(2)
        with col_jd:
            st.markdown("**📄 Job Descriptions**")
            for n in st.session_state.jd_files:
                st.markdown(
                    f'<div class="file-pill">'
                    f'<span class="file-pill-icon">📄</span>'
                    f'<span class="file-pill-name">{fmt_name(n)}</span>'
                    f'<span class="file-pill-size">{len(st.session_state.raw_jd_texts[n]):,} chars</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        with col_cv:
            st.markdown("**👤 Candidate CVs**")
            for n in st.session_state.cv_files:
                st.markdown(
                    f'<div class="file-pill">'
                    f'<span class="file-pill-icon">👤</span>'
                    f'<span class="file-pill-name">{fmt_name(n)}</span>'
                    f'<span class="file-pill-size">{len(st.session_state.raw_cv_texts[n]):,} chars</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.divider()

        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            use_rag = st.checkbox(
                "✅ Use RAG (recommended)", value=True,
                help="Retrieves the most relevant CV sections before scoring. More accurate."
            )
        with col_opt2:
            delay = st.number_input(
                "Delay between calls (s)",
                min_value=0, max_value=5, value=REQUEST_DELAY, step=1,
                help="Increase if you hit 429 rate-limit errors."
            )

        total_pairs = len(st.session_state.jd_files) * len(st.session_state.cv_files)
        st.markdown(
            f'<div class="info-box">Ready to screen '
            f'<strong>{len(st.session_state.cv_files)} candidate(s)</strong> against '
            f'<strong>{len(st.session_state.jd_files)} job description(s)</strong> '
            f'— {total_pairs} API call(s) total.</div>',
            unsafe_allow_html=True
        )

        if st.button("▶️ Run Screening", type="primary", use_container_width=True):
            prog = st.progress(0, text="Starting…")

            def progress_cb(step, total, jd_name, cv_name):
                prog.progress(step / total,
                              text=f"({step}/{total}) Scoring {fmt_name(cv_name)} for {fmt_name(jd_name)}…")

            try:
                # ── Single call into backend — all logic lives there ──
                all_results = backend.screen_candidates(
                    jd_files=st.session_state.jd_files,
                    cv_files=st.session_state.cv_files,
                    groq_key=st.session_state.groq_key,
                    use_rag=use_rag,
                    delay=int(delay),
                    progress_callback=progress_cb,
                )
                prog.empty()
                st.session_state.results      = all_results
                st.session_state.analysis_done = True

                first_jd = list(all_results.keys())[0]
                best     = all_results[first_jd][0]
                st.markdown(
                    f'<div class="success-box">✅ Screening complete! '
                    f'Top candidate: <strong>{fmt_name(best["cv_name"])}</strong> '
                    f'— Score: <strong>{best["score"]}/10</strong>. '
                    f'Open the <strong>Results</strong> tab to see full rankings.</div>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                prog.empty()
                st.error(f"❌ Screening failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_res:
    st.markdown("## Results")

    if not st.session_state.analysis_done:
        st.markdown(
            '<div class="warn-box">⚠️ No results yet — run screening in the '
            '<strong>Run Screening</strong> tab first.</div>',
            unsafe_allow_html=True
        )
    else:
        res_dict = st.session_state.results
        jd_names = list(res_dict.keys())

        sel_jd = st.selectbox(
            "Select Job Description", jd_names, format_func=fmt_name
        ) if len(jd_names) > 1 else jd_names[0]

        jd_res = res_dict[sel_jd]
        scores = [r["score"] for r in jd_res if r["score"] is not None]

        # Top candidate banner
        if jd_res:
            best = jd_res[0]
            verdict, vcls = verdict_label(best["score"])
            color = score_color(best["score"])
            st.markdown(
                f'<div class="card card-top">'
                f'<span class="best-badge">⭐ Best Match</span><br>'
                f'<span style="font-size:1.3rem;font-weight:700">{fmt_name(best["cv_name"])}</span>'
                f'<span style="opacity:0.6;margin-left:10px">Score: '
                f'<span style="color:{color};font-weight:700">'
                f'{best["score"] if best["score"] is not None else "N/A"}/10</span></span><br>'
                f'<span class="verdict {vcls}">{verdict}</span>'
                f'{"<br><small>" + best["summary"] + "</small>" if best.get("summary") else ""}'
                f'</div>',
                unsafe_allow_html=True
            )

        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CVs Screened",  len(jd_res))
        m2.metric("Best Score",    f"{max(scores):.1f}/10"               if scores else "—")
        m3.metric("Avg Score",     f"{sum(scores)/len(scores):.1f}/10"   if scores else "—")
        m4.metric("Top Candidate", fmt_name(jd_res[0]["cv_name"])        if jd_res else "—")

        st.divider()

        # Bar chart
        st.markdown("**Score Comparison**")
        chart_data = pd.DataFrame({
            "Candidate": [fmt_name(r["cv_name"]) for r in jd_res],
            "Score /10": [r["score"] if r["score"] is not None else 0 for r in jd_res],
        }).set_index("Candidate")
        st.bar_chart(chart_data, color="#3b82f6")

        st.divider()
        st.markdown("### Candidate Cards")

        for i, r in enumerate(jd_res):
            verdict, vcls = verdict_label(r["score"])
            color         = score_color(r["score"])
            rank_icon     = "🥇" if i == 0 else ("🥈" if i == 1 else ("🥉" if i == 2 else f"#{i+1}"))
            score_str     = f"{r['score']}/10" if r["score"] is not None else "N/A"
            mode_badge    = (f' <span style="font-size:11px;opacity:0.6;'
                             f'padding:2px 8px;border-radius:10px;'
                             f'border:1px solid rgba(100,100,100,0.3)">'
                             f'{r.get("mode","")}</span>')

            with st.expander(f"{rank_icon}  {fmt_name(r['cv_name'])}  —  {score_str}", expanded=(i == 0)):
                left, right = st.columns([3, 1])

                with left:
                    if i == 0:
                        st.markdown('<span class="best-badge">⭐ Best Match</span>', unsafe_allow_html=True)
                    st.markdown(
                        f'<span class="verdict {vcls}">{verdict}</span>{mode_badge}',
                        unsafe_allow_html=True
                    )
                    if r.get("summary"):
                        st.markdown(f"**Summary:** {r['summary']}")
                    if r.get("skills"):
                        tags = [s.strip() for s in re.split(r"[,;•\-\n]+", r["skills"]) if s.strip()][:8]
                        st.markdown("**Matching Skills:**")
                        st.markdown(" ".join(f'<span class="tag tag-s">{t}</span>' for t in tags),
                                    unsafe_allow_html=True)
                    if r.get("gaps"):
                        gtags = [g.strip() for g in re.split(r"[,;•\-\n]+", r["gaps"]) if g.strip()][:5]
                        st.markdown("**Gaps / Weak Areas:**")
                        st.markdown(" ".join(f'<span class="tag tag-g">{g}</span>' for g in gtags),
                                    unsafe_allow_html=True)

                with right:
                    st.markdown(
                        f'<div class="score-big" style="color:{color}">'
                        f'{r["score"] if r["score"] is not None else "—"}'
                        f'</div><div class="score-label">/10</div>',
                        unsafe_allow_html=True
                    )
                    if r["score"] is not None:
                        st.progress(float(r["score"]) / 10.0)

                with st.expander("📋 Full AI Evaluation"):
                    clean = re.sub(r"<think>.*?</think>", "", r["output"],
                                   flags=re.DOTALL | re.IGNORECASE).strip()
                    st.markdown(clean)

        st.divider()

        # Download CSV
        rows = [{
            "Rank":      i + 1,
            "Candidate": fmt_name(r["cv_name"]),
            "File":      r["cv_name"],
            "Score /10": r["score"],
            "Verdict":   verdict_label(r["score"])[0],
            "Mode":      r.get("mode", ""),
            "Summary":   (r.get("summary") or "")[:200],
            "Skills":    (r.get("skills")  or "")[:200],
            "Gaps":      (r.get("gaps")    or "")[:200],
        } for i, r in enumerate(jd_res)]

        st.download_button(
            "⬇️ Download Full Results CSV",
            pd.DataFrame(rows).to_csv(index=False),
            file_name=f"talentlens_{Path(sel_jd).stem}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("## 💬 Ask About Candidates")
    st.caption("Ask anything — CV details, compare candidates, generate interview questions, etc.")

    if not st.session_state.groq_key:
        st.markdown(
            '<div class="warn-box">👈 Set your Groq API key in the sidebar first.</div>',
            unsafe_allow_html=True
        )
    else:
        # ── Central LLM call — delegates entirely to backend ──
        def run_chat_llm(question: str):
            """Build compact context via backend, call LLM, append reply to history."""
            context = backend.build_chat_context(
                raw_jd_texts=st.session_state.raw_jd_texts,
                raw_cv_texts=st.session_state.raw_cv_texts,
                results=st.session_state.results,
            )
            # Only pass last 6 turns to avoid token bloat
            recent_history = st.session_state.chat_history[-6:]
            try:
                reply = backend.ask_about_candidates(
                    groq_key=st.session_state.groq_key,
                    context=context,
                    question=question,
                    chat_history=recent_history,
                )
                reply = re.sub(r"<think>.*?</think>", "", reply,
                               flags=re.DOTALL | re.IGNORECASE).strip()
            except Exception as e:
                err = str(e)
                if "413" in err or "rate_limit" in err.lower() or "tokens" in err.lower():
                    reply = (
                        "⚠️ Request too large for the free API tier. "
                        "Try a more specific question, or clear the chat history."
                    )
                else:
                    reply = f"❌ Error: {e}"
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        # Quick prompts (only when chat is empty)
        if not st.session_state.chat_history:
            if not st.session_state.files_loaded:
                st.markdown(
                    '<div class="info-box">ℹ️ Upload files and run screening first '
                    'to get context-aware answers. You can still chat without files.</div>',
                    unsafe_allow_html=True
                )
            st.markdown("**💡 Try asking:**")
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
                        st.session_state.chat_history.append({"role": "user", "content": q})
                        run_chat_llm(q)
                        st.rerun()

        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Typed input
        user_msg = st.chat_input("Ask about any candidate, skill gap, or comparison…")
        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    run_chat_llm(user_msg)
                st.markdown(st.session_state.chat_history[-1]["content"])

        # Clear chat
        if st.session_state.chat_history:
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=False):
                st.session_state.chat_history = []
                st.rerun()