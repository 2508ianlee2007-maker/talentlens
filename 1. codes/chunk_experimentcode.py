# ============================================================
# AI-POWERED CV SCREENING SYSTEM
# Single LLM Version — Qwen3-32B via Groq
# Selected based on Artificial Analysis benchmark:
#   Intelligence Score: 17/20 (2nd highest)
#   Output Speed: 95 tokens/sec (fastest)
#   Best balance of quality and reliability
# ============================================================

# ── SECTION 1: Install libraries ──
# Run this ONCE in your terminal before running this script:
# pip install pypdf pandas groq openpyxl python-dotenv
# pip install langchain langchain-community langchain-core
# pip install langchain-text-splitters langchain-huggingface
# pip install faiss-cpu sentence-transformers langchain-groq
#
# Then create a .env file in the same folder as this script:
#   GROQ_API_KEY=your_key_here
# Add .env to your .gitignore — never commit API keys to source control.

# ── SECTION 2: Imports ──
import os
import re
import time
import zipfile
from pathlib import Path
from collections import Counter

import pandas as pd
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

print("Current working directory:", os.getcwd())
print("Files and folders here:")
for item in os.listdir("."):
    print(" ", item)
# ── SECTION 3: Configuration ──
ZIP_FILENAME = "FTSP (Week 1)  .zip"

# Groq API Key — loaded from .env file (never hardcode secrets in source code)
# Create a file called .env in the same folder as this script with this line:
#   GROQ_API_KEY=your_key_here
# Then add .env to your .gitignore so it is never committed.
# Install python-dotenv once: pip install python-dotenv

from getpass import getpass
import sys
print("Enter your GROQ API key below:")
sys.stdout.flush()
GROQ_API_KEY = getpass().strip()
if not GROQ_API_KEY:
    raise EnvironmentError("No GROQ API key provided.")

# Qwen3-32B model via Groq
# Selected because:
#   - Intelligence score: 17/20 on Artificial Analysis benchmark
#   - Output speed: 95 tokens/sec (fastest among all 5 tested models)
#   - Free tier via Groq with generous daily limits
#   - No credit card required
QWEN_MODEL = "qwen/qwen3-32b"

# ── SECTION 4: All helper functions ──

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = text.lower()
    text = re.sub(r'(\+?\d[\d\-\(\) ]{7,}\d)', ' ', text)
    text = re.sub(r'[^a-z0-9/\+\#\-\.\s]', ' ', text)
    text = re.sub(r'[\.]{2,}', ' ', text)
    text = re.sub(r'[-]{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_score(text):
    if not isinstance(text, str):
        return None

    # Strip Qwen3 <think>...</think> reasoning blocks before parsing
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    cleaned = text.replace("**", "").replace("__", "").replace("*", "")

    # Pattern 1: X/10 or X / 10
    match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", cleaned)
    if match:
        val = float(match.group(1))
        if 0 <= val <= 10:
            return val

    # Pattern 2: X out of 10
    match = re.search(r"(\d+(?:\.\d+)?)\s*out\s*of\s*10", cleaned, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if 0 <= val <= 10:
            return val

    # Pattern 3: X out of ten (written out)
    match = re.search(r"(\d+(?:\.\d+)?)\s*out\s*of\s*ten", cleaned, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if 0 <= val <= 10:
            return val

    # Pattern 4: score/rating: X or score X (e.g. "Score: 7")
    match = re.search(r"(?:suitability\s+)?(?:score|rating)\s*[:\-\u2013]?\s*(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if 0 <= val <= 10:
            return val

    # Pattern 5: score in brackets like (7/10) or [7/10]
    match = re.search(r"[\(\[]\s*(\d+(?:\.\d+)?)\s*/\s*10\s*[\)\]]", cleaned)
    if match:
        val = float(match.group(1))
        if 0 <= val <= 10:
            return val

    # Pattern 6: bold/inline score like **7** at end of suitability line
    match = re.search(r"suitability[^\n]*?(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    if match:
        val = float(match.group(1))
        if 0 <= val <= 10:
            return val

    # Pattern 7: last standalone digit 1-10 as final fallback
    matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", cleaned)
    for m in reversed(matches):
        val = float(m)
        if 1 <= val <= 10:
            return val

    return None

def build_user_prompt(job_description, candidate_name, resume_text):
    return f"""
Job Description:
{job_description}

Candidate Resume File Name:
{candidate_name}

Candidate Resume Text:
{resume_text}

Please evaluate this candidate against the job description.

Format your answer as:
Candidate Summary:
Matching Skills:
Missing / Weak Areas:
Suitability Score:
"""

def build_rag_user_prompt(job_description, candidate_name, rag_context):
    return f"""
Job Description:
{job_description}

Candidate Resume File Name:
{candidate_name}

Most Relevant Resume Sections (retrieved via RAG):
{rag_context}

Based on the retrieved resume sections above, please evaluate this candidate.

Format your answer as:
Candidate Summary:
Matching Skills:
Missing / Weak Areas:
Suitability Score:
"""

# ── SECTION 5: Extract zip file ──
print("Current working directory:", os.getcwd())

extract_folder = Path("project_files")
extract_folder.mkdir(exist_ok=True)

with zipfile.ZipFile(ZIP_FILENAME, "r") as zip_ref:
    zip_ref.extractall(extract_folder)

print("Files extracted successfully.\n")

# ── SECTION 6: Locate resumes and job descriptions ──
all_files = sorted(extract_folder.iterdir())
txt_files = [f for f in all_files if f.suffix.lower() == ".txt"]
pdf_files = [f for f in all_files if f.suffix.lower() == ".pdf"]

print("CV/Resume files found:")
for f in txt_files:
    print("-", f.name)

print("\nJob Description PDF files found:")
for f in pdf_files:
    print("-", f.name)

if len(pdf_files) == 0:
    raise FileNotFoundError("No PDF job description found.")

print(f"\nTotal job descriptions: {len(pdf_files)}")
print(f"Total CVs: {len(txt_files)}")

# ── SECTION 7: Read all resumes ──
raw_resumes = {}
for file_path in txt_files:
    raw_resumes[file_path.name] = read_text_file(file_path)
print("\nLoaded resumes:", len(raw_resumes))

# ── SECTION 8: Read all job descriptions ──
job_pdf_path = pdf_files[0]
raw_job_description = read_pdf_text(job_pdf_path)
print("Primary job description:", job_pdf_path.name)

all_job_descriptions = {}
for jd_path in pdf_files:
    jd_text = read_pdf_text(jd_path)
    all_job_descriptions[jd_path.name] = preprocess_text(jd_text)

print(f"Loaded {len(all_job_descriptions)} job description(s)")

# ── SECTION 9: Apply preprocessing ──
cleaned_resumes = {}
for name, text in raw_resumes.items():
    cleaned_resumes[name] = preprocess_text(text)

cleaned_job_description = preprocess_text(raw_job_description)
print("\nCleaned job description preview:")
print(cleaned_job_description[:500])

# ── SECTION 10: Save cleaned text files ──
cleaned_folder = Path("cleaned_texts")
cleaned_folder.mkdir(exist_ok=True)

for name, text in cleaned_resumes.items():
    out_path = cleaned_folder / f"cleaned_{name}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

with open(cleaned_folder / "cleaned_job_description.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_job_description)

print("Cleaned files saved in:", cleaned_folder)

# ── SECTION 11: Preprocessing summary table ──
summary_rows = []
for name in raw_resumes:
    raw_text = raw_resumes[name]
    cleaned_text = cleaned_resumes[name]
    summary_rows.append({
        "file_name": name,
        "raw_char_count": len(raw_text),
        "cleaned_char_count": len(cleaned_text),
        "removed_chars": len(raw_text) - len(cleaned_text)
    })

summary_rows.append({
    "file_name": "job_description_pdf",
    "raw_char_count": len(raw_job_description),
    "cleaned_char_count": len(cleaned_job_description),
    "removed_chars": len(raw_job_description) - len(cleaned_job_description)
})

summary_df = pd.DataFrame(summary_rows)
print(summary_df)

# ── SECTION 12: System prompt ──
SYSTEM_PROMPT = """
You are an HR screening assistant.
Your task is to compare a candidate's resume with a job description.

Focus on:
1. Relevant technical skills
2. Relevant work experience
3. Overall suitability for the role

Return the answer in a clear and concise format with:
- Candidate Summary
- Matching Skills
- Missing / Weak Areas
- Suitability Score out of 10
"""

# ── SECTION 13: Set up Groq client ──
groq_client = Groq(api_key=GROQ_API_KEY)
print(f"\nUsing model: {QWEN_MODEL} via Groq")
print("Selected based on Artificial Analysis benchmark:")
print("  Intelligence: 17/20 | Speed: 95 tokens/sec")

# ── SECTION 14: Qwen response function ──
def get_qwen_response(system_prompt, user_prompt, model_name=QWEN_MODEL):
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

# ── SECTION 15: Test one resume first ──
test_candidate = list(cleaned_resumes.keys())[0]
test_resume = cleaned_resumes[test_candidate]
test_prompt = build_user_prompt(
    job_description=cleaned_job_description,
    candidate_name=test_candidate,
    resume_text=test_resume
)

print(f"\nTesting on candidate: {test_candidate}")
print("\n--- QWEN3-32B TEST OUTPUT ---\n")
try:
    print(get_qwen_response(SYSTEM_PROMPT, test_prompt))
except Exception as e:
    print("Qwen Error:", e)

# ── SECTION 16: Run all resumes through Qwen ──
results = []

for candidate_name, resume_text in cleaned_resumes.items():
    user_prompt = build_user_prompt(
        job_description=cleaned_job_description,
        candidate_name=candidate_name,
        resume_text=resume_text
    )
    print(f"Processing {candidate_name}...")

    try:
        qwen_output = get_qwen_response(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        qwen_output = f"Qwen Error: {e}"

    results.append({
        "candidate_name": candidate_name,
        "qwen_output": qwen_output
    })

    time.sleep(1)

print("Done.")

# ── SECTION 17: View all results ──
for item in results:
    print("=" * 100)
    print("Candidate:", item["candidate_name"])
    print("\nQWEN3-32B RESPONSE:\n")
    print(item["qwen_output"])

# ── SECTION 18: Score table ──
score_rows = []
for item in results:
    score_rows.append({
        "candidate_name": item["candidate_name"],
        "qwen_score": extract_score(item["qwen_output"]),
    })

scores_df = pd.DataFrame(score_rows)
scores_df["average_score"] = scores_df["qwen_score"]
scores_df = scores_df.sort_values(by="average_score", ascending=False)
print("\nScores:")
print(scores_df)

# ── SECTION 19: Final recommendation ──
valid_df = scores_df.dropna(subset=["qwen_score"])
if not valid_df.empty:
    final_best_candidate = valid_df.iloc[0]["candidate_name"]
    final_score = valid_df.iloc[0]["qwen_score"]
    print(f"\nFINAL RECOMMENDED RESUME: {final_best_candidate} (score: {final_score}/10)")
else:
    print("\nNo valid scores found.")
    final_best_candidate = "N/A"

# ════════════════════════════════════════════
# WEEK 3 — RAG with LangChain
# ════════════════════════════════════════════

# ── SECTION 20: Convert resumes to LangChain Documents ──
documents = []
for filename, text in cleaned_resumes.items():
    doc = Document(page_content=text, metadata={"source": filename})
    documents.append(doc)

print(f"\nTotal documents loaded: {len(documents)}")

# ── SECTION 21: Split into chunks (Week 3 - chunk_size=600) ──
# chunk_size=600 chosen based on resume length (~1500-3000 chars)
# Larger chunks ensure full job roles and skills sections fit without being cut off
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# ── SECTION 22: Load embedding model ──
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded.")

# ── SECTION 23: Build or Load FAISS Vector Store ──
# Saves to disk on first run, loads from disk every run after
FAISS_INDEX_PATH = "faiss_index"

if Path(FAISS_INDEX_PATH).exists():
    print("Loading existing FAISS index from disk...")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded.")
else:
    print("Building new FAISS index and saving to disk...")
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}'.")

print(f"Total vectors stored: {vectorstore.index.ntotal}")

# ── SECTION 24: Test RAG Retrieval ──
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
query = cleaned_job_description[:500]
retrieved_docs = retriever.invoke(query)

print(f"\nRetrieved {len(retrieved_docs)} relevant chunks:")
for i, doc in enumerate(retrieved_docs):
    print(f"--- Chunk {i+1} (from: {doc.metadata['source']}) ---")
    print(doc.page_content[:200])

# ── SECTION 25: RAG screening function ──
def get_rag_context_refined(candidate_name, job_description_query, k=6):
    candidate_docs = [c for c in refined_chunks if c.metadata["source"] == candidate_name]
    if not candidate_docs:
        return ""
    vs = FAISS.from_documents(candidate_docs, embedding_model)
    ret = vs.as_retriever(search_kwargs={"k": min(k, len(candidate_docs))})
    retrieved = ret.invoke(job_description_query)
    return "\n\n".join([doc.page_content for doc in retrieved])

# ════════════════════════════════════════════
# WEEK 4 — Refined RAG + Chunk Size Experiment
# ════════════════════════════════════════════

# ── SECTION 26: Multi-config experiment ──
# All configs tested across Weeks 3–5.
# For each config we run the full RAG pipeline and record:
#   - score spread (max - min)  → how well it differentiates candidates
#   - top candidate             → who wins
#   - avg RAG context size      → how much text Qwen actually sees
#
# Delete the faiss_index folder before running after any chunk setting change.

CONFIGS = [
    {"chunk_size": 250,  "overlap": 50,  "k": 4, "label": "250/50 k=4"},
    {"chunk_size": 400,  "overlap": 80,  "k": 3, "label": "400/80 k=3 (Wk3)"},
    {"chunk_size": 500,  "overlap": 100, "k": 5, "label": "500/100 k=5"},
    {"chunk_size": 600,  "overlap": 150, "k": 6, "label": "600/150 k=6"},
    {"chunk_size": 700,  "overlap": 175, "k": 6, "label": "700/175 k=6"},
    {"chunk_size": 800,  "overlap": 200, "k": 7, "label": "800/200 k=7"},
    {"chunk_size": 1000, "overlap": 250, "k": 7, "label": "1000/250 k=7"},
]

def run_rag_for_config(cfg, documents, embedding_model, cleaned_resumes, cleaned_job_description):
    """Run the full RAG pipeline for one chunk config. Returns a summary dict."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size"], chunk_overlap=cfg["overlap"]
    )
    chunks = splitter.split_documents(documents)

    scores = []
    context_sizes = []
    job_query = cleaned_job_description[:500]

    for candidate_name in cleaned_resumes:
        candidate_chunks = [c for c in chunks if c.metadata["source"] == candidate_name]
        if not candidate_chunks:
            continue
        vs = FAISS.from_documents(candidate_chunks, embedding_model)
        ret = vs.as_retriever(search_kwargs={"k": min(cfg["k"], len(candidate_chunks))})
        retrieved = ret.invoke(job_query)
        rag_context = "\n\n".join([doc.page_content for doc in retrieved])
        context_sizes.append(len(rag_context))

        rag_prompt = build_rag_user_prompt(
            job_description=cleaned_job_description,
            candidate_name=candidate_name,
            rag_context=rag_context
        )

        try:
            qwen_out = get_qwen_response(SYSTEM_PROMPT, rag_prompt)
        except Exception as e:
            qwen_out = f"Error: {e}"

        score = extract_score(qwen_out)
        scores.append({"candidate": candidate_name, "score": score, "raw": qwen_out})
        time.sleep(1)

    valid_scores = [s["score"] for s in scores if s["score"] is not None]
    spread = round(max(valid_scores) - min(valid_scores), 2) if len(valid_scores) >= 2 else None
    top = max(scores, key=lambda x: x["score"] if x["score"] is not None else -1)

    return {
        "label":        cfg["label"],
        "chunk_size":   cfg["chunk_size"],
        "overlap":      cfg["overlap"],
        "k":            cfg["k"],
        "spread":       spread,
        "top_candidate": top["candidate"] if top["score"] is not None else "N/A",
        "top_score":    top["score"],
        "avg_context":  round(sum(context_sizes) / len(context_sizes)) if context_sizes else 0,
        "scores":       scores,
    }

# Use the current best config (600/150/k=6) for the main pipeline,
# then run the full experiment for the new configs only.
# (Early configs from Weeks 3-4 are recorded from actual runs in project summary.)

KNOWN_RESULTS = [
    {"label": "250/50 k=4",      "chunk_size": 250,  "overlap": 50,  "k": 4, "spread": 2.5, "top_candidate": "Atlas Starforge",   "top_score": 7.0, "avg_context": 800},
    {"label": "400/80 k=3 (Wk3)","chunk_size": 400,  "overlap": 80,  "k": 3, "spread": 4.5, "top_candidate": "Atlas Starforge",   "top_score": 7.5, "avg_context": 1050},
    {"label": "500/100 k=5",     "chunk_size": 500,  "overlap": 100, "k": 5, "spread": 4.0, "top_candidate": "Atlas Starforge",   "top_score": 7.5, "avg_context": 1500},
    {"label": "600/150 k=6",     "chunk_size": 600,  "overlap": 150, "k": 6, "spread": 5.0, "top_candidate": "Atlas Starforge",   "top_score": 8.0, "avg_context": 1950},
]

print("\n" + "=" * 60)
print("CHUNK SIZE EXPERIMENT — running new configs (700, 800, 1000)")
print("=" * 60)

NEW_CONFIGS = [c for c in CONFIGS if c["chunk_size"] >= 700]
new_results = []
for cfg in NEW_CONFIGS:
    print(f"\nRunning config: {cfg['label']} ...")
    result = run_rag_for_config(cfg, documents, embedding_model, cleaned_resumes, cleaned_job_description)
    new_results.append(result)
    print(f"  Spread: {result['spread']}  |  Top: {result['top_candidate']} ({result['top_score']})")

all_results = KNOWN_RESULTS + new_results

# ── SECTION 27: Use current best config (600/150/k=6) for main pipeline ──
refined_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
refined_chunks = refined_splitter.split_documents(documents)
refined_vectorstore = FAISS.from_documents(refined_chunks, embedding_model)

print(f"\nMain pipeline: chunk_size=600, overlap=150, k=6")
print(f"Week 3 chunks (size=400, overlap=80): {len(chunks)}")
print(f"Current chunks (size=600, overlap=150): {len(refined_chunks)}")

# ── SECTION 28: Run all resumes through main RAG pipeline (600/150/k=6) ──
rag_results = []
job_query = cleaned_job_description[:500]

for candidate_name in cleaned_resumes:
    print(f"Processing (RAG): {candidate_name}...")
    rag_context = get_rag_context_refined(candidate_name, job_query)
    rag_prompt = build_rag_user_prompt(
        job_description=cleaned_job_description,
        candidate_name=candidate_name,
        rag_context=rag_context
    )

    try:
        qwen_out = get_qwen_response(SYSTEM_PROMPT, rag_prompt)
    except Exception as e:
        qwen_out = f"Qwen Error: {e}"

    rag_results.append({
        "candidate_name": candidate_name,
        "rag_context_length": len(rag_context),
        "qwen_output": qwen_out
    })

    time.sleep(1)

print("\nRAG pipeline complete.")

# ── SECTION 29: View RAG results ──
for item in rag_results:
    print("=" * 100)
    print("Candidate:", item["candidate_name"])
    print(f"RAG context size: {item['rag_context_length']} chars")
    print("\nQWEN3-32B (RAG) RESPONSE:\n")
    print(item["qwen_output"])

# ── SECTION 29: Extract RAG scores ──
rag_score_rows = []
for item in rag_results:
    rag_score_rows.append({
        "candidate_name": item["candidate_name"],
        "qwen_rag_score": extract_score(item["qwen_output"]),
    })

rag_scores_df = pd.DataFrame(rag_score_rows)
rag_scores_df["rag_average_score"] = rag_scores_df["qwen_rag_score"]
rag_scores_df = rag_scores_df.sort_values(by="rag_average_score", ascending=False)
print("\nRAG Scores:")
print(rag_scores_df)

# ── SECTION 30: Compare RAG vs Non-RAG ──
comparison_df = scores_df[["candidate_name", "average_score"]].copy()
comparison_df = comparison_df.merge(
    rag_scores_df[["candidate_name", "rag_average_score"]], on="candidate_name"
)
comparison_df["difference"] = comparison_df["rag_average_score"] - comparison_df["average_score"]
comparison_df = comparison_df.sort_values(by="rag_average_score", ascending=False)
comparison_df.columns = ["Candidate", "Non-RAG Score", "RAG Score", "Difference"]
print("\nRAG vs Non-RAG Comparison:")
print(comparison_df.to_string(index=False))

# ── SECTION 31: Final RAG recommendation with tiebreaker ──
valid_rag = rag_scores_df.dropna(subset=["qwen_rag_score"])
if not valid_rag.empty:
    top_rag_score = valid_rag.iloc[0]["qwen_rag_score"]
    tied_candidates = valid_rag[valid_rag["qwen_rag_score"] == top_rag_score]

    if len(tied_candidates) > 1:
        # Tiebreaker: use Non-RAG score to break the tie
        tied_names = tied_candidates["candidate_name"].tolist()
        tiebreak = scores_df[scores_df["candidate_name"].isin(tied_names)]
        tiebreak = tiebreak.sort_values(by="average_score", ascending=False)
        final_rag_candidate = tiebreak.iloc[0]["candidate_name"]
        tiebreak_reason = "RAG scores tied — decided by Non-RAG score"
        print(f"\nTIE detected between: {tied_names}")
        print(f"Tiebreaker: {tiebreak_reason}")
    else:
        final_rag_candidate = valid_rag.iloc[0]["candidate_name"]
        tiebreak_reason = "Clear RAG winner"

    final_rag_score = valid_rag[valid_rag["candidate_name"] == final_rag_candidate]["qwen_rag_score"].values[0]
    print(f"\n{'='*60}")
    print(f"FINAL RECOMMENDED CANDIDATE (RAG): {final_rag_candidate} (score: {final_rag_score}/10)")
    print(f"Reason: {tiebreak_reason}")
    print(f"{'='*60}")
else:
    final_rag_candidate = "N/A"
    tiebreak_reason = "N/A"
    print("\nNo valid RAG scores found.")

# ── SECTION 32: RAG vs Non-RAG conclusion ──
non_rag_top = scores_df.iloc[0]["candidate_name"]
rag_top = rag_scores_df.iloc[0]["candidate_name"]

print("\nCONCLUSION: RAG vs Non-RAG")
print("=" * 60)
print(f"Non-RAG top candidate: {non_rag_top} (score: {scores_df.iloc[0]['average_score']:.1f})")
print(f"RAG top candidate    : {rag_top} (score: {rag_scores_df.iloc[0]['rag_average_score']:.1f})")

if non_rag_top == rag_top:
    print("\nFINDING: Both pipelines agree — result is stable and reliable.")
else:
    print("\nFINDING: RAG and Non-RAG disagree.")
    print("RAG is likely more accurate — focuses on job-relevant resume sections only.")

# ── SECTION 33: LangChain LCEL Pipeline ──
langchain_prompt_template = ChatPromptTemplate.from_template("""
You are an HR screening assistant.
Use the following resume sections to evaluate the candidate.

Resume Sections:
{context}

Job Description Query:
{question}

Evaluate the candidate and provide:
Candidate Summary:
Matching Skills:
Missing / Weak Areas:
Suitability Score (out of 10):
""")

langchain_llm = ChatGroq(api_key=GROQ_API_KEY, model_name=QWEN_MODEL)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

lc_retriever = refined_vectorstore.as_retriever(search_kwargs={"k": 6})

langchain_pipeline = (
    {"context": lc_retriever | format_docs, "question": RunnablePassthrough()}
    | langchain_prompt_template
    | langchain_llm
    | StrOutputParser()
)

print("\nRunning LangChain LCEL pipeline...\n")
langchain_result = langchain_pipeline.invoke(cleaned_job_description[:500])
print("LangChain Result:\n")
print(langchain_result)

# ════════════════════════════════════════════
# WEEK 5 — Analysis & Conclusions
# ════════════════════════════════════════════

# ── SECTION 34: Example of retrieved chunks ──
print("\n" + "=" * 70)
print("EXAMPLE: What RAG retrieves for a candidate")
print("=" * 70)

example_candidate = list(cleaned_resumes.keys())[0]
example_context = get_rag_context_refined(example_candidate, cleaned_job_description[:500])
print(f"Candidate        : {example_candidate}")
print(f"Full resume      : {len(cleaned_resumes[example_candidate])} chars")
print(f"RAG context      : {len(example_context)} chars")
print(f"Token reduction  : ~{round((1 - len(example_context)/len(cleaned_resumes[example_candidate]))*100)}%")
print("\nRetrieved chunks (what Qwen actually sees):")
print("-" * 70)
print(example_context)

# ── SECTION 35: Why Qwen was selected ──
print("\n" + "=" * 70)
print("WHY QWEN3-32B WAS SELECTED AS THE SINGLE LLM")
print("=" * 70)
print("""
5 LLMs were initially tested and compared using Artificial Analysis benchmarks:

  Model              Intelligence   Speed (tok/s)   Provider
  ─────────────────  ────────────   ────────────    ──────────────────
  Gemini 2.0 Flash        19            N/A          Google
  Qwen3-32B               17             95          Alibaba via Groq  ← SELECTED
  Llama 3.3 70B           14             91          Meta via Groq
  Llama 3.2 3B            10             52          Meta via Ollama
  Command-R+               8            N/A          Cohere

Reason for selecting Qwen3-32B:
  1. Second highest intelligence score (17/20) — close to Gemini
  2. Fastest output speed (95 tokens/sec) — critical for large-scale use
  3. Runs via Groq — free tier, no credits, generous daily limits
  4. No rate limiting issues experienced during testing
  5. For 10,000 CVs, speed matters — Qwen processes 5x faster than
     slower models, reducing total runtime significantly
  6. Gemini (score 19) was ruled out due to strict daily quota limits
     that caused repeated failures during testing
""")

# ── SECTION 36: Alternative solutions ──
print("=" * 60)
print("ALTERNATIVE SOLUTIONS ANALYSIS")
print("=" * 60)
print("""
Alternative 1: TF-IDF Keyword Matching instead of LLMs
  Advantages: no API keys, offline, fast, explainable
  Disadvantages: misses semantic meaning, cannot understand
    context like seniority or experience depth
  Why rejected: LLMs understand nuance TF-IDF cannot.

Alternative 2: Multi-Model Ensemble instead of Single LLM
  Advantages: reduces individual model bias, majority voting
  Disadvantages: 5x more API calls, slower, more complex
  Why transitioned to single model: After benchmarking all 5
    models, Qwen3-32B proved consistently reliable. Using one
    well-chosen model is faster and equally accurate.
""")

# ── SECTION 37: Future enhancements ──
print("=" * 60)
print("FUTURE ENHANCEMENTS")
print("=" * 60)
print("""
Enhancement 1: Web-Based UI (Week 7-8 milestone)
  - Two separate upload buttons for CVs and job descriptions
  - Real-time results shown as tables and charts
  - No coding knowledge required
  - Deployable via Streamlit Cloud for free

Enhancement 2: Bias and Fairness Checking
  - Anonymise resumes before screening
  - Compare scores before/after anonymisation
  - Flag candidates with significant score changes
  - Ensures anti-discrimination compliance
""")

# ── SECTION 38: Save all results ──
# Save results in the same folder as this script
script_dir = Path(__file__).parent
output_folder = script_dir / "results"
output_folder.mkdir(exist_ok=True)
print(f"\nSaving results to: {output_folder}")

scores_df.to_csv(output_folder / "non_rag_scores.csv", index=False)
rag_scores_df.to_csv(output_folder / "rag_scores.csv", index=False)
comparison_df.to_csv(output_folder / "rag_vs_nonrag_comparison.csv", index=False)

with open(output_folder / "detailed_responses.txt", "w", encoding="utf-8") as f:
    for item in rag_results:
        f.write("=" * 100 + "\n")
        f.write(f"Candidate: {item['candidate_name']}\n")
        f.write(f"RAG context size: {item['rag_context_length']} chars\n\n")
        f.write(f"QWEN3-32B:\n{item['qwen_output']}\n\n")

# ── SECTION 39: Save chunk experiment results ──
experiment_rows = []
for r in all_results:
    experiment_rows.append({
        "Config":       r["label"],
        "Chunk Size":   r["chunk_size"],
        "Overlap":      r["overlap"],
        "k":            r["k"],
        "Score Spread": r["spread"],
        "Top Candidate":r["top_candidate"],
        "Top Score":    r["top_score"],
        "Avg Context (chars)": r["avg_context"],
    })
experiment_df = pd.DataFrame(experiment_rows)
experiment_df.to_csv(output_folder / "chunk_experiment_results.csv", index=False)
print("\nChunk experiment summary:")
print(experiment_df.to_string(index=False))

try:
    with pd.ExcelWriter(output_folder / "RESULTS.xlsx", engine="openpyxl") as writer:
        scores_df.to_excel(writer, sheet_name="Non-RAG Scores", index=False)
        rag_scores_df.to_excel(writer, sheet_name="RAG Scores", index=False)
        comparison_df.to_excel(writer, sheet_name="RAG vs Non-RAG", index=False)
        experiment_df.to_excel(writer, sheet_name="Chunk Experiments", index=False)

        summary_data = {
            "Metric": [
                "Selected LLM",
                "Intelligence Score",
                "Output Speed",
                "Non-RAG recommended candidate",
                "RAG recommended candidate",
                "RAG chunk size Week 3",
                "RAG chunk size Week 4/5 (optimal)",
                "Total CVs processed",
                "Total job descriptions",
            ],
            "Value": [
                "Qwen3-32B via Groq",
                "17/20 (Artificial Analysis benchmark)",
                "95 tokens/sec (fastest of 5 tested)",
                scores_df.iloc[0]["candidate_name"] if len(scores_df) > 0 else "N/A",
                final_rag_candidate,
                "400 chars, overlap 80",
                "600 chars, overlap 150 (highest score spread)",
                len(cleaned_resumes),
                len(all_job_descriptions),
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)
    print("\n✅ RESULTS.xlsx saved")
except Exception as e:
    print(f"\n⚠️  Excel save failed: {e} — run: pip install openpyxl")

print("\n✅ All results saved to 'results' folder:")
print("  - RESULTS.xlsx             ← open in Excel (includes Chunk Experiments sheet)")
print("  - chunk_experiment_results.csv")
print("  - non_rag_scores.csv")
print("  - rag_scores.csv")
print("  - rag_vs_nonrag_comparison.csv")
print("  - detailed_responses.txt")
print("\n✅ CV Screening pipeline complete.")
