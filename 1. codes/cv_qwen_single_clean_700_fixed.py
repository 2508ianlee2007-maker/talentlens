# ============================================================
# AI-POWERED CV SCREENING SYSTEM
# Single LLM Version — Qwen3-32B via Groq
# Final main pipeline uses 700/175/k=6 based on chunk experiment
# ============================================================

# ── SECTION 1: Install libraries ──
# Run this ONCE in your terminal before running this script:
# pip install pypdf pandas groq openpyxl
# pip install langchain langchain-community langchain-core
# pip install langchain-text-splitters langchain-huggingface
# pip install faiss-cpu sentence-transformers langchain-groq

# ── SECTION 2: Imports ──
import os
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from groq import Groq

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
QWEN_MODEL = "qwen/qwen3-32b"

# Final RAG settings
FINAL_CHUNK_SIZE = 700
FINAL_CHUNK_OVERLAP = 175
FINAL_K = 6


import tkinter as tk
from tkinter import simpledialog

try:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)   # force popup to front on Windows
    GROQ_API_KEY = simpledialog.askstring(
        "Groq API Key",
        "Enter your GROQ API key:",
        show="*"
    )
    root.destroy()
except Exception:
    # Fallback: plain terminal prompt (works on headless/server environments)
    import getpass
    GROQ_API_KEY = getpass.getpass("Enter your GROQ API key (input hidden): ")

if not GROQ_API_KEY or not GROQ_API_KEY.strip():
    raise EnvironmentError("No GROQ API key provided.")

GROQ_API_KEY = GROQ_API_KEY.strip()
print("API key loaded successfully.")

# ── SECTION 4: Helper functions ──
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

    # Remove Qwen reasoning block first
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove markdown formatting
    cleaned = text.replace("**", "").replace("__", "").replace("*", "")
    cleaned = cleaned.replace("\r", "\n")

    patterns = [
        r"suitability\s*score\s*[:\-–]?\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"suitability\s*score\s*[:\-–]?\s*(\d+(?:\.\d+)?)\s*out\s*of\s*(?:10|ten)",
        r"suitability\s*score\s*[:\-–]?\s*(\d+(?:\.\d+)?)",
        r"(?:^|\n)\s*(?:overall\s+)?(?:final\s+)?(?:score|rating)\s*[:\-–]?\s*(\d+(?:\.\d+)?)\s*/\s*10",
        r"(?:^|\n)\s*(?:overall\s+)?(?:final\s+)?(?:score|rating)\s*[:\-–]?\s*(\d+(?:\.\d+)?)\s*out\s*of\s*(?:10|ten)",
        r"(?:^|\n)\s*(?:overall\s+)?(?:final\s+)?(?:score|rating)\s*[:\-–]?\s*(\d+(?:\.\d+)?)",
        r"[\(\[]\s*(\d+(?:\.\d+)?)\s*/\s*10\s*[\)\]]",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if 0 <= val <= 10:
                return val

    fallback_matches = re.findall(
        r"(?:suitability|score|rating)[^0-9\n]{0,20}(\d+(?:\.\d+)?)",
        cleaned,
        re.IGNORECASE,
    )
    for m in reversed(fallback_matches):
        val = float(m)
        if 0 <= val <= 10:
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
        ],
        temperature=0.1
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

# ── SECTION 21: Split into chunks (Final config: 700/175) ──
splitter = RecursiveCharacterTextSplitter(chunk_size=FINAL_CHUNK_SIZE, chunk_overlap=FINAL_CHUNK_OVERLAP)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)} (chunk_size={FINAL_CHUNK_SIZE}, overlap={FINAL_CHUNK_OVERLAP})")

# ── SECTION 22: Load embedding model ──
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded.")

# ── SECTION 23: Build or Load FAISS Vector Store ──
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

# ── SECTION 24: Vector store info ──
print(f"Total vectors stored: {vectorstore.index.ntotal}")
print("(Retrieval test runs after refined_vectorstore is built below)")

# ── SECTION 25: Test RAG retrieval with the initial vectorstore ──
# (get_rag_context_refined is defined after refined_chunks is built below)

# ════════════════════════════════════════════
# WEEK 4 — Refined RAG
# ════════════════════════════════════════════

# ── SECTION 27: refined_chunks = chunks (same 700/175 config, no need to rebuild) ──
refined_chunks = chunks
refined_vectorstore = FAISS.from_documents(refined_chunks, embedding_model)

# ── SECTION 25b: RAG screening function (defined here so refined_chunks exists) ──
def get_rag_context_refined(candidate_name, job_description_query, k=None):
    if k is None:
        k = FINAL_K
    candidate_docs = [c for c in refined_chunks if c.metadata["source"] == candidate_name]
    if not candidate_docs:
        return ""
    vs = FAISS.from_documents(candidate_docs, embedding_model)
    ret = vs.as_retriever(search_kwargs={"k": min(k, len(candidate_docs))})
    retrieved = ret.invoke(job_description_query)
    return "\n\n".join([doc.page_content for doc in retrieved])

print(f"\nMain pipeline running: chunk_size={FINAL_CHUNK_SIZE}, overlap={FINAL_CHUNK_OVERLAP}, k={FINAL_K}")
print(f"Total chunks: {len(refined_chunks)} (refined_chunks == chunks, same config)")

# Quick retrieval test now that everything is ready
query = cleaned_job_description[:500]
test_retriever = refined_vectorstore.as_retriever(search_kwargs={"k": FINAL_K})
retrieved_docs = test_retriever.invoke(query)
print(f"\nRetrieved {len(retrieved_docs)} relevant chunks (retrieval test):")
for i, doc in enumerate(retrieved_docs):
    print(f"--- Chunk {i+1} (from: {doc.metadata['source']}) ---")
    print(doc.page_content[:200])

# ── SECTION 28: Run all resumes through main RAG pipeline ──
rag_results = []
job_query = cleaned_job_description[:500]

for candidate_name in cleaned_resumes:
    print(f"Processing (RAG): {candidate_name}...")
    rag_context = get_rag_context_refined(candidate_name, job_query, k=FINAL_K)
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

# ── SECTION 30: Extract RAG scores ──
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

# ── SECTION 31: Compare RAG vs Non-RAG ──
comparison_df = scores_df[["candidate_name", "average_score"]].copy()
comparison_df = comparison_df.merge(
    rag_scores_df[["candidate_name", "rag_average_score"]], on="candidate_name"
)
comparison_df["difference"] = comparison_df["rag_average_score"] - comparison_df["average_score"]
comparison_df = comparison_df.sort_values(by="rag_average_score", ascending=False)
comparison_df.columns = ["Candidate", "Non-RAG Score", "RAG Score", "Difference"]
print("\nRAG vs Non-RAG Comparison:")
print(comparison_df.to_string(index=False))

# ── SECTION 32: Final RAG recommendation with tiebreaker ──
valid_rag = rag_scores_df.dropna(subset=["qwen_rag_score"])
if not valid_rag.empty:
    top_rag_score = valid_rag.iloc[0]["qwen_rag_score"]
    tied_candidates = valid_rag[valid_rag["qwen_rag_score"] == top_rag_score]

    if len(tied_candidates) > 1:
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

# ── SECTION 33: RAG vs Non-RAG conclusion ──
non_rag_top = scores_df.iloc[0]["candidate_name"]
rag_top = rag_scores_df.iloc[0]["candidate_name"]

print("\nCONCLUSION: RAG vs Non-RAG")
print("=" * 60)
non_rag_top_score = scores_df.iloc[0]["average_score"]
rag_top_score = rag_scores_df.iloc[0]["rag_average_score"]
print(f"Non-RAG top candidate: {non_rag_top} (score: {non_rag_top_score:.1f})" if pd.notna(non_rag_top_score) else f"Non-RAG top candidate: {non_rag_top} (score: N/A)")
print(f"RAG top candidate    : {rag_top} (score: {rag_top_score:.1f})" if pd.notna(rag_top_score) else f"RAG top candidate    : {rag_top} (score: N/A)")

if non_rag_top == rag_top:
    print("\nFINDING: Both pipelines agree — result is stable and reliable.")
else:
    print("\nFINDING: RAG and Non-RAG disagree.")
    print("RAG is likely more accurate — focuses on job-relevant resume sections only.")

# ── SECTION 34: LangChain LCEL Pipeline ──
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


lc_retriever = refined_vectorstore.as_retriever(search_kwargs={"k": FINAL_K})

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

# ── SECTION 35: Example of retrieved chunks ──
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

# ── SECTION 36: Why Qwen was selected ──
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

# ── SECTION 37: Alternative solutions ──
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

# ── SECTION 38: Future enhancements ──
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

# ── SECTION 39: Save all results ──
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

try:
    with pd.ExcelWriter(output_folder / "RESULTS.xlsx", engine="openpyxl") as writer:
        scores_df.to_excel(writer, sheet_name="Non-RAG Scores", index=False)
        rag_scores_df.to_excel(writer, sheet_name="RAG Scores", index=False)
        comparison_df.to_excel(writer, sheet_name="RAG vs Non-RAG", index=False)

        # Safe display values for summary sheet
        non_rag_candidate = valid_df.iloc[0]["candidate_name"] if not valid_df.empty else "N/A"
        non_rag_score = valid_df.iloc[0]["qwen_score"] if not valid_df.empty else None
        rag_valid_df = rag_scores_df.dropna(subset=["qwen_rag_score"])
        rag_candidate = rag_valid_df.iloc[0]["candidate_name"] if not rag_valid_df.empty else "N/A"
        rag_score = rag_valid_df.iloc[0]["qwen_rag_score"] if not rag_valid_df.empty else None

        summary_data = {
            "Metric": [
                "Selected LLM",
                "Intelligence Score",
                "Output Speed",
                "Non-RAG recommended candidate",
                "RAG recommended candidate",
                "RAG chunk size (final)",
                "Top chunks retrieved (k)",
                "Total CVs processed",
                "Total job descriptions",
            ],
            "Value": [
                "Qwen3-32B via Groq",
                "17/20 (Artificial Analysis benchmark)",
                "95 tokens/sec (fastest of tested models)",
                f"{non_rag_candidate} ({non_rag_score}/10)" if non_rag_score is not None else non_rag_candidate,
                f"{rag_candidate} ({rag_score}/10)" if rag_score is not None else rag_candidate,
                f"{FINAL_CHUNK_SIZE} chars, overlap {FINAL_CHUNK_OVERLAP}",
                FINAL_K,
                len(cleaned_resumes),
                len(all_job_descriptions),
            ],
        }
        summary_df_excel = pd.DataFrame(summary_data)
        summary_df_excel.to_excel(writer, sheet_name="Summary", index=False)

    print("Excel file saved: RESULTS.xlsx")
except PermissionError:
    print("Could not save RESULTS.xlsx because it is open. Close Excel and run again.")
