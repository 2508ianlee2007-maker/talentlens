# ============================================================
# AI-POWERED CV SCREENING SYSTEM
# Single LLM Version — Qwen3-32B via Groq
# Updated to support MULTIPLE job descriptions
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
REQUEST_DELAY_SECONDS = 1

import tkinter as tk
from tkinter import simpledialog

try:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    GROQ_API_KEY = simpledialog.askstring(
        "Groq API Key",
        "Enter your GROQ API key:",
        show="*"
    )
    root.destroy()
except Exception:
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

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

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



def safe_name(name):
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', Path(name).stem)



def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])


# ── SECTION 5: Extract zip file ──
print("Current working directory:", os.getcwd())

import shutil

extract_folder = Path("project_files")
if extract_folder.exists():
    shutil.rmtree(extract_folder)  # wipe old contents first
extract_folder.mkdir()

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
raw_job_descriptions = {}
all_job_descriptions = {}
for jd_path in pdf_files:
    jd_text = read_pdf_text(jd_path)
    raw_job_descriptions[jd_path.name] = jd_text
    all_job_descriptions[jd_path.name] = preprocess_text(jd_text)

print(f"Loaded {len(all_job_descriptions)} job description(s)")

# Keep a primary JD only for preview/backward-friendly display
primary_jd_name = pdf_files[0].name
primary_raw_job_description = raw_job_descriptions[primary_jd_name]
primary_cleaned_job_description = all_job_descriptions[primary_jd_name]
print("Primary job description (preview only):", primary_jd_name)

# ── SECTION 9: Apply preprocessing ──
cleaned_resumes = {}
for name, text in raw_resumes.items():
    cleaned_resumes[name] = preprocess_text(text)

print("\nCleaned primary job description preview:")
print(primary_cleaned_job_description[:500])

# ── SECTION 10: Save cleaned text files ──
cleaned_folder = Path("cleaned_texts")
cleaned_folder.mkdir(exist_ok=True)

for name, text in cleaned_resumes.items():
    out_path = cleaned_folder / f"cleaned_{name}"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

for jd_name, jd_text in all_job_descriptions.items():
    jd_out_path = cleaned_folder / f"cleaned_{safe_name(jd_name)}.txt"
    with open(jd_out_path, "w", encoding="utf-8") as f:
        f.write(jd_text)

print("Cleaned files saved in:", cleaned_folder)

# ── SECTION 11: Preprocessing summary table ──
summary_rows = []
for name in raw_resumes:
    raw_text = raw_resumes[name]
    cleaned_text = cleaned_resumes[name]
    summary_rows.append({
        "file_name": name,
        "file_type": "resume",
        "raw_char_count": len(raw_text),
        "cleaned_char_count": len(cleaned_text),
        "removed_chars": len(raw_text) - len(cleaned_text)
    })

for jd_name in raw_job_descriptions:
    raw_text = raw_job_descriptions[jd_name]
    cleaned_text = all_job_descriptions[jd_name]
    summary_rows.append({
        "file_name": jd_name,
        "file_type": "job_description",
        "raw_char_count": len(raw_text),
        "cleaned_char_count": len(cleaned_text),
        "removed_chars": len(raw_text) - len(cleaned_text)
    })

preprocessing_summary_df = pd.DataFrame(summary_rows)
print(preprocessing_summary_df)

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


# ── SECTION 15: Test one resume first (primary JD only preview test) ──
test_candidate = list(cleaned_resumes.keys())[0]
test_resume = cleaned_resumes[test_candidate]
test_prompt = build_user_prompt(
    job_description=primary_cleaned_job_description,
    candidate_name=test_candidate,
    resume_text=test_resume
)

print(f"\nTesting on candidate: {test_candidate}")
print(f"Using preview JD: {primary_jd_name}")
print("\n--- QWEN3-32B TEST OUTPUT ---\n")
try:
    print(get_qwen_response(SYSTEM_PROMPT, test_prompt))
except Exception as e:
    print("Qwen Error:", e)

# ════════════════════════════════════════════
# SHARED RAG SETUP
# ════════════════════════════════════════════

# ── SECTION 16: Convert resumes to LangChain Documents ──
documents = []
for filename, text in cleaned_resumes.items():
    doc = Document(page_content=text, metadata={"source": filename})
    documents.append(doc)

print(f"\nTotal documents loaded: {len(documents)}")

# ── SECTION 17: Split into chunks ──
splitter = RecursiveCharacterTextSplitter(
    chunk_size=FINAL_CHUNK_SIZE,
    chunk_overlap=FINAL_CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)} (chunk_size={FINAL_CHUNK_SIZE}, overlap={FINAL_CHUNK_OVERLAP})")

# ── SECTION 18: Load embedding model ──
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded.")

# ── SECTION 19: Build shared refined chunks ──
refined_chunks = chunks

# ── SECTION 20: Build or load shared FAISS index on disk ──
FAISS_INDEX_PATH = Path("faiss_index")
print("FAISS is saved at:", FAISS_INDEX_PATH.resolve())

if FAISS_INDEX_PATH.exists():
    print("Loading existing FAISS index from disk...")
    refined_vectorstore = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embedding_model,
        allow_dangerous_deserialization=True
    )
    print("FAISS index loaded.")
else:
    print("Building new FAISS index and saving to disk...")
    refined_vectorstore = FAISS.from_documents(refined_chunks, embedding_model)
    refined_vectorstore.save_local(str(FAISS_INDEX_PATH))
    print(f"FAISS index saved to '{FAISS_INDEX_PATH}'.")


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


# ════════════════════════════════════════════
# MULTI-JD PROCESSING
# ════════════════════════════════════════════

def process_single_job_description(jd_name, cleaned_job_description):
    print("\n" + "=" * 100)
    print(f"PROCESSING JOB DESCRIPTION: {jd_name}")
    print("=" * 100)

    # ── Non-RAG ──
    results = []
    for candidate_name, resume_text in cleaned_resumes.items():
        user_prompt = build_user_prompt(
            job_description=cleaned_job_description,
            candidate_name=candidate_name,
            resume_text=resume_text
        )
        print(f"Processing {candidate_name} for JD '{jd_name}' (Non-RAG)...")

        try:
            qwen_output = get_qwen_response(SYSTEM_PROMPT, user_prompt)
        except Exception as e:
            qwen_output = f"Qwen Error: {e}"

        results.append({
            "job_description": jd_name,
            "candidate_name": candidate_name,
            "qwen_output": qwen_output
        })

        time.sleep(REQUEST_DELAY_SECONDS)

    score_rows = []
    for item in results:
        score_rows.append({
            "job_description": jd_name,
            "candidate_name": item["candidate_name"],
            "qwen_score": extract_score(item["qwen_output"]),
        })

    scores_df = pd.DataFrame(score_rows)
    scores_df["average_score"] = scores_df["qwen_score"]
    scores_df = scores_df.sort_values(by="average_score", ascending=False, na_position="last")

    valid_df = scores_df.dropna(subset=["qwen_score"])
    if not valid_df.empty:
        final_best_candidate = valid_df.iloc[0]["candidate_name"]
        final_score = valid_df.iloc[0]["qwen_score"]
        print(f"\nFINAL RECOMMENDED RESUME (Non-RAG) for {jd_name}: {final_best_candidate} (score: {final_score}/10)")
    else:
        final_best_candidate = "N/A"
        final_score = None
        print(f"\nNo valid Non-RAG scores found for {jd_name}.")

    # ── RAG ──
    rag_results = []
    job_query = cleaned_job_description[:500]

    for candidate_name in cleaned_resumes:
        print(f"Processing {candidate_name} for JD '{jd_name}' (RAG)...")
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
            "job_description": jd_name,
            "candidate_name": candidate_name,
            "rag_context_length": len(rag_context),
            "qwen_output": qwen_out
        })

        time.sleep(REQUEST_DELAY_SECONDS)

    rag_score_rows = []
    for item in rag_results:
        rag_score_rows.append({
            "job_description": jd_name,
            "candidate_name": item["candidate_name"],
            "qwen_rag_score": extract_score(item["qwen_output"]),
        })

    rag_scores_df = pd.DataFrame(rag_score_rows)
    rag_scores_df["rag_average_score"] = rag_scores_df["qwen_rag_score"]
    rag_scores_df = rag_scores_df.sort_values(by="rag_average_score", ascending=False, na_position="last")

    comparison_df = scores_df[["job_description", "candidate_name", "average_score"]].copy()
    comparison_df = comparison_df.merge(
        rag_scores_df[["job_description", "candidate_name", "rag_average_score"]],
        on=["job_description", "candidate_name"],
        how="left"
    )
    comparison_df["difference"] = comparison_df["rag_average_score"] - comparison_df["average_score"]
    comparison_df.columns = ["Job Description", "Candidate", "Non-RAG Score", "RAG Score", "Difference"]
    comparison_df = comparison_df.sort_values(by="RAG Score", ascending=False, na_position="last")

    valid_rag = rag_scores_df.dropna(subset=["qwen_rag_score"])
    if not valid_rag.empty:
        top_rag_score = valid_rag.iloc[0]["qwen_rag_score"]
        tied_candidates = valid_rag[valid_rag["qwen_rag_score"] == top_rag_score]

        if len(tied_candidates) > 1:
            tied_names = tied_candidates["candidate_name"].tolist()
            tiebreak = scores_df[scores_df["candidate_name"].isin(tied_names)]
            tiebreak = tiebreak.sort_values(by="average_score", ascending=False, na_position="last")
            final_rag_candidate = tiebreak.iloc[0]["candidate_name"]
            tiebreak_reason = "RAG scores tied — decided by Non-RAG score"
            print(f"\nTIE detected for JD '{jd_name}' between: {tied_names}")
            print(f"Tiebreaker: {tiebreak_reason}")
        else:
            final_rag_candidate = valid_rag.iloc[0]["candidate_name"]
            tiebreak_reason = "Clear RAG winner"

        final_rag_score = valid_rag[valid_rag["candidate_name"] == final_rag_candidate]["qwen_rag_score"].values[0]
        print(f"\n{'=' * 60}")
        print(f"FINAL RECOMMENDED CANDIDATE (RAG) for {jd_name}: {final_rag_candidate} (score: {final_rag_score}/10)")
        print(f"Reason: {tiebreak_reason}")
        print(f"{'=' * 60}")
    else:
        final_rag_candidate = "N/A"
        final_rag_score = None
        tiebreak_reason = "N/A"
        print(f"\nNo valid RAG scores found for {jd_name}.")

    # ── LCEL demo for this JD ──
    lc_retriever = refined_vectorstore.as_retriever(search_kwargs={"k": FINAL_K})
    langchain_pipeline = (
        {"context": lc_retriever | format_docs, "question": RunnablePassthrough()}
        | langchain_prompt_template
        | langchain_llm
        | StrOutputParser()
    )

    print(f"\nRunning LangChain LCEL pipeline for JD: {jd_name}...\n")
    try:
        langchain_result = langchain_pipeline.invoke(cleaned_job_description[:500])
    except Exception as e:
        langchain_result = f"LangChain Error: {e}"

    # Example retrieved chunks using top RAG candidate if available
    example_candidate = final_rag_candidate if final_rag_candidate != "N/A" else list(cleaned_resumes.keys())[0]
    example_context = get_rag_context_refined(example_candidate, cleaned_job_description[:500])
    token_reduction = None
    if len(cleaned_resumes[example_candidate]) > 0:
        token_reduction = round((1 - len(example_context) / len(cleaned_resumes[example_candidate])) * 100)

    return {
        "jd_name": jd_name,
        "cleaned_job_description": cleaned_job_description,
        "non_rag_results": results,
        "non_rag_scores_df": scores_df,
        "non_rag_best_candidate": final_best_candidate,
        "non_rag_best_score": final_score,
        "rag_results": rag_results,
        "rag_scores_df": rag_scores_df,
        "comparison_df": comparison_df,
        "rag_best_candidate": final_rag_candidate,
        "rag_best_score": final_rag_score,
        "tiebreak_reason": tiebreak_reason,
        "langchain_result": langchain_result,
        "example_candidate": example_candidate,
        "example_context": example_context,
        "token_reduction": token_reduction,
    }


all_job_outputs = {}
for jd_name, cleaned_jd in all_job_descriptions.items():
    all_job_outputs[jd_name] = process_single_job_description(jd_name, cleaned_jd)


# ════════════════════════════════════════════
# FINAL CROSS-JD SUMMARY
# ════════════════════════════════════════════
print("\n" + "#" * 100)
print("FINAL SUMMARY ACROSS ALL JOB DESCRIPTIONS")
print("#" * 100)

summary_rows = []
for jd_name, output in all_job_outputs.items():
    summary_rows.append({
        "job_description": jd_name,
        "non_rag_best_candidate": output["non_rag_best_candidate"],
        "non_rag_best_score": output["non_rag_best_score"],
        "rag_best_candidate": output["rag_best_candidate"],
        "rag_best_score": output["rag_best_score"],
        "tiebreak_reason": output["tiebreak_reason"],
    })

multi_jd_summary_df = pd.DataFrame(summary_rows)
print(multi_jd_summary_df.to_string(index=False))


# ── Save all results ──
script_dir = Path(__file__).parent
output_folder = script_dir / "results"
output_folder.mkdir(exist_ok=True)
print(f"\nSaving results to: {output_folder}")

# Save preprocessing summary
preprocessing_summary_df.to_csv(output_folder / "preprocessing_summary.csv", index=False)

# Save summary across all JDs
multi_jd_summary_df.to_csv(output_folder / "multi_jd_final_summary.csv", index=False)

# Save per-JD files
for jd_name, output in all_job_outputs.items():
    jd_folder = output_folder / safe_name(jd_name)
    jd_folder.mkdir(exist_ok=True)

    output["non_rag_scores_df"].to_csv(jd_folder / "non_rag_scores.csv", index=False)
    output["rag_scores_df"].to_csv(jd_folder / "rag_scores.csv", index=False)
    output["comparison_df"].to_csv(jd_folder / "rag_vs_nonrag_comparison.csv", index=False)

    with open(jd_folder / "detailed_responses.txt", "w", encoding="utf-8") as f:
        f.write(f"JOB DESCRIPTION: {jd_name}\n")
        f.write("=" * 100 + "\n\n")

        f.write("NON-RAG RESPONSES\n")
        f.write("-" * 50 + "\n")
        for item in output["non_rag_results"]:
            f.write("=" * 100 + "\n")
            f.write(f"Candidate: {item['candidate_name']}\n\n")
            f.write(f"QWEN3-32B (Non-RAG):\n{item['qwen_output']}\n\n")

        f.write("\nRAG RESPONSES\n")
        f.write("-" * 50 + "\n")
        for item in output["rag_results"]:
            f.write("=" * 100 + "\n")
            f.write(f"Candidate: {item['candidate_name']}\n")
            f.write(f"RAG context size: {item['rag_context_length']} chars\n\n")
            f.write(f"QWEN3-32B (RAG):\n{item['qwen_output']}\n\n")

        f.write("\nLANGCHAIN LCEL RESULT\n")
        f.write("-" * 50 + "\n")
        f.write(output["langchain_result"] + "\n\n")

        f.write("\nEXAMPLE RETRIEVED CONTEXT\n")
        f.write("-" * 50 + "\n")
        f.write(f"Candidate: {output['example_candidate']}\n")
        f.write(f"Token reduction: {output['token_reduction']}%\n\n")
        f.write(output["example_context"])

try:
    with pd.ExcelWriter(output_folder / "RESULTS_MULTI_JD.xlsx", engine="openpyxl") as writer:
        preprocessing_summary_df.to_excel(writer, sheet_name="Preprocessing Summary", index=False)
        multi_jd_summary_df.to_excel(writer, sheet_name="Final Summary", index=False)

        for jd_name, output in all_job_outputs.items():
            base_sheet = safe_name(jd_name)[:20] or "JD"
            output["non_rag_scores_df"].to_excel(writer, sheet_name=f"{base_sheet}_NR"[:31], index=False)
            output["rag_scores_df"].to_excel(writer, sheet_name=f"{base_sheet}_RAG"[:31], index=False)
            output["comparison_df"].to_excel(writer, sheet_name=f"{base_sheet}_CMP"[:31], index=False)

    print("Excel file saved: RESULTS_MULTI_JD.xlsx")
except PermissionError:
    print("Could not save RESULTS_MULTI_JD.xlsx because it is open. Close Excel and run again.")


# ── Final printed summary ──
print("\n" + "=" * 80)
print("BEST CV FOR EACH JOB DESCRIPTION")
print("=" * 80)
for jd_name, output in all_job_outputs.items():
    rag_score_text = f"{output['rag_best_score']}/10" if output['rag_best_score'] is not None else "N/A"
    print(f"- {jd_name} -> {output['rag_best_candidate']} (RAG score: {rag_score_text})")

print("\nDone. Multi-job-description support is now enabled.")


# ── Presentation-friendly summary helpers ──
def extract_named_section(text, section_names):
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    escaped = [re.escape(name) for name in section_names]
    pattern = r"(?is)(?:^|\n)\s*(%s)\s*:\s*(.*?)(?=\n\s*(?:Candidate\ Summary|Candidate\ Summary|Matching\ Skills|Missing\s*/\s*Weak\ Areas|Missing\s*Areas|Weak\ Areas|Suitability\ Score|Overall\s+Score|Final\s+Score)\s*:|\Z)" % "|".join(escaped)
    match = re.search(pattern, cleaned)
    if match:
        return re.sub(r"\s+", " ", match.group(2)).strip(" -\n\t")
    return ""

def build_short_recommendation_reason(output):
    recommended_name = output.get("rag_best_candidate")
    if not recommended_name or recommended_name == "N/A":
        return "No valid recommendation found."
    best_row = None
    for item in output.get("rag_results", []):
        if item.get("candidate_name") == recommended_name:
            best_row = item
            break
    rag_output_text = best_row.get("qwen_output", "") if best_row else ""
    candidate_summary = extract_named_section(rag_output_text, ["Candidate Summary"])
    matching_skills = extract_named_section(rag_output_text, ["Matching Skills"])
    reason_parts = []
    if candidate_summary:
        reason_parts.append(candidate_summary)
    if matching_skills:
        skills = matching_skills
        if len(skills) > 180:
            skills = skills[:177].rstrip() + "..."
        reason_parts.append(f"Key match: {skills}")
    if output.get("tiebreak_reason") and output.get("tiebreak_reason") != "Clear RAG winner":
        reason_parts.append(output["tiebreak_reason"])
    if not reason_parts:
        return "Recommended based on the highest RAG suitability score."
    return " ".join(reason_parts)

presentation_summary_rows = []
for jd_name, output in all_job_outputs.items():
    recommended_cv = output["rag_best_candidate"]
    recommended_reason = build_short_recommendation_reason(output)
    non_rag_score_value = None
    if recommended_cv != "N/A":
        nr_match = output["non_rag_scores_df"][output["non_rag_scores_df"]["candidate_name"] == recommended_cv]
        if not nr_match.empty:
            non_rag_score_value = nr_match.iloc[0]["average_score"]
    presentation_summary_rows.append({
        "Job Description": jd_name,
        "Recommended CV": recommended_cv,
        "RAG Score": output["rag_best_score"],
        "Non-RAG Score": non_rag_score_value,
        "Reason": recommended_reason,
    })

presentation_summary_df = pd.DataFrame(presentation_summary_rows)

print("\n" + "=" * 80)
print("PRESENTATION-FRIENDLY FINAL TABLE")
print("=" * 80)
if not presentation_summary_df.empty:
    print(presentation_summary_df.to_string(index=False))
else:
    print("No presentation summary available.")

presentation_summary_df.to_csv(output_folder / "presentation_summary_table.csv", index=False)

try:
    with pd.ExcelWriter(output_folder / "PRESENTATION_SUMMARY.xlsx", engine="openpyxl") as writer:
        presentation_summary_df.to_excel(writer, sheet_name="Presentation Summary", index=False)
    print("Excel file saved: PRESENTATION_SUMMARY.xlsx")
except PermissionError:
    print("Could not save PRESENTATION_SUMMARY.xlsx because it is open. Close Excel and run again.")
