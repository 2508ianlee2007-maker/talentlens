import os
import re
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any

from pypdf import PdfReader
from docx import Document as DocxDocument
from groq import Groq
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VERSION = "v3.0-improvements"  # bump this when you redeploy to confirm Render picked up the new file

QWEN_MODEL = "qwen/qwen3-32b"
FINAL_CHUNK_SIZE = 700
FINAL_CHUNK_OVERLAP = 175
FINAL_K = 6
SYSTEM_PROMPT = """You are a senior HR screening assistant with expertise in 
technical hiring for semiconductor and hardware engineering roles.

You will be given a job description (JD) and a candidate's resume. Your task 
is to produce a structured screening report.

Follow these rules strictly:
- Only assess skills and experience explicitly stated in the resume. Do not infer 
  or assume unstated competencies.
- Distinguish between hands-on experience and exposure/familiarity.
- Weight criteria in this order: (1) Must-have technical skills, (2) Years of 
  relevant experience, (3) Nice-to-have skills, (4) Education.
- If a requirement from the JD has no match in the resume, explicitly flag it 
  as a gap — do not omit it.

Return your report in this exact structure:

**Candidate Summary** (2-3 sentences: role fit at a glance)

**Must-Have Skills Match**
| Requirement | Found in Resume? | Evidence |
|---|---|---|

**Nice-to-Have Skills Match**
| Requirement | Found? | Evidence |
|---|---|---|

**Experience Assessment**
- Years of relevant experience: 
- Seniority alignment:
- Notable projects/achievements:

**Key Gaps**
(List missing or weak areas with specific JD requirements they fail to meet)

**Suitability Score: X/10**
Scoring rationale: (2-3 sentences explaining the score)

**Recommendation:** Advance / Hold / Reject
"""

_CHAT_SYSTEM_PROMPT = """You are an AI assistant embedded in a CV screening 
tool used by an HR team to evaluate engineering candidates.

You have access to:
- The job description
- Candidate resumes (raw text)
- Pre-generated screening reports for each candidate

Your behaviour rules:
- Answer only from the provided context. Never fabricate experience or skills 
  a candidate does not have.
- When comparing candidates, use specific evidence (e.g. "Candidate A has 5 
  years of UVM experience vs Candidate B's 2 years").
- If asked to rank candidates, justify each placement with data from their reports.
- If the context is insufficient to answer confidently, say exactly what 
  information is missing rather than guessing.
- Keep answers concise but evidence-based. Avoid vague qualifiers like 
  "seems strong" — prefer "has 3 years of hands-on SystemVerilog per resume".
"""

_EMBEDDINGS = None


def get_embedding_model():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDINGS


def read_pdf_text(path: str) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_txt_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_docx_text(path: str) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())


def read_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    try:
        if suffix == ".pdf":
            return read_pdf_text(tmp_path)
        if suffix == ".docx":
            return read_docx_text(tmp_path)
        if suffix == ".txt":
            return read_txt_text(tmp_path)
        raise ValueError(f"Unsupported file type: {suffix}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("\xa0", " ").replace("\n", " ").replace("\t", " ").lower()
    text = re.sub(r'(\+?\d[\d\-\(\) ]{7,}\d)', ' ', text)
    text = re.sub(r'[^a-z0-9/\+\#\-\.\s]', ' ', text)
    text = re.sub(r'[\.]{2,}', ' ', text)
    text = re.sub(r'[-]{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_score(text: str):
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = text.replace("**", "").replace("__", "").replace("*", "").replace("\r", "\n")
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
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if 0 <= val <= 10:
                return val
    fallback = re.findall(r"(?:suitability|score|rating)[^0-9\n]{0,20}(\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)
    for item in reversed(fallback):
        val = float(item)
        if 0 <= val <= 10:
            return val
    return None


def extract_section(text: str, names: List[str]) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    escaped = [re.escape(n) for n in names]
    pattern = (
        r"(?is)(?:^|\n)\s*(%s)\s*:\s*(.*?)"
        r"(?=\n\s*(?:Candidate Summary|Matching Skills|Missing\s*/\s*Weak Areas"
        r"|Missing Areas|Weak Areas|Suitability Score|Overall Score|Final Score)\s*:|\Z)"
    ) % "|".join(escaped)
    m = re.search(pattern, cleaned)
    if not m:
        return ""
    return re.sub(r"\s+", " ", m.group(2)).strip(" -\n\t")


def fmt_name(filename: str) -> str:
    return Path(filename).stem.replace("_", " ").replace("-", " ")


def verdict_label(score):
    if score is None:
        return "Unknown", "v-weak"
    if score >= 8:
        return "Strong Match", "v-strong"
    if score >= 6:
        return "Good Match", "v-good"
    if score >= 4:
        return "Partial Match", "v-partial"
    return "Weak Match", "v-weak"


def score_color(score):
    if score is None:
        return "#94a3b8"
    if score >= 8:
        return "#0f766e"
    if score >= 6:
        return "#1d4ed8"
    if score >= 4:
        return "#b45309"
    return "#b91c1c"


def build_user_prompt(jd: str, name: str, resume: str) -> str:
    return (
        f"Job Description:\n{jd}\n\nCandidate Resume File Name:\n{name}\n\n"
        f"Candidate Resume Text:\n{resume}\n\nPlease evaluate this candidate.\n\n"
        "Format your answer as:\nCandidate Summary:\nMatching Skills:\n"
        "Missing / Weak Areas:\nSuitability Score:"
    )


def build_rag_prompt(jd: str, name: str, context: str) -> str:
    return (
        f"Job Description:\n{jd}\n\nCandidate Resume File Name:\n{name}\n\n"
        f"Most Relevant Resume Sections (retrieved via RAG):\n{context}\n\n"
        "Based on the retrieved resume sections above, please evaluate this candidate.\n\n"
        "Format your answer as:\nCandidate Summary:\nMatching Skills:\n"
        "Missing / Weak Areas:\nSuitability Score:"
    )


def load_and_preprocess_files(jd_uploads, cv_uploads):
    jd_files, raw_jd, cv_files, raw_cv = {}, {}, {}, {}
    errors = []

    for f in jd_uploads or []:
        try:
            raw = read_uploaded_file(f)
            raw_jd[f.name] = raw
            jd_files[f.name] = preprocess_text(raw)
        except Exception as e:
            errors.append(f"JD '{f.name}': {e}")

    for f in cv_uploads or []:
        try:
            raw = read_uploaded_file(f)
            raw_cv[f.name] = raw
            cv_files[f.name] = preprocess_text(raw)
        except Exception as e:
            errors.append(f"CV '{f.name}': {e}")

    return {
        "jd_files": jd_files,
        "raw_jd_texts": raw_jd,
        "cv_files": cv_files,
        "raw_cv_texts": raw_cv,
        "errors": errors,
    }


def _build_chunks(cv_files: Dict[str, str]):
    documents = [Document(page_content=text, metadata={"source": name}) for name, text in cv_files.items()]
    splitter = RecursiveCharacterTextSplitter(chunk_size=FINAL_CHUNK_SIZE, chunk_overlap=FINAL_CHUNK_OVERLAP)
    return splitter.split_documents(documents)


def _get_rag_context(chunks, emb, cv_name: str, query: str) -> str:
    cand_chunks = [c for c in chunks if c.metadata.get("source") == cv_name]
    if not cand_chunks:
        return ""
    vectorstore = FAISS.from_documents(cand_chunks, emb)
    retriever = vectorstore.as_retriever(search_kwargs={"k": min(FINAL_K, len(cand_chunks))})
    return "\n\n".join(d.page_content for d in retriever.invoke(query))


def _call_llm(client: Groq, prompt: str, retries: int = 3, base_delay: float = 5.0) -> str:
    """Call the LLM with exponential backoff retry on rate-limit or server errors."""
    last_err = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Only retry on rate limits or server errors, not auth/bad request
            if any(code in err_str for code in ["429", "500", "503", "rate_limit", "timeout"]):
                wait = base_delay * (2 ** attempt)  # 5s, 10s, 20s
                time.sleep(wait)
            else:
                raise  # non-retryable error, fail immediately
    raise RuntimeError(f"LLM call failed after {retries} retries: {last_err}")


# ── ANONYMISATION ─────────────────────────────────────────────────────────────
# NOTE: anonymise_text() must be called on RAW text (before preprocess_text),
# because preprocess_text lowercases and strips punctuation, which breaks
# capitalised-name detection and email/URL matching.

# ── Emails: standard + obfuscated variants ────────────────────────────────────
_EMAIL = re.compile(
    r"[\w.\-+]+\s*(?:@|\[at\]|\(at\)|＠)\s*[\w.\-]+\s*(?:\.|"
    r"\[dot\]|\(dot\))\s*[a-z]{2,}",
    re.IGNORECASE,
)

# ── Phone numbers: international, local, spaced, dotted, bracketed, bare SG ───
# Bare 8-digit SG numbers: mobile starts with 8/9, home with 6
_PHONE = re.compile(
    r"(?<!\d)(?:"
    r"\+\d{1,3}[\s\-.]?\d{4,5}[\s\-.]?\d{4,5}"   # +65 9123 4567
    r"|\(\d{1,4}\)[\s\-.]?\d{3,5}[\s\-.]?\d{3,5}"  # (65) 9123 4567
    r"|(?<!\d)[689]\d{7}(?!\d)"                           # bare SG 8-digit
    r"|\d{3,5}[\s\-.]\d{3,5}(?:[\s\-.]\d{2,5})?"     # generic separated
    r")(?!\d)",
    re.IGNORECASE,
)

# ── Singapore postal code: 6-digit, optionally preceded by "Singapore" ────────
_SG_POSTAL = re.compile(r"\b(?:singapore\s+|s\()?\d{6}\b", re.IGNORECASE)

# ── Singapore block/unit address ──────────────────────────────────────────────
_SG_ADDR = re.compile(
    r"\b(?:blk|block)\s+\d+\b[^,\n]*"   # Blk 85 Street Name
    r"|\#\d+[\-\u2013]\d+",              # #10-01
    re.IGNORECASE,
)

# ── Generic street addresses ──────────────────────────────────────────────────
_STREET_ADDR = re.compile(
    r"\b\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+"
    r"(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Way|Close|Crescent|Cres|"
    r"Place|Pl|Court|Ct|Boulevard|Blvd|Walk|Rise|View|Hill|Gardens|Park)\b",
    re.IGNORECASE,
)

# ── URLs: with or without http/https, linkedin, github, personal sites ─────────
_URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|(?:linkedin\.com|github\.com|gitlab\.com|behance\.net|"
    r"dribbble\.com|portfolio\.|medium\.com/@?)\S*",
    re.IGNORECASE,
)

# ── Name titles ───────────────────────────────────────────────────────────────
_NAME_TITLES = re.compile(
    r"\b(mr|mrs|ms|miss|dr|prof|sir|assoc\.?\s+prof)\.?\s+",
    re.IGNORECASE,
)

# ── ALL-CAPS names: e.g. "JOHN SMITH" or "TAN WEI MING" ──────────────────────
_ALLCAPS_NAME = re.compile(r"\b([A-Z]{2,15})(?:\s+[A-Z]{2,15}){1,2}\b")

# ── Whitelisted title-case phrases that look like names but aren't ────────────
_NAME_WHITELIST = re.compile(
    r"^("
    r"Software Engineer|Hardware Engineer|Data Scientist|Data Analyst|"
    r"Machine Learning|Deep Learning|Computer Science|Electrical Engineering|"
    r"Civil Engineering|Mechanical Engineering|Chemical Engineering|"
    r"Project Manager|Product Manager|Business Analyst|Systems Engineer|"
    r"Network Engineer|Security Engineer|Cloud Engineer|DevOps Engineer|"
    r"Full Stack|Front End|Back End|Quality Assurance|Test Engineer|"
    r"Research Engineer|Research Scientist|Senior Engineer|Junior Engineer|"
    r"Team Lead|Tech Lead|Chief Executive|Chief Technology|Chief Financial|"
    r"Vice President|Managing Director|General Manager|"
    r"New York|San Francisco|Los Angeles|Hong Kong|Kuala Lumpur|"
    r"South Korea|North America|South East|South Asia|"
    r"January|February|March|April|June|July|August|September|October|"
    r"November|December|Bachelor|Master|Doctor|Honours|"
    r"National Service|Civil Service|"
    r"GCE O|GCE A|Net Aggregate|Non Sensitive|Officially Closed"
    r")$",
    re.IGNORECASE,
)

# ── Title-case names ──────────────────────────────────────────────────────────
_PROPER_NAME = re.compile(r"\b([A-Z][a-z]{1,20})(?:\s+[A-Z][a-z]{1,20}){1,2}\b")

def _safe_name_sub(m: re.Match) -> str:
    if _NAME_WHITELIST.match(m.group(0).strip()):
        return m.group(0)
    return "[Candidate]"

# ── Universities / institutions (run BEFORE name matching) ────────────────────
_UNIVERSITY = re.compile(
    r"\b(?:[A-Z][\w]*\s+)*(?:university|polytechnic|institute\s+of\s+technology)\b"
    r"(?:\s+of\s+[\w\s]{1,30})?",
    re.IGNORECASE,
)
_COLLEGE = re.compile(r"\b[A-Z][\w\s]{0,30}(?:college|academy|school)\b")

# ── Gendered pronouns ──────────────────────────────────────────────────────────
_PRONOUNS = re.compile(r"\b(he|him|his|she|her|hers|himself|herself)\b", re.IGNORECASE)

# ── NRIC / national ID ────────────────────────────────────────────────────────
_NRIC = re.compile(r"\b[STFG]\d{7}[A-Z]\b", re.IGNORECASE)

# ── Nationality / race / Chinese dialects (can bias scoring) ─────────────────
_NATIONALITY = re.compile(
    r"\b(singaporean|malaysian|indonesian|filipino|vietnamese|burmese|"
    r"thai|chinese|indian|malay|eurasian|caucasian|american|british|"
    r"australian|canadian|korean|japanese|"
    r"mandarin|tamil|hokkien|cantonese|teochew|hakka)\b",
    re.IGNORECASE,
)

# ── Date of birth / age ───────────────────────────────────────────────────────
_DOB = re.compile(
    r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|age|born\s+(?:in|on))"
    r"\s*[:\-]?\s*[\d/\-\w,\s]{0,30}",
    re.IGNORECASE,
)

# ── Marital status ────────────────────────────────────────────────────────────
_MARITAL = re.compile(
    r"\b(single|married|divorced|widowed|marital\s+status\s*[:\-]?\s*\w+)\b",
    re.IGNORECASE,
)


def anonymise_text(text: str) -> str:
    """Strip PII from a CV so the LLM scores on skills alone, not identity.

    IMPORTANT: call this on the RAW extracted text, before preprocess_text(),
    so that capitalisation, @ symbols, phone digits and address formats are intact.
    """
    # 1. Contacts & links
    text = _EMAIL.sub("[email]", text)
    text = _URL.sub("[url]", text)
    text = _PHONE.sub("[phone]", text)
    text = _NRIC.sub("[ID]", text)

    # 2. Address components (before name matching to avoid cross-contamination)
    text = _SG_ADDR.sub("[address]", text)
    text = _STREET_ADDR.sub("[address]", text)
    text = _SG_POSTAL.sub("[postal]", text)

    # 3. Demographic markers
    text = _DOB.sub("[dob]", text)
    text = _MARITAL.sub("[marital-status]", text)
    text = _NATIONALITY.sub("[nationality]", text)

    # 4. Institutions BEFORE names (so "Nanyang Technological University" is
    #    caught as a whole unit, not split into a name + leftover keyword)
    text = _UNIVERSITY.sub("[University]", text)
    text = _COLLEGE.sub("[Institution]", text)

    # 5. Titles, then names
    text = _NAME_TITLES.sub("", text)
    text = _ALLCAPS_NAME.sub("[Candidate]", text)
    text = _PROPER_NAME.sub(_safe_name_sub, text)

    # 6. Pronouns
    text = _PRONOUNS.sub("they", text)

    return text
# ── INPUT VALIDATION ──────────────────────────────────────────────────────────
MIN_CV_CHARS  = 200   # anything shorter is probably a blank/corrupt file
MIN_JD_CHARS  = 100
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB


def validate_uploads(jd_uploads, cv_uploads):
    """Return a list of warning strings. Empty list = all good."""
    warnings = []

    # Check for empty upload slots
    if not jd_uploads:
        warnings.append("No Job Description files uploaded.")
    if not cv_uploads:
        warnings.append("No CV files uploaded.")
    if not jd_uploads or not cv_uploads:
        return warnings  # no point checking further

    # Duplicate filenames within each slot
    jd_names = [f.name for f in jd_uploads]
    cv_names = [f.name for f in cv_uploads]
    for name in set(n for n in jd_names if jd_names.count(n) > 1):
        warnings.append(f"Duplicate JD file: '{name}' uploaded more than once.")
    for name in set(n for n in cv_names if cv_names.count(n) > 1):
        warnings.append(f"Duplicate CV file: '{name}' uploaded more than once.")

    # Same filename uploaded in both slots (JD and CV)
    cross = set(jd_names) & set(cv_names)
    for name in cross:
        warnings.append(f"'{name}' was uploaded as both a JD and a CV — is that intentional?")

    # File size
    for f in jd_uploads + cv_uploads:
        size = f.size if hasattr(f, "size") else len(f.getbuffer())
        if size == 0:
            warnings.append(f"'{f.name}' appears to be empty (0 bytes).")
        elif size > MAX_FILE_SIZE:
            warnings.append(f"'{f.name}' is larger than 5 MB — consider splitting it.")

    return warnings


def screen_candidates(
    jd_files: Dict[str, str],
    cv_files: Dict[str, str],
    groq_key: str,
    use_rag: bool = True,
    anonymise: bool = False,
    delay: int = 1,
    progress_callback=None,
    raw_cv_files: Dict[str, str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if not groq_key or not groq_key.strip():
        raise ValueError("Groq API key is required.")
    if not jd_files:
        raise ValueError("No job descriptions loaded.")
    if not cv_files:
        raise ValueError("No CV files loaded.")

    groq_client = Groq(api_key=groq_key.strip())

    # Apply anonymisation to CV text before chunking/screening if requested.
    # IMPORTANT: anonymise_text() must run on the RAW (un-preprocessed) text so
    # that capitalisation, @ signs and phone separators are still present.
    # raw_cv_files should be the original extracted text before preprocess_text().
    if anonymise and raw_cv_files:
        processed_cv_files = {
            name: preprocess_text(anonymise_text(raw_cv_files[name]))
            for name in cv_files
            if name in raw_cv_files
        }
    elif anonymise:
        # Fallback: anonymise already-preprocessed text (less effective but safe)
        processed_cv_files = {name: anonymise_text(text) for name, text in cv_files.items()}
    else:
        processed_cv_files = cv_files

    chunks = _build_chunks(processed_cv_files) if use_rag else []
    emb = get_embedding_model() if use_rag else None

    total_pairs = max(len(jd_files) * len(processed_cv_files), 1)
    step = 0
    all_results = {}

    for jd_name, cleaned_jd in jd_files.items():
        jd_res = []
        for cv_name, cleaned_cv in processed_cv_files.items():
            step += 1
            if progress_callback:
                progress_callback(step, total_pairs, jd_name, cv_name)

            if use_rag:
                ctx = _get_rag_context(chunks, emb, cv_name, cleaned_jd[:500])
                prompt = build_rag_prompt(cleaned_jd, cv_name, ctx)
            else:
                prompt = build_user_prompt(cleaned_jd, cv_name, cleaned_cv)

            try:
                output = _call_llm(groq_client, prompt)
            except Exception as e:
                output = f"Error: {e}"

            jd_res.append({
                "cv_name": cv_name,
                "score": extract_score(output),
                "output": output,
                "summary": extract_section(output, ["Candidate Summary"]),
                "skills": extract_section(output, ["Matching Skills"]),
                "gaps": extract_section(output, ["Missing / Weak Areas", "Missing Areas", "Weak Areas"]),
                "mode": ("RAG" if use_rag else "Direct") + (" · Anon" if anonymise else ""),
            })
            if delay:
                time.sleep(delay)

        jd_res.sort(key=lambda x: x["score"] if x["score"] is not None else -1, reverse=True)
        all_results[jd_name] = jd_res

    return all_results


def build_chat_context(raw_jd_texts, raw_cv_texts, results) -> str:
    """Build a context string kept well under ~2 000 tokens so the full chat
    request (system + context + history + user question) stays inside Groq's
    free-tier per-request limit (~6 000 TPM / ~32 000 token context window).
    Rough budget: context ≤ 1 800 tokens ≈ 7 200 characters."""

    MAX_SUMMARY = 120   # chars per field
    MAX_SKILLS  = 100
    MAX_GAPS    = 100

    ctx = ""

    if raw_jd_texts:
        ctx += "=== JOB DESCRIPTIONS LOADED ===\n"
        for name in raw_jd_texts:
            ctx += f"  - {name}\n"

    if raw_cv_texts:
        ctx += "\n=== CANDIDATE CVs LOADED ===\n"
        for name in raw_cv_texts:
            ctx += f"  - {name}\n"

    if results:
        ctx += "\n=== SCREENING RESULTS ===\n"
        for jd_name, rows in results.items():
            ctx += f"\nJD: {jd_name}\n"
            for row in rows:
                score_str = f"{row['score']}/10" if row['score'] is not None else "N/A"
                verdict   = verdict_label(row['score'])[0]
                ctx += (
                    f"  [{score_str} | {verdict}] "
                    f"{Path(row['cv_name']).stem}\n"
                )
                if row.get("summary"):
                    ctx += f"    Summary: {(row['summary'] or '')[:MAX_SUMMARY]}\n"
                if row.get("skills"):
                    ctx += f"    Skills:  {(row['skills']  or '')[:MAX_SKILLS]}\n"
                if row.get("gaps"):
                    ctx += f"    Gaps:    {(row['gaps']    or '')[:MAX_GAPS]}\n"

    return ctx.strip()


def generate_shortlist_email(
    groq_key: str,
    candidate_name: str,
    job_title: str,
    company_name: str = "our company",
    extra_notes: str = "",
) -> str:
    """Generate a professional interview invitation email for a shortlisted candidate."""
    if not groq_key or not groq_key.strip():
        raise ValueError("Groq API key is required.")

    client = Groq(api_key=groq_key.strip())
    prompt = (
        f"Write a professional, warm interview invitation email to a candidate.\n\n"
        f"Candidate name: {candidate_name}\n"
        f"Job title: {job_title}\n"
        f"Company: {company_name}\n"
        f"Extra notes for personalisation: {extra_notes or 'None'}\n\n"
        f"The email should:\n"
        f"- Have a subject line on the first line starting with 'Subject: '\n"
        f"- Be concise (under 200 words)\n"
        f"- Sound human and warm, not robotic\n"
        f"- Ask them to reply with their availability for an interview\n"
        f"- Leave placeholders like [Interviewer Name] and [Company Email] where needed\n"
        f"- NOT include any commentary before or after the email\n"
        f" /nothink"
    )

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=400,
    )
    raw = response.choices[0].message.content
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

def ask_about_candidates(
    groq_key: str,
    context: str,
    question: str,
    chat_history: list = None,
) -> str:
    """Send a chat question to the LLM, keeping the total request size safe for
    Groq's free tier. Strategy:
      - Context is already capped by build_chat_context (~1 800 tokens).
      - We keep only the last 6 turns of chat history to avoid runaway growth.
      - max_tokens is set to 1 024 — enough for a thorough answer while leaving
        room in the 6 000 TPM budget for the prompt itself.
      - /nothink suffix disables Qwen-3's hidden chain-of-thought, preventing
        hundreds of extra tokens from being generated silently.
    """
    if not groq_key or not groq_key.strip():
        raise ValueError("Groq API key is required.")

    client = Groq(api_key=groq_key.strip())

    messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})

    # Keep only the last 6 exchanges (12 messages) to cap history size
    recent_history = (chat_history or [])[-12:]
    for item in recent_history:
        role = item.get("role")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})

    # Append /nothink to suppress hidden reasoning tokens on Qwen-3
    messages.append({"role": "user", "content": question.rstrip() + " /nothink"})

    response = client.chat.completions.create(
        model=QWEN_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )
    raw = response.choices[0].message.content
    # Strip any <think>...</think> blocks just in case
    clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    return clean
