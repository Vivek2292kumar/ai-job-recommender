import os
import io
import re
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader

# ---------- Load environment variables ----------
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# ---------- Apify LinkedIn Fetcher ----------
from apify_client import ApifyClient

def fetch_linkedin_jobs(search_query: str, location: str = "India", rows: int = 60):
    token = os.getenv("APIFY_API_TOKEN")
    if not token:
        raise RuntimeError("❌ APIFY_API_TOKEN not found in .env")
    ACTOR_ID = os.getenv("APIFY_LINKEDIN_ACTOR_ID", "BHzefUZlZRKWxkTck")
    client = ApifyClient(token)
    run_input = {
        "title": search_query,
        "location": location,
        "rows": rows,
        "sortby": "relevance",
        "freshness": "all",
        "experience": "all",
        "proxy": {"useApifyProxy": True, "apifyProxyGroups": ["RESIDENTIAL"]},
    }
    run = client.actor(ACTOR_ID).call(run_input=run_input)
    dataset_id = run.get("defaultDatasetId") or run.get("defaultDatasetID")
    if not dataset_id:
        raise RuntimeError("❌ No dataset ID from Apify run.")
    items = list(client.dataset(dataset_id).iterate_items())
    jobs = []
    for it in items:
        jobs.append({
            "title": it.get("title"),
            "company": it.get("companyName") or it.get("company"),
            "location": it.get("location"),
            "type": it.get("workType") or it.get("type") or "",
            "description": it.get("description") or it.get("jobDescription") or "",
            "url": it.get("jobUrl") or it.get("applyUrl") or "",
        })
    return [j for j in jobs if j["title"] and j["url"]]

# ---------- Resume Parser ----------
def parse_resume_text(file_bytes: bytes, mime_type: str):
    if mime_type == "application/pdf" or mime_type.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)
    else:
        try:
            return file_bytes.decode("utf-8")
        except Exception:
            return file_bytes.decode("latin-1")

# ---------- Text Cleaning & Chunking ----------
def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ---------- Embeddings & FAISS ----------
from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
FAISS_INDEX = None
FAISS_METADATA = []

def create_faiss_index(jobs):
    global FAISS_INDEX, FAISS_METADATA
    docs = [f"{job['title']} {job['description']}" for job in jobs]
    FAISS_METADATA = jobs
    vectors = EMBED_MODEL.encode(docs)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    FAISS_INDEX = index

def search_faiss(query, k=5):
    if FAISS_INDEX is None:
        return []
    vec = EMBED_MODEL.encode([query])
    D, I = FAISS_INDEX.search(vec, k)
    return [FAISS_METADATA[i] for i in I[0]]

# ---------- Gemini Job Retriever ----------
import google.generativeai as genai
class JobRetriever:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not set in .env")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def retrieve(self, user_text, location=None, job_type=None, top_k=10):
        filters = []
        if location:
            filters.append(f"Location: {location}")
        if job_type and job_type.lower() != "any":
            filters.append(f"Type: {job_type}")
        filters_text = "\n".join(filters) if filters else "No specific filters."
        prompt = f"""
        You are a job recommendation AI.
        Based on the following user profile, return ONLY a JSON array with exactly {top_k} job postings.
        Fields: title, company, location, type, description, url.
        Profile: {user_text}
        Filters: {filters_text}
        """
        response = self.model.generate_content(prompt)
        try:
            return json.loads(response.text.strip())
        except:
            return []
        


        
# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Job Recommender", layout="wide")
st.title("💼 AI Job Recommender")

uploaded_resume = st.sidebar.file_uploader("Upload resume (PDF or TXT)", type=["pdf", "txt"])
skills = st.sidebar.text_input("Skills (comma separated)")
location = st.sidebar.text_input("Preferred location")
job_type = st.sidebar.selectbox("Job type", ["Any", "Full-time", "Part-time", "Contract"])
max_recs = st.sidebar.slider("Max AI recommendations", 1, 10, 5)
search_query = st.sidebar.text_input("LinkedIn search title", "Data Scientist")
rows = st.sidebar.slider("LinkedIn rows", 5, 50, 20)

retriever = JobRetriever()

if "jobs" not in st.session_state:
    st.session_state.jobs = []

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🤖 AI Job Recommendations"):
        if not uploaded_resume and not skills:
            st.error("Upload a resume or enter skills.")
        else:
            resume_text = ""
            if uploaded_resume:
                resume_text = parse_resume_text(uploaded_resume.read(), uploaded_resume.type)
            resume_text += "\n" + skills
            jobs = retriever.retrieve(resume_text, location, job_type, max_recs)
            st.session_state.jobs.extend(jobs)
            st.success(f"Added {len(jobs)} AI jobs.")

with col2:
    if st.button("🕸 Fetch from LinkedIn (Apify)"):
        try:
            apify_jobs = fetch_linkedin_jobs(search_query, location or "India", rows)
            st.session_state.jobs.extend(apify_jobs)
            st.success(f"Fetched {len(apify_jobs)} jobs from LinkedIn.")
        except Exception as e:
            st.error(str(e))

with col3:
    if st.button("💾 Export CSV"):
        if not st.session_state.jobs:
            st.warning("No jobs to export.")
        else:
            df = pd.DataFrame(st.session_state.jobs)
            st.download_button("Download jobs.csv", df.to_csv(index=False), "jobs.csv", "text/csv")

if st.button("📚 Build FAISS index"):
    create_faiss_index(st.session_state.jobs)
    st.success("FAISS index built.")

search_term = st.text_input("🔍 Semantic Search in Jobs")
if st.button("Search"):
    results = search_faiss(search_term)
    for job in results:
        st.markdown(f"### [{job['title']}]({job['url']})")
        st.write(f"**Company:** {job['company']} | **Location:** {job['location']} | **Type:** {job['type']}")
        st.write(job['description'])

if st.session_state.jobs:
    st.subheader("All Jobs")
    st.dataframe(pd.DataFrame(st.session_state.jobs))
