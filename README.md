# 💼 AI Job Recommender

An AI-powered job recommendation system built with **Streamlit**, **Apify API**, **Google Gemini**, **FAISS**, and **SentenceTransformers**.  
It allows you to upload your resume or skills, fetch real LinkedIn jobs, get AI-powered recommendations, perform semantic search, and export results.

---

## 📌 How the Project Works (Step-by-Step)

### 1️⃣ **Setup and Environment Variables**
- A `.env` file stores sensitive API keys for:
  - **Apify API** → Fetching LinkedIn jobs.
  - **Google Gemini API** → AI-based job recommendations.
- These keys are loaded using `python-dotenv`.

---

### 2️⃣ **Resume Upload and Parsing**
- You can upload **PDF** or **TXT** resumes.
- **PyPDF2** extracts text from PDFs page by page.
- For TXT files, UTF-8 or Latin-1 decoding is used to handle different encodings.
- Extracted resume text is later sent to Gemini for AI processing.

---

### 3️⃣ **AI Job Recommendations (Gemini)**
- The extracted resume text (plus any skills entered manually) is sent to **Google Gemini**.
- Gemini is prompted to return job recommendations in **JSON format** with:
  - `title`, `company`, `location`, `type`, `description`, `url`
- These AI-suggested jobs are added to the job list in the app.

---

### 4️⃣ **Fetching Jobs from LinkedIn (Apify API)**
- Apify’s LinkedIn Scraper Actor is called with:
  - Job title
  - Location
  - Number of job listings
- The scraper returns job details, which are added to the job list.
- Missing data is handled gracefully.

---

### 5️⃣ **Text Cleaning & Chunking**
- A small cleaning step removes extra spaces.
- Text chunking is available to split long content into manageable pieces (for embeddings if needed).

---

### 6️⃣ **Semantic Search with FAISS**
- All job descriptions are turned into **embeddings** using:
  - **SentenceTransformer:** `all-MiniLM-L6-v2` model.
- **FAISS** stores these embeddings in a vector index.
- When you search, FAISS finds jobs with the closest semantic meaning to your query.

---

### 7️⃣ **Streamlit UI**
- **Sidebar:**
  - Upload resume
  - Enter skills
  - Set preferred location and job type
  - Select number of AI recommendations
  - Input LinkedIn search parameters
- **Main Area:**
  - Buttons to get AI recommendations, fetch LinkedIn jobs, and export CSV.
  - Build FAISS index for semantic search.
  - Search bar to find jobs by meaning.
  - Data table showing all collected jobs.

---

### 8️⃣ **Exporting Jobs**
- All jobs (AI + LinkedIn) can be downloaded as a CSV file.
- This makes it easy to review or share job opportunities later.

---

---

## 🛠 Tech Stack
- **Python**
- **Streamlit** – UI and interactivity
- **Apify API** – LinkedIn job data
- **Google Gemini API** – AI recommendations
- **PyPDF2** – Resume text extraction
- **SentenceTransformers** – Text embeddings
- **FAISS** – Semantic search

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/your-username/ai-job-recommender.git
cd ai-job-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
