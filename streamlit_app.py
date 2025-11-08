import streamlit as st
import pickle
import os
import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# CONFIG: ARTIFACT FOLDER & GOOGLE DRIVE FILE IDS
# -------------------------
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Map filename -> Google Drive file id (use the IDs you uploaded)
FILE_IDS = {
    "resume_dataframe.pkl": "1715Wxq8_drtgL93vtedB-F_7gQznl6RT",
    "resume_embeddings.pkl": "12DsAMfvVjyZ7yuewebT7XwD0-jfcLpVL",
    "job_embeddings.pkl": "1SSm7k6wX9rSjpPhzO8tdRwOppy05iOky",
    "embed_model_name.pkl": "1q3h1JFJfpFmatWg1R-PxxDTa107OYzm_",
    # If you have other files, add them here with the correct filenames as keys
    # "some_other.pkl": "1A178G08uz0CSiUTRhAZVSsqxqbtDMl46",
    # "extra_file.pkl": "1nvbnFAZrwRZM0w11rOUXMZACMN7QI7D3",
}

# Local paths used by the rest of your code (kept names consistent)
RESUME_EMB_PATH = os.path.join(ARTIFACTS_DIR, "resume_embeddings.pkl")
JOB_EMB_PATH = os.path.join(ARTIFACTS_DIR, "job_embeddings.pkl")
DF_PATH = os.path.join(ARTIFACTS_DIR, "resume_dataframe.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "embed_model_name.pkl")

# -------------------------
# GOOGLE DRIVE DOWNLOADER
# -------------------------
def download_from_google_drive(file_id: str, destination: str):
    """
    Download a possibly-large file from Google Drive, handling the
    'large file' confirmation token if required.
    """
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, val in response.cookies.items():
        if key.startswith("download_warning"):
            token = val
            break
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def ensure_file_downloaded(filename: str):
    """Ensure a specific artifact is present locally; download from Drive if missing."""
    local_path = os.path.join(ARTIFACTS_DIR, filename)
    if os.path.exists(local_path):
        return local_path

    file_id = FILE_IDS.get(filename)
    if not file_id:
        # Not configured to download this file; caller should handle the missing file.
        return None

    try:
        st.info(f"Downloading {filename} from Google Drive (first-run)...")
        download_from_google_drive(file_id, local_path)
        st.success(f"{filename} downloaded.")
        return local_path
    except Exception as e:
        st.error(f"Failed to download {filename}: {e}")
        return None

# Ensure required files exist locally (download if possible)
# We'll attempt to ensure the DataFrame and model-name file, then try embeddings.
ensure_file_downloaded("resume_dataframe.pkl")
ensure_file_downloaded("embed_model_name.pkl")
ensure_file_downloaded("resume_embeddings.pkl")
ensure_file_downloaded("job_embeddings.pkl")  # optional; safe if missing

# Show where artifacts are loaded from (useful debug)
st.write("Using artifacts directory:", ARTIFACTS_DIR)
st.write("Files present:", sorted(os.listdir(ARTIFACTS_DIR)))

# -------------------------
# CACHED LOADERS & FALLBACK EMBEDDING COMPUTE
# -------------------------
@st.cache_resource
def get_model():
    # Load model name from MODEL_PATH if present; otherwise default
    model_name = "all-MiniLM-L6-v2"
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, str):
                model_name = loaded
            else:
                # if it was stored differently (e.g., bytes), attempt str conversion
                model_name = str(loaded)
        except Exception:
            # fallback to default
            model_name = "all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)

@st.cache_data
def load_dataframe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"DataFrame file not found at {path}")
    return pd.read_pickle(path)

@st.cache_data
def load_or_build_embeddings(df, emb_path, model):
    # If embeddings file exists, load it; otherwise compute from df and save
    if os.path.exists(emb_path):
        with open(emb_path, "rb") as f:
            return pickle.load(f)

    # Compute embeddings (expects df to have a 'resume_text' column)
    if "resume_text" not in df.columns:
        raise ValueError("DataFrame must contain a 'resume_text' column to build embeddings.")
    st.info("Resume embeddings missing — computing embeddings now (first-run, may take time)...")
    texts = df["resume_text"].astype(str).tolist()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    with open(emb_path, "wb") as f:
        pickle.dump(emb, f)
    st.success("Resume embeddings computed and saved.")
    return emb

# -------------------------
# LOAD ARTIFACTS (keeps original logic)
# -------------------------
st.set_page_config(page_title="Resume Matcher", layout="wide")

@st.cache_resource
def load_artifacts():
    # DataFrame
    if not os.path.exists(DF_PATH):
        # try to download if not present
        ensure_file_downloaded("resume_dataframe.pkl")
    df = load_dataframe(DF_PATH)

    # Model
    model = get_model()

    # Embeddings: attempt download, else compute
    if not os.path.exists(RESUME_EMB_PATH):
        ensure_file_downloaded("resume_embeddings.pkl")
    resume_embeddings_local = load_or_build_embeddings(df, RESUME_EMB_PATH, model)

    return df, resume_embeddings_local, model

# Keep the rest of your app logic unchanged
df, resume_embeddings, model = load_artifacts()

st.title("📄 Resume Matcher — AI Job Resume Screening")
st.markdown("Upload or paste your **Job Description**, and the app will show the most relevant resumes.")

job_desc = st.text_area("📝 Job Description", height=200, placeholder="Paste job description here...")
top_k = st.slider("Select number of top resumes to show", 1, 20, 5)
search_button = st.button("🔍 Find Matching Resumes")

if search_button and job_desc.strip():
    with st.spinner("Encoding job description and finding matches..."):
        job_emb = model.encode([job_desc], convert_to_numpy=True)
        sims = cosine_similarity(job_emb, resume_embeddings)[0]
        top_idx = np.argsort(-sims)[:top_k]
        results = [(int(i), float(sims[i]), df.iloc[i]['resume_text'][:1500]) for i in top_idx]

    st.success(f"Found Top {top_k} Matching Resumes")
    for rank, (idx, score, text) in enumerate(results, start=1):
        st.markdown(f"### #{rank} — Similarity Score: {score:.4f}")
        st.write(text)
        st.divider()
elif search_button:
    st.warning("Please enter a valid job description.")

st.sidebar.title("ℹ️ About")
st.sidebar.info("This app uses **SentenceTransformer** embeddings to compute similarity between resumes and job descriptions.")
