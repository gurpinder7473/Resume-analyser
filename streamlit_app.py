import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Paths for your uploaded artifacts
# ---------------------------
ARTIFACTS_DIR = os.path.dirname(os.path.abspath(__file__))  # root of repo

FILES = {
    "df": os.path.join(ARTIFACTS_DIR, "resume_dataframe.pkl.bz2"),  # compressed dataframe
    "resume_embeddings": os.path.join(ARTIFACTS_DIR, "resume_embeddings.pkl"),
    "job_embeddings": os.path.join(ARTIFACTS_DIR, "job_embeddings.pkl"),
    "embed_model_name": os.path.join(ARTIFACTS_DIR, "embed_model_name.pkl"),
    "faiss_index": os.path.join(ARTIFACTS_DIR, "faiss_resume_index.idx"),
    "faiss_meta": os.path.join(ARTIFACTS_DIR, "faiss_meta.pkl"),
    "nn_index": os.path.join(ARTIFACTS_DIR, "nn_resume_index.pkl"),
}

# ---------------------------
# Robust pickle / bz2 loader
# ---------------------------
def robust_load(path, expected_type=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    try:
        # pandas can handle bz2 compressed pickles directly
        obj = pd.read_pickle(path)
    except Exception:
        with open(path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")

    if expected_type and not isinstance(obj, expected_type):
        st.warning(f"Expected {expected_type} but got {type(obj)} for {path}")
    return obj

# ---------------------------
# Load FAISS index
# ---------------------------
def load_faiss_index(path):
    import faiss
    if not os.path.exists(path):
        return None
    try:
        return faiss.read_index(path)
    except Exception as e:
        st.warning(f"Failed to load FAISS index: {e}")
        return None

# ---------------------------
# Main artifact loader
# ---------------------------
@st.cache_resource
def load_artifacts():
    # DataFrame
    df = robust_load(FILES["df"], pd.DataFrame)

    # Embeddings
    resume_embeddings = robust_load(FILES["resume_embeddings"])
    job_embeddings = None
    if os.path.exists(FILES["job_embeddings"]):
        job_embeddings = robust_load(FILES["job_embeddings"])

    # FAISS + NN indices
    faiss_index = load_faiss_index(FILES["faiss_index"])
    nn_index = None
    if os.path.exists(FILES["nn_index"]):
        nn_index = robust_load(FILES["nn_index"])

    # Model
    try:
        model_name_obj = robust_load(FILES["embed_model_name"])
        if isinstance(model_name_obj, str):
            model_name = model_name_obj
        elif isinstance(model_name_obj, (list, tuple)) and model_name_obj:
            model_name = model_name_obj[0]
        else:
            model_name = "all-MiniLM-L6-v2"
    except Exception:
        model_name = "all-MiniLM-L6-v2"

    st.write(f"✅ Using model: {model_name}")
    model = SentenceTransformer(model_name)

    return df, resume_embeddings, model

# ---------------------------
# Load artifacts and run app
# ---------------------------
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
