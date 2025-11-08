import streamlit as st
import pandas as pd
import numpy as np
import pickle
import bz2
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Artifact paths ---
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

FILES = {
    "df": os.path.join(ARTIFACTS_DIR, "resume_dataframe.pkl.bz2"),
    "resume_embeddings": os.path.join(ARTIFACTS_DIR, "resume_embeddings.pkl"),
    "job_embeddings": os.path.join(ARTIFACTS_DIR, "job_embeddings.pkl"),
    "faiss_index": os.path.join(ARTIFACTS_DIR, "faiss_resume_index.idx"),
    "faiss_meta": os.path.join(ARTIFACTS_DIR, "faiss_meta.pkl"),
    "embed_model_name": os.path.join(ARTIFACTS_DIR, "embed_model_name.pkl"),
    "nn_index": os.path.join(ARTIFACTS_DIR, "nn_resume_index.pkl"),
}

# --- Helper functions ---
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_dataframe(path):
    if path.endswith(".bz2"):
        with bz2.BZ2File(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)

@st.cache_resource
def load_artifacts():
    # Load DataFrame
    df = load_dataframe(FILES["df"])

    # Load model name and initialize SentenceTransformer
    model_name = load_pickle(FILES["embed_model_name"])
    model = SentenceTransformer(model_name)

    # Load embeddings
    resume_embeddings = load_pickle(FILES["resume_embeddings"])

    return df, resume_embeddings, model

# --- Load everything ---
st.title("📄 Resume Matcher — AI Job Resume Screening")
st.markdown("Upload or paste your **Job Description**, and the app will show the most relevant resumes.")

try:
    df, resume_embeddings, model = load_artifacts()
    st.sidebar.success("✅ Artifacts loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Error loading artifacts: {e}")
    st.stop()

# --- App logic ---
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
