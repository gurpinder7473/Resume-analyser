import streamlit as st
import pickle, os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = "artifacts"
RESUME_EMB_PATH = os.path.join(ARTIFACTS_DIR, "resume_embeddings.pkl")
JOB_EMB_PATH = os.path.join(ARTIFACTS_DIR, "job_embeddings.pkl")
DF_PATH = os.path.join(ARTIFACTS_DIR, "resume_dataframe.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "embed_model_name.pkl")

st.set_page_config(page_title="Resume Matcher", layout="wide")

@st.cache_resource
def load_artifacts():
    with open(RESUME_EMB_PATH, "rb") as f:
        resume_embeddings = pickle.load(f)
    df = pd.read_pickle(DF_PATH)
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model_name = pickle.load(f)
    else:
        model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    return df, resume_embeddings, model

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
