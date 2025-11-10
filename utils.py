# utils.py

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_pickle(path):
    """Loads pickle file safely"""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts(artifacts_dir="artifacts"):
    """
    Loads embeddings, dataframe, and index metadata from artifacts_dir.
    Returns:
        df,
        resume_embeddings (np.ndarray),
        job_embeddings (np.ndarray or None),
        model (SentenceTransformer),
        faiss_index_or_None,
        sklearn_nn_or_None,
        faiss_meta_or_None
    """

    print(f"📂 Loading artifacts from: {artifacts_dir}")

    # ========== Required Artifacts (must exist) ==========
    df_path = os.path.join(artifacts_dir, "resume_dataframe.pkl")
    resume_emb_path = os.path.join(artifacts_dir, "resume_embeddings.pkl")
    job_emb_path = os.path.join(artifacts_dir, "job_embeddings.pkl")
    model_name_path = os.path.join(artifacts_dir, "embed_model_name.pkl")

    # ========== Optional (FAISS / sklearn) ==========
    nn_path = os.path.join(artifacts_dir, "nn_resume_index.pkl")
    faiss_idx_path = os.path.join(artifacts_dir, "faiss_resume_index.idx")
    faiss_meta_path = os.path.join(artifacts_dir, "faiss_meta.pkl")

    # Load dataframe + embeddings
    df = load_pickle(df_path)
    resume_embeddings = load_pickle(resume_emb_path)

    # Job embedding may not exist if mode = resume-only matching
    job_embeddings = None
    if os.path.exists(job_emb_path):
        job_embeddings = load_pickle(job_emb_path)

    # Load model and initialize
    model_name = load_pickle(model_name_path)
    model = SentenceTransformer(model_name)

    # Load optional sklearn NN
    sklearn_nn = None
    if os.path.exists(nn_path):
        sklearn_nn = load_pickle(nn_path)

    # Load FAISS index if exists
    faiss_index = None
    faiss_meta = None
    if os.path.exists(faiss_idx_path) and os.path.exists(faiss_meta_path):
        import faiss
        faiss_index = faiss.read_index(faiss_idx_path)
        faiss_meta = load_pickle(faiss_meta_path)

    return df, resume_embeddings, job_embeddings, model, faiss_index, sklearn_nn, faiss_meta

