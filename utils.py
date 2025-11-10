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

    # ---------- Required files ----------
    df_path = os.path.join(artifacts_dir, "resume_dataframe.pkl")
    resume_emb_path = os.path.join(artifacts_dir, "resume_embeddings.pkl")
    job_emb_path = os.path.join(artifacts_dir, "job_embeddings.pkl")
    model_name_path = os.path.join(artifacts_dir, "embed_model_name.pkl")

    # ---------- Optional files ----------
    nn_path = os.path.join(artifacts_dir, "nn_resume_index.pkl")
    faiss_idx_path = os.path.join(artifacts_dir, "faiss_resume_index.idx")
    faiss_meta_path = os.path.join(artifacts_dir, "faiss_meta.pkl")

    # Load dataframe + embeddings
    df = load_pickle(df_path)
    resume_embeddings = load_pickle(resume_emb_path)

    # Optional job embeddings (if multi-job mode enabled)
    job_embeddings = None
    if os.path.exists(job_emb_path):
        job_embeddings = load_pickle(job_emb_path)

    # Load embed model
    model_name = load_pickle(model_name_path)
    model = SentenceTransformer(model_name)

    # Optional sklearn NearestNeighbors index
    sklearn_nn = None
    if os.path.exists(nn_path):
        sklearn_nn = load_pickle(nn_path)

    # Optional FAISS index
    faiss_index = None
    faiss_meta = None
    if os.path.exists(faiss_idx_path) and os.path.exists(faiss_meta_path):
        try:
            import faiss
            faiss_index = faiss.read_index(faiss_idx_path)
            faiss_meta = load_pickle(faiss_meta_path)
        except Exception as e:
            print(f"⚠️ FAISS could not be loaded: {e}")
            faiss_index = None
            faiss_meta = None

    return df, resume_embeddings, job_embeddings, model, faiss_index, sklearn_nn, faiss_meta
