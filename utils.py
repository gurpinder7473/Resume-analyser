import os, pickle, numpy as np, pandas as pd

def load_artifacts(artifacts_dir="artifacts"):
    # Loads embeddings, dataframe and index metadata from artifacts_dir.
    # Returns: df, resume_embeddings(np.ndarray), job_embeddings(np.ndarray or None), MODEL_NAME, faiss_index_or_None, sklearn_nn_or_None, faiss_meta_or_None
    df_path = os.path.join(artifacts_dir, 'resume_dataframe.pkl')
    re_path = os.path.join(artifacts_dir, 'resume_embeddings.pkl')
    je_path = os.path.join(artifacts_dir, 'job_embeddings.pkl')
    model_name_path = os.path.join(artifacts_dir, 'embed_model_name.pkl')
    nn_path = os.path.join(artifacts_dir, 'nn_resume_index.pkl')
    faiss_meta_path = os.path.join(artifacts_dir, 'faiss_meta.pkl')

    if not os.path.exists(df_path):
        raise FileNotFoundError(f"Required file not found: {df_path}")
    df = pd.read_pickle(df_path)

    if not os.path.exists(re_path):
        raise FileNotFoundError(f"Required file not found: {re_path}")
    with open(re_path, 'rb') as f:
        resume_embeddings = pickle.load(f)

    job_embeddings = None
    if os.path.exists(je_path):
        with open(je_path, 'rb') as f:
            job_embeddings = pickle.load(f)

    MODEL_NAME = 'all-MiniLM-L6-v2'
    if os.path.exists(model_name_path):
        with open(model_name_path, 'rb') as f:
            try:
                MODEL_NAME = pickle.load(f)
            except:
                MODEL_NAME = 'all-MiniLM-L6-v2'

    # Try load FAISS meta (preferred). Otherwise load sklearn NN if present.
    faiss_index = None
    faiss_meta = None
    try:
        if os.path.exists(faiss_meta_path):
            with open(faiss_meta_path, 'rb') as f:
                faiss_meta = pickle.load(f)
            import faiss
            faiss_index = faiss.read_index(faiss_meta['index_path'])
    except Exception as e:
        faiss_index = None
        faiss_meta = None

    nn = None
    if os.path.exists(nn_path):
        with open(nn_path, 'rb') as f:
            nn = pickle.load(f)

    # ensure numpy arrays
    if isinstance(resume_embeddings, list):
        resume_embeddings = np.array(resume_embeddings)
    if job_embeddings is not None and isinstance(job_embeddings, list):
        job_embeddings = np.array(job_embeddings)

    return df, resume_embeddings, job_embeddings, MODEL_NAME, faiss_index, nn, faiss_meta

def search_topk(query_emb, top_k, resume_embeddings, df, index=None, nn=None):
    # query_emb: shape (1, dim)
    # resume_embeddings: np.ndarray (N, dim)
    results = []
    if index is not None:
        # FAISS: index.search returns distances (inner-product for normalized vectors) and indices
        import faiss, numpy as _np
        q = query_emb.astype('float32').copy()
        # If the index was built with normalized vectors, faiss expects normalized query
        try:
            faiss.normalize_L2(q)
        except Exception:
            pass
        D, I = index.search(q, top_k)
        # If index is IndexFlatIP over normalized vectors, D is inner product (similarity)
        for score, idx in zip(D[0], I[0]):
            # try to coerce values to native types
            results.append({"resume_index": int(idx), "similarity_score": float(score), "resume_text": str(df.iloc[int(idx)].get('resume_text',''))})
        return results
    elif nn is not None:
        distances, indices = nn.kneighbors(query_emb, n_neighbors=top_k)
        # sklearn NN with metric='cosine' returns distances, smaller is closer
        for dist, idx in zip(distances[0], indices[0]):
            sim = 1 - float(dist)
            results.append({"resume_index": int(idx), "similarity_score": float(sim), "resume_text": str(df.iloc[int(idx)].get('resume_text',''))})
        return results
    else:
        # Fallback: brute-force cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(query_emb, resume_embeddings)[0]
        top_idx = sims.argsort()[::-1][:top_k]
        for idx in top_idx:
            results.append({"resume_index": int(idx), "similarity_score": float(sims[int(idx)]), "resume_text": str(df.iloc[int(idx)].get('resume_text',''))})
        return results
