# ---------------------------
# Artifact download + robust loader (Google Drive)
# ---------------------------
import streamlit as st
import os, pickle, pandas as pd, requests, math
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ---------------------------
# Put the Drive file IDs you already gave here.
# If you upload nn_resume_index.pkl to Drive, add its file id to the dict.
# ---------------------------
FILE_IDS = {
    "resume_dataframe.pkl": "1715Wxq8_drtgL93vtedB-F_7gQznl6RT",
    "resume_embeddings.pkl": "12DsAMfvVjyZ7yuewebT7XwD0-jfcLpVL",
    "job_embeddings.pkl": "1SSm7k6wX9rSjpPhzO8tdRwOppy05iOky",
    "embed_model_name.pkl": "1q3h1JFJfpFmatWg1R-PxxDTa107OYzm_",
    "faiss_meta.pkl": "1A178G08uz0CSiUTRhAZVSsqxqqbtDMl46"  # <-- double-check this id if needed
    # NOTE: I put the IDs you provided earlier for the last two files below.
    # If your faiss index file has the other id, switch them.
    , "faiss_resume_index.idx": "1nvbnFAZrwRZM0w11rOUXMZACMN7QI7D3",
    # Add nn_resume_index.pkl ID here if/when you upload it to Drive:
    "nn_resume_index.pkl": ""  # <-- fill this with Drive file id if you upload it
}

# Local paths used by your app
DF_PATH = os.path.join(ARTIFACTS_DIR, "resume_dataframe.pkl")
RESUME_EMB_PATH = os.path.join(ARTIFACTS_DIR, "resume_embeddings.pkl")
JOB_EMB_PATH = os.path.join(ARTIFACTS_DIR, "job_embeddings.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "embed_model_name.pkl")
FAISS_META_PATH = os.path.join(ARTIFACTS_DIR, "faiss_meta.pkl")
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss_resume_index.idx")
NN_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "nn_resume_index.pkl")

# ---------------------------
# Helper: detect HTML (Drive error pages)
# ---------------------------
def is_probably_html(path):
    try:
        with open(path, "rb") as f:
            head = f.read(2048)
        txt = head.decode("utf-8", errors="ignore").lower()
        return "<html" in txt or "google drive" in txt or "quota" in txt or "sign in" in txt
    except Exception:
        return False

# ---------------------------
# Robust Google Drive downloader (handles confirm token)
# ---------------------------
def download_from_gdrive(file_id: str, dest: str, show_progress=True):
    if not file_id:
        raise ValueError("No file_id provided for download.")

    session = requests.Session()
    URL = "https://docs.google.com/uc?export=download"
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for k, v in response.cookies.items():
        if k.startswith('download_warning') or 'download' in k:
            token = v
            break
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    total = response.headers.get('Content-Length')
    if total is not None:
        total = int(total)

    # write file with progress
    chunk_size = 32768
    written = 0
    if show_progress and total:
        prog = st.progress(0)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                written += len(chunk)
                if show_progress and total:
                    prog.progress(min(100, math.floor((written/total)*100)))
    if show_progress and total:
        prog.progress(100)
    return dest

# ---------------------------
# Ensure a named artifact is present locally. Behavior:
# - If file exists and looks valid -> use it
# - If missing and a FILE_ID exists -> download
# - If missing and no FILE_ID -> look in /mnt/data (dev/test) or raise
# ---------------------------
def ensure_artifact(filename, max_retries=2):
    local_path = os.path.join(ARTIFACTS_DIR, filename)

    # already present and looks ok
    if os.path.exists(local_path) and not is_probably_html(local_path):
        return local_path

    file_id = FILE_IDS.get(filename, "") or ""
    # try container upload locations first (useful in dev/testing)
    alt_paths = [os.path.join("/mnt/data", filename), filename]  # check working dir too
    for p in alt_paths:
        if os.path.exists(p) and not is_probably_html(p):
            # copy into artifacts dir for consistent behavior
            try:
                with open(p, "rb") as r, open(local_path, "wb") as w:
                    w.write(r.read())
                return local_path
            except Exception:
                pass

    # if we have a drive id, attempt download (with retries if we detect html)
    if file_id:
        for attempt in range(max_retries):
            try:
                download_from_gdrive(file_id, local_path, show_progress=True)
            except Exception as e:
                st.warning(f"Download attempt {attempt+1} for {filename} failed: {e}")
                continue
            if os.path.exists(local_path) and not is_probably_html(local_path):
                return local_path
            else:
                st.warning(f"Downloaded file for {filename} looks invalid (HTML). Retrying...")
        # exhausted retries
        st.error(f"Could not obtain a valid copy of {filename} from Drive after {max_retries} attempts.")
        return None

    # no drive id and not found
    st.warning(f"No Drive ID for {filename} and file not present locally.")
    return None

# ---------------------------
# Robust pickle loader (DataFrame, embeddings, other pickled objects)
# ---------------------------
def robust_load_pickle(local_path, expected_type=None):
    if local_path is None:
        raise FileNotFoundError("Provided path is None")

    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path + " not found")

    # detect html
    if is_probably_html(local_path):
        raise RuntimeError(f"{local_path} appears to be an HTML page (Drive error).")

    # try pandas pickles first (fast path for DataFrame)
    try:
        obj = pd.read_pickle(local_path)
    except Exception:
        # fallback to plain pickle.load (latin1 for py2->py3 compatibility)
        with open(local_path, "rb") as f:
            obj = pickle.load(f, encoding="latin1")
    if expected_type and not isinstance(obj, expected_type):
        st.warning(f"Expected {expected_type} but got {type(obj)} when loading {local_path}")
    return obj

# ---------------------------
# Load FAISS index if present
# ---------------------------
def load_faiss_index(index_path):
    try:
        import faiss
    except Exception as e:
        st.error("faiss is not available in the environment. faiss-cpu must be installed in requirements.")
        raise e
    if not os.path.exists(index_path):
        raise FileNotFoundError(index_path + " not found")
    # faiss.read_index works with the index file path
    return faiss.read_index(index_path)

# ---------------------------
# Main cached loader used by the rest of your app
# ---------------------------
@st.cache_resource
def load_artifacts():
    # Ensure and load DataFrame
    df_local = ensure_artifact("resume_dataframe.pkl")
    if not df_local:
        st.error("resume_dataframe.pkl missing — upload to Drive or place in artifacts/")
        raise FileNotFoundError("resume_dataframe.pkl missing")
    df = robust_load_pickle(df_local, expected_type=pd.DataFrame)

    # Embeddings (numpy array)
    emb_local = ensure_artifact("resume_embeddings.pkl")
    if not emb_local:
        st.error("resume_embeddings.pkl missing — upload to Drive or place in artifacts/")
        raise FileNotFoundError("resume_embeddings.pkl missing")
    resume_embeddings = robust_load_pickle(emb_local)  # may be numpy array or list

    # Optional: job embeddings
    job_emb_local = ensure_artifact("job_embeddings.pkl")
    job_embeddings = None
    if job_emb_local:
        job_embeddings = robust_load_pickle(job_emb_local)

    # FAISS meta & index (optional)
    faiss_meta_local = ensure_artifact("faiss_meta.pkl")
    faiss_index_local = ensure_artifact("faiss_resume_index.idx")
    faiss_index = None
    if faiss_index_local:
        try:
            faiss_index = load_faiss_index(faiss_index_local)
        except Exception as e:
            st.warning(f"Failed to load faiss index: {e}")

    # NN index (pickled nearest-neighbor index, optional)
    nn_local = ensure_artifact("nn_resume_index.pkl")
    nn_index = None
    if nn_local:
        try:
            nn_index = robust_load_pickle(nn_local)
        except Exception as e:
            st.warning(f"Failed to load nn_resume_index.pkl: {e}")

    # Model name / embedding model
    model_name_local = ensure_artifact("embed_model_name.pkl")
    if model_name_local:
        try:
            model_name_obj = robust_load_pickle(model_name_local)
            # model_name might be stored as string or inside container; guard it
            if isinstance(model_name_obj, str):
                model_name = model_name_obj
            elif isinstance(model_name_obj, (list, tuple)) and len(model_name_obj) > 0:
                model_name = str(model_name_obj[0])
            else:
                model_name = "all-MiniLM-L6-v2"
        except Exception:
            model_name = "all-MiniLM-L6-v2"
    else:
        model_name = "all-MiniLM-L6-v2"

    st.write(f"Using sentence-transformers model: {model_name}")
    model = SentenceTransformer(model_name)

    # Return a dict with everything; your original code expects df, resume_embeddings, model
    return {
        "df": df,
        "resume_embeddings": resume_embeddings,
        "job_embeddings": job_embeddings,
        "faiss_index": faiss_index,
        "faiss_meta_path": faiss_meta_local,
        "nn_index": nn_index,
        "model": model
    }
# End of artifact loader
