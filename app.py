# backend app for Resume Matching (FastAPI)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle, os, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Optional
from utils import load_artifacts, search_topk

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")

app = FastAPI(title="Resume Matcher API")

# Allow all CORS for ease of testing (change in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# -----------------------------
# Load artifacts at startup
# -----------------------------
@app.on_event("startup")
def startup_event():
    global df, resume_embeddings, job_embeddings, embed_model, index, nn, MODEL_NAME, FAISS_META
    df, resume_embeddings, job_embeddings, MODEL_NAME, index, nn, FAISS_META = load_artifacts(ARTIFACTS_DIR)
    embed_model = SentenceTransformer(MODEL_NAME)
    print(f"✅ Loaded artifacts from {ARTIFACTS_DIR} | resumes={len(df)} | emb_dim={resume_embeddings.shape[1]}")

class JobDescription(BaseModel):
    text: str
    top_k: Optional[int] = 5

@app.get("/", tags=["health"])
def root():
    return {"message":"Resume Matcher API running"}

@app.post("/match-resume", tags=["inference"])
def match_resume(payload: JobDescription):
    if payload.text is None or payload.text.strip()=="":
        raise HTTPException(status_code=400, detail="Job description text is required.")
    jd = payload.text
    top_k = int(payload.top_k or 5)
    # create embedding for query jd
    jd_emb = embed_model.encode([jd], convert_to_numpy=True)
    results = search_topk(jd_emb, top_k, resume_embeddings, df, index=index, nn=nn)
    return {"top_matches": results, "count": len(results)}
