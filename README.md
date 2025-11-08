# Resume Matcher Backend (FastAPI)

This backend provides a Resume Matching API that ranks resumes by similarity to a job description.
It expects precomputed artifacts to exist under an `artifacts/` directory (created by your Colab notebook).

## Required artifacts (place under `artifacts/`):
- resume_embeddings.pkl        (numpy array, shape: [N, dim])
- job_embeddings.pkl           (optional)
- resume_dataframe.pkl         (pandas DataFrame with at least 'resume_text' column)
- embed_model_name.pkl         (string model name, e.g., 'all-MiniLM-L6-v2')
- nn_resume_index.pkl          (optional sklearn NearestNeighbors index) OR
- faiss_resume_index.idx + faiss_meta.pkl  (optional FAISS index + meta)

## How to run locally
1. Create a Python virtual environment
2. Install requirements: `pip install -r requirements.txt`
3. Ensure `artifacts/` folder is present next to this app and contains the files above.
4. Run: `uvicorn app:app --reload --port 8000`

## API Endpoints
- GET /                -> health check
- POST /match-resume   -> body: {"text":"<job description>", "top_k":5}

## Notes
- If FAISS index is present it will be used for fast retrieval. Otherwise the sklearn NN index (if present) will be used.
- The embedding model name is loaded from `embed_model_name.pkl`. The server will load the SentenceTransformer model at startup.
