"""
FastAPI backend for AI Autopsy.

Endpoints:
  POST /analyze          — upload model + CSV, start analysis job
  GET  /results/{job_id} — poll for analysis results
  GET  /health           — health check
  GET  /jobs             — list all jobs (debug)

Usage:
  uvicorn api.main:app --reload --port 8000
"""
"""
FastAPI backend — Week 4 hardened version.

Changes from Week 2:
  - Model file validator (rejects non-sklearn files before analysis starts)
  - Timing tracked per job (agent1_s, agent2_s, total_s)
  - Better error messages for common failures
  - /jobs endpoint shows timing info
  - Wake-up endpoint for Render free tier
"""
import os
import uuid
import shutil
import logging
import time
import pickle
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Autopsy API",
    description="Autonomous ML failure investigation system",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}
os.makedirs("tmp", exist_ok=True)


# ── Validators ────────────────────────────────────────────────────────────────

def validate_model_file(file_path: str) -> str:
    """
    Validates that the uploaded file is a loadable sklearn model.
    Returns None if valid, error message string if invalid.

    Called BEFORE starting the background analysis task so the user
    gets an immediate error instead of waiting 30s for a job to fail.
    """
    try:
        import joblib
        model = joblib.load(file_path)
        if not hasattr(model, "predict"):
            return ("File loaded but has no predict() method. "
                    "Please upload a trained sklearn classifier.")
        return None  # valid
    except Exception as e:
        return (f"Could not load model file: {str(e)}. "
                f"Please upload a joblib-saved sklearn .pkl file.")


def validate_csv_file(file_path: str) -> str:
    """
    Validates the mispredictions CSV has required columns.
    Returns None if valid, error message string if invalid.
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        if "predicted" not in df.columns:
            return ("CSV is missing 'predicted' column. "
                    "Use scripts/train_models.py to generate valid CSVs.")
        if "actual" not in df.columns:
            return ("CSV is missing 'actual' column. "
                    "Use scripts/train_models.py to generate valid CSVs.")
        if len(df) == 0:
            return "CSV has 0 rows — no mispredictions to analyse."
        return None  # valid
    except Exception as e:
        return f"Could not read CSV file: {str(e)}"


# ── Background task ───────────────────────────────────────────────────────────

def run_analysis(job_id: str, model_path: str,
                 csv_path: str, model_name: str):
    """Runs full 3-agent pipeline in background."""
    logger.info(f"Job {job_id[:8]} starting — {model_name}")
    jobs[job_id]["status"] = "running"
    total_start = time.time()

    try:
        from src.graph import build_graph, make_initial_state
        app_graph = build_graph()
        state     = make_initial_state(model_path, csv_path, model_name)
        result    = app_graph.invoke(state)
        total_s   = round(time.time() - total_start, 2)

        if result.get("error"):
            jobs[job_id] = {
                "status": "failed",
                "error": result["error"],
                "total_s": total_s,
            }
            logger.error(f"Job {job_id[:8]} failed: {result['error']}")
            return

        jobs[job_id] = {
            "status":         "complete",
            "model_name":     model_name,
            "investigator":   result.get("investigator_output"),
            "counterfactual": result.get("counterfactual_output"),
            "report":         result.get("report_output"),
            "timing":         result.get("timing", {}),
            "total_s":        total_s,
            "error":          None,
        }
        logger.info(f"Job {job_id[:8]} complete in {total_s}s ✓")

    except Exception as e:
        total_s = round(time.time() - total_start, 2)
        logger.error(f"Job {job_id[:8]} crashed after {total_s}s: {e}")
        jobs[job_id] = {
            "status": "failed",
            "error": str(e),
            "total_s": total_s,
        }
    finally:
        tmp_dir = f"tmp/{job_id}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check — also wakes up Render free tier."""
    return {"status": "ok", "version": "1.1.0", "jobs_in_memory": len(jobs)}


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    csv_file:   UploadFile = File(...),
    model_name: str        = "Unknown Model",
):
    """
    Accepts model (.pkl) + mispredictions (.csv).
    Validates both files immediately before starting analysis.
    Returns job_id to poll with GET /results/{job_id}.
    """
    # File extension check
    if not model_file.filename.endswith(".pkl"):
        raise HTTPException(400,
            "Model file must be a .pkl file (joblib-saved sklearn model)")
    if not csv_file.filename.endswith(".csv"):
        raise HTTPException(400,
            "CSV file must be a .csv file")

    # Save to temp folder
    job_id  = str(uuid.uuid4())
    tmp_dir = f"tmp/{job_id}"
    os.makedirs(tmp_dir, exist_ok=True)

    model_path = f"{tmp_dir}/model.pkl"
    csv_path   = f"{tmp_dir}/mispredictions.csv"

    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(csv_file.file, f)

    # Validate model content (before queuing)
    model_error = validate_model_file(model_path)
    if model_error:
        shutil.rmtree(tmp_dir)
        raise HTTPException(400, model_error)

    # Validate CSV content (before queuing)
    csv_error = validate_csv_file(csv_path)
    if csv_error:
        shutil.rmtree(tmp_dir)
        raise HTTPException(400, csv_error)

    # Queue the job
    jobs[job_id] = {"status": "queued", "model_name": model_name}
    background_tasks.add_task(
        run_analysis, job_id, model_path, csv_path, model_name)

    logger.info(f"Job {job_id[:8]} queued — {model_file.filename}")
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Analysis started. Poll GET /results/{job_id} every 3s."
    }


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Poll for job status and results."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found.")
    return job


@app.get("/jobs")
async def list_jobs():
    """List all jobs with timing info — useful for debugging."""
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id":    jid,
                "status":    j.get("status"),
                "model":     j.get("model_name", "unknown"),
                "total_s":   j.get("total_s"),
            }
            for jid, j in jobs.items()
        ]
    }