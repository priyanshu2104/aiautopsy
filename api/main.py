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
import os
import uuid
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Autopsy API",
    description="Autonomous ML failure investigation system",
    version="1.0.0",
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


def run_analysis(job_id: str, model_path: str,
                 csv_path: str, model_name: str):
    """Runs in background after /analyze is called."""
    logger.info(f"Job {job_id[:8]} starting — {model_name}")
    jobs[job_id]["status"] = "running"
    try:
        from src.graph import build_graph, make_initial_state
        app_graph = build_graph()
        state = make_initial_state(model_path, csv_path, model_name)
        result = app_graph.invoke(state)

        if result.get("error"):
            jobs[job_id] = {"status": "failed", "error": result["error"]}
            logger.error(f"Job {job_id[:8]} failed: {result['error']}")
            return

        jobs[job_id] = {
            "status": "complete",
            "model_name": model_name,
            "investigator": result.get("investigator_output"),
            "counterfactual": result.get("counterfactual_output"),
            "report": result.get("report_output"),
            "error": None,
        }
        logger.info(f"Job {job_id[:8]} complete ✓")

    except Exception as e:
        logger.error(f"Job {job_id[:8]} crashed: {e}")
        jobs[job_id] = {"status": "failed", "error": str(e)}
    finally:
        tmp_dir = f"tmp/{job_id}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(...),
    csv_file: UploadFile   = File(...),
    model_name: str        = "Unknown Model",
):
    if not model_file.filename.endswith(".pkl"):
        raise HTTPException(400, "Model file must be a .pkl file")
    if not csv_file.filename.endswith(".csv"):
        raise HTTPException(400, "CSV file must be a .csv file")

    job_id  = str(uuid.uuid4())
    tmp_dir = f"tmp/{job_id}"
    os.makedirs(tmp_dir, exist_ok=True)

    model_path = f"{tmp_dir}/model.pkl"
    csv_path   = f"{tmp_dir}/mispredictions.csv"

    with open(model_path, "wb") as f:
        shutil.copyfileobj(model_file.file, f)
    with open(csv_path, "wb") as f:
        shutil.copyfileobj(csv_file.file, f)

    jobs[job_id] = {"status": "queued"}
    background_tasks.add_task(
        run_analysis, job_id, model_path, csv_path, model_name)

    logger.info(f"Job {job_id[:8]} queued")
    return {"job_id": job_id, "status": "queued",
            "message": "Poll GET /results/{job_id} for progress."}


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return job


@app.get("/jobs")
async def list_jobs():
    return {
        "total": len(jobs),
        "jobs": [
            {"job_id": jid, "status": j.get("status"),
             "model": j.get("model_name", "unknown")}
            for jid, j in jobs.items()
        ]
    }