from fastapi import FastAPI

from backend.config import configure_logging, load_project_env

configure_logging()
load_project_env()

app = FastAPI(title="ResumeRadar", version="2.0.0")


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}