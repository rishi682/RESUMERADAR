from fastapi import FastAPI

from backend.api.routes import router
from backend.config import configure_logging, load_project_env

configure_logging()
load_project_env()

app = FastAPI(title="ResumeRadar", version="2.0.0")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}