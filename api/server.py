"""FastAPI web interface for the Medical Rehab Agent System."""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List

from rag.retriever import RAGRetriever
from agents.dispatcher import DispatcherAgent
from utils.report import to_structured_json, to_pdf

app = FastAPI(
    title="Medical Rehab Agent System",
    description="Multi-Agent + RAG system for language/psychological rehabilitation",
    version="0.1.0",
)

# ── Mount static files ─────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Global state (initialized on startup) ──────────────────────
_dispatcher: Optional[DispatcherAgent] = None


@app.on_event("startup")
def startup():
    global _dispatcher
    retriever = RAGRetriever()
    retriever.initialize()
    _dispatcher = DispatcherAgent(retriever)


# ── Serve frontend ─────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


# ── Request / Response models ──────────────────────────────────
class PatientInput(BaseModel):
    age: int = Field(..., description="Patient age")
    symptoms: str = Field(..., description="Symptom description")
    history: str = Field(default="无", description="Medical history")


class RehabResponse(BaseModel):
    diagnosis: str
    risk_level: str
    plan: List[str]


# ── Endpoints ──────────────────────────────────────────────────
@app.post("/analyze", response_model=RehabResponse)
def analyze_patient(patient: PatientInput):
    """Run the full agent pipeline and return structured results."""
    if _dispatcher is None:
        raise HTTPException(503, "System is initializing, please retry.")

    result = _dispatcher.run(patient.model_dump())
    structured = json.loads(to_structured_json(result))
    return RehabResponse(
        diagnosis=structured["diagnosis"],
        risk_level=structured["risk_level"],
        plan=structured["plan"],
    )


@app.post("/analyze/full")
def analyze_full(patient: PatientInput):
    """Run the full pipeline and return the complete result (including raw agent outputs)."""
    if _dispatcher is None:
        raise HTTPException(503, "System is initializing, please retry.")
    return _dispatcher.run(patient.model_dump())


@app.post("/analyze/pdf")
def analyze_pdf(patient: PatientInput):
    """Run the pipeline and return a PDF report."""
    if _dispatcher is None:
        raise HTTPException(503, "System is initializing, please retry.")
    result = _dispatcher.run(patient.model_dump())
    path = to_pdf(result, "/tmp/rehab_report.pdf")
    return FileResponse(path, media_type="application/pdf", filename="rehab_report.pdf")


@app.get("/health")
def health():
    return {"status": "ok", "dispatcher_ready": _dispatcher is not None}


# ── Run directly ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    from config import config
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
