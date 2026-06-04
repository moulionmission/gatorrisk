"""
API Server — FastAPI REST Endpoint
===================================
Exposes the pipeline as a REST API for integration with
UF Health systems, REDCap, or other clinical research tools.

Endpoints:
    POST /extract          — Process a single note
    POST /extract/batch    — Process multiple notes
    GET  /health           — Health check
    GET  /docs             — Auto-generated Swagger UI (FastAPI built-in)

Usage:
    uvicorn api.server:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time

from modules.pipeline import Pipeline

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Lifestyle Risk Factor Extractor API",
    description=(
        "Clinical NLP pipeline that extracts structured lifestyle risk factors "
        "from unstructured clinical notes. "
        "Built at UF as an extension of the CTSI NLP Core's smoking extractor. "
        "RESEARCH USE ONLY — not a certified clinical decision support system."
    ),
    version="1.0.0",
    contact={
        "name": "UF CISE / CTSI NLP Research",
        "url": "https://www.ctsi.ufl.edu/research/laboratory-services/nlp-core/",
    },
)

# Initialize pipeline (singleton across requests)
_pipeline: Optional[Pipeline] = None

def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing NLP pipeline...")
        _pipeline = Pipeline(use_transformer=False, deidentify=True)
        logger.info("Pipeline ready.")
    return _pipeline


# ─────────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────────

class NoteRequest(BaseModel):
    note_id: str = Field(..., example="NOTE_001", description="Unique note identifier")
    text: str = Field(..., example="Patient smokes 1.5 ppd. BMI 34.2.", description="Raw clinical note text")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata (patient_id, note_type, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "note_id": "NOTE_001",
                "text": "Patient is a 58yo male. Smokes 1.5 packs per day for 30 years. Drinks 3 beers nightly. BMI 34.2, class I obese. Sedentary. Sleeps 4-5 hours, OSA suspected. Poor diet, high sodium. Denies drug use.",
                "metadata": {"patient_id": "PT_001", "note_type": "Progress Note"}
            }
        }


class BatchRequest(BaseModel):
    notes: List[NoteRequest] = Field(..., description="List of clinical notes to process")
    class Config:
        json_schema_extra = {
            "example": {
                "notes": [
                    {"note_id": "N001", "text": "Smokes 2 ppd. Obese BMI 38."},
                    {"note_id": "N002", "text": "Non-smoker. Exercises daily. BMI 22."},
                ]
            }
        }


class FactorRiskResponse(BaseModel):
    factor: str
    score: float
    tier: str
    rationale: str
    weight: float


class RiskProfileResponse(BaseModel):
    note_id: str
    composite_score: float
    composite_tier: str
    factors: List[FactorRiskResponse]
    disclaimer: str


class ExtractionResponse(BaseModel):
    note_id: str
    processing_time_ms: float
    sentences_extracted: int
    entities_found: Dict[str, int]
    normalized_profile: Dict[str, Any]
    risk_profile: RiskProfileResponse
    errors: List[str]
    status: str


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Check if the API and pipeline are ready."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None,
        "version": "1.0.0",
        "disclaimer": "RESEARCH USE ONLY",
    }


@app.post("/extract", response_model=ExtractionResponse, tags=["Extraction"])
def extract_single(request: NoteRequest):
    """
    Extract lifestyle risk factors from a single clinical note.

    Returns structured JSON with:
    - Normalized values per risk factor (smoking, alcohol, BMI, activity, sleep, diet, drugs)
    - Risk scores (0–100) and tiers (LOW/MODERATE/HIGH/CRITICAL) per factor
    - Composite risk score and tier
    """
    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="Note text is too short or empty.")
    if len(request.text) > 10000:
        raise HTTPException(status_code=400, detail="Note text exceeds maximum length of 10,000 characters.")

    pipeline = get_pipeline()

    try:
        output = pipeline.run_note(
            note_id=request.note_id,
            text=request.text,
            metadata=request.metadata,
        )
    except Exception as e:
        logger.error(f"Pipeline error for {request.note_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return ExtractionResponse(
        note_id=output.note_id,
        processing_time_ms=output.processing_time_ms,
        sentences_extracted=len(output.processed_note.sentences),
        entities_found=output.ner_result.summary(),
        normalized_profile=output.normalized_profile.to_dict(),
        risk_profile=RiskProfileResponse(
            note_id=output.risk_profile.note_id,
            composite_score=output.risk_profile.composite_score,
            composite_tier=output.risk_profile.composite_tier,
            factors=[FactorRiskResponse(**f.__dict__) for f in output.risk_profile.factors],
            disclaimer=output.risk_profile.disclaimer,
        ),
        errors=output.errors,
        status="success" if not output.errors else "partial",
    )


@app.post("/extract/batch", tags=["Extraction"])
def extract_batch(request: BatchRequest):
    """
    Extract lifestyle risk factors from multiple clinical notes.

    Processes notes sequentially. For large batches (>100 notes),
    consider running on HiPerGator with GPU acceleration.
    """
    if len(request.notes) == 0:
        raise HTTPException(status_code=400, detail="No notes provided.")
    if len(request.notes) > 100:
        raise HTTPException(status_code=400, detail="Max 100 notes per batch request.")

    pipeline = get_pipeline()
    notes_dicts = [
        {"note_id": n.note_id, "text": n.text, **(n.metadata or {})}
        for n in request.notes
    ]

    try:
        outputs = pipeline.run_batch(notes_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch pipeline error: {str(e)}")

    results = []
    for output in outputs:
        results.append({
            "note_id": output.note_id,
            "processing_time_ms": output.processing_time_ms,
            "composite_score": output.risk_profile.composite_score,
            "composite_tier": output.risk_profile.composite_tier,
            "normalized_profile": output.normalized_profile.to_dict(),
            "errors": output.errors,
        })

    return {
        "total_notes": len(results),
        "total_time_ms": sum(r["processing_time_ms"] for r in results),
        "results": results,
    }


@app.get("/factors", tags=["Info"])
def list_factors():
    """List all supported lifestyle risk factors and their descriptions."""
    return {
        "factors": [
            {"name": "smoking", "description": "Tobacco use — status, ppd, pack-years"},
            {"name": "alcohol", "description": "Alcohol use — status, drinks/day or /week, pattern"},
            {"name": "bmi", "description": "Body mass index — value, class, weight"},
            {"name": "physical_activity", "description": "Exercise — level, frequency, duration, type"},
            {"name": "sleep", "description": "Sleep — hours/night, OSA, insomnia, CPAP"},
            {"name": "diet", "description": "Diet quality — poor/moderate/good, flags"},
            {"name": "drug_use", "description": "Illicit substance use — status, substances, route"},
        ]
    }


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
