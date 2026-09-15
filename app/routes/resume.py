from fastapi import APIRouter, HTTPException

from app.schemas.resume_schema import (
    ResumeTailorRequest,
    ResumeTailorResponse,
)
from app.services.resume_service import (
    ResumeServiceError,
    tailor_section,
)


router = APIRouter(
    prefix="/api/resume",
    tags=["WAYVORA Resume"],
)


@router.post("/tailor", response_model=ResumeTailorResponse)
def tailor_resume(payload: ResumeTailorRequest):
    try:
        return tailor_section(
            section=payload.section,
            source_text=payload.source_text,
            job_description=payload.job_description,
        )
    except ResumeServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.message,
        ) from exc