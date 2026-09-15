from fastapi import APIRouter, HTTPException, Request
from app.schemas.career_schema import StudentProfile, PredictionResponse

router = APIRouter(prefix="/api/careers", tags=["CareerMate"])

@router.post("/predict", response_model=PredictionResponse)
def predict(profile: StudentProfile, request: Request):
    service = request.app.state.career_service
    if profile.field_filter not in service.field_values:
        raise HTTPException(status_code=422, detail="Invalid preferred career field.")

    valid_hobbies = {x["value"] for x in service.metadata_response()["hobbies"]}
    unknown_hobbies = sorted(set(profile.hobbies) - valid_hobbies)
    if unknown_hobbies:
        raise HTTPException(status_code=422, detail=f"Unknown hobbies: {', '.join(unknown_hobbies)}")

    return PredictionResponse(
        success=True,
        recommendations=service.predict(
            profile.field_filter,
            profile.hobbies,
            profile.grades.model_dump(),
            profile.scores.model_dump(),
        ),
    )

@router.get("/metadata")
def metadata(request: Request):
    return request.app.state.career_service.metadata_response()
