from fastapi import APIRouter, HTTPException, Request

from app.schemas.course_schema import StudentProfile


router = APIRouter(
    prefix="/api/courses",
    tags=["CoursePick"],
)


@router.get("/metadata")
def metadata(request: Request):
    return request.app.state.course_service.metadata()


@router.post("/predict")
def predict(profile: StudentProfile, request: Request):
    service = request.app.state.course_service

    if profile.field_filter not in service.field_mapping:
        raise HTTPException(
            status_code=422,
            detail="Please choose a valid course field.",
        )

    return service.predict(profile.model_dump())


@router.get("/model-info")
def model_info(request: Request):
    return request.app.state.course_service.model_status()


@router.get("/catalog-status")
def catalog_status(request: Request):
    return request.app.state.course_service.catalog_status()


@router.get("/catalog-comparison")
def catalog_comparison(request: Request):
    return request.app.state.course_service.catalog_comparison()