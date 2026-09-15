from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.careers import router as career_router
from app.routes.course import router as course_router
from app.routes.chat import router as chat_router
from app.routes.resume import router as resume_router
from app.routes.ats import router as ats_router
from app.routes.interview import router as interview_router
from app.routes.mock_interview import router as mock_interview_router

from app.services.career_service import PredictionService as CareerService
from app.services.course_service import PredictionService as CourseService


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.career_service = CareerService()
    app.state.course_service = CourseService()
    yield


app = FastAPI(
    title="WAYVORA API",
    description="Explore your way toward the future.",
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    value.strip()
    for value in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if value.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(course_router)
app.include_router(career_router)
app.include_router(chat_router)
app.include_router(resume_router)
app.include_router(ats_router)
app.include_router(interview_router)
app.include_router(mock_interview_router)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "WAYVORA",
        "sections": ["CoursePick", "CareerMate"],
    }


# Serve the frontend build when it is available.
dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if dist.is_dir():
    app.mount(
        "/",
        StaticFiles(directory=dist, html=True),
        name="website",
    )