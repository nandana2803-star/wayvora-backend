from pydantic import BaseModel, ConfigDict, Field

class Grades(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grade_math: float = Field(ge=0, le=100)
    grade_science: float = Field(ge=0, le=100)
    grade_lang: float = Field(ge=0, le=100)
    grade_social: float = Field(ge=0, le=100)
    grade_cs: float = Field(ge=0, le=100)

class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    score_analytical: float = Field(ge=0, le=100)
    score_numeric: float = Field(ge=0, le=100)
    score_verbal: float = Field(ge=0, le=100)
    score_creative: float = Field(ge=0, le=100)
    score_social: float = Field(ge=0, le=100)

class StudentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_filter: str = Field(min_length=1)
    hobbies: list[str] = Field(default_factory=list)
    grades: Grades
    scores: Scores

class Recommendation(BaseModel):
    rank: int
    career: str
    details: dict | None = None

class PredictionResponse(BaseModel):
    success: bool
    recommendations: list[Recommendation]
