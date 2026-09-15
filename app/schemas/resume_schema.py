from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ResumeTailorRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    section: Literal[
        "summary",
        "skills",
        "experience",
        "projects",
    ]

    source_text: str = Field(
        min_length=10,
        max_length=12000,
    )

    job_description: str = Field(
        min_length=30,
        max_length=20000,
    )


class ResumeTailorResponse(BaseModel):
    section: str
    original: str
    suggestion: str
    truncated: bool = False
    review_required: bool = True