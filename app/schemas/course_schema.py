from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


Score = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]
Hobby = Literal[0, 1]


class StudentProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field_filter: str = Field(min_length=1)

    hobby_coding: Hobby = 0
    hobby_gaming: Hobby = 0
    hobby_reading: Hobby = 0
    hobby_writing: Hobby = 0
    hobby_music: Hobby = 0
    hobby_drawing_art: Hobby = 0
    hobby_sports: Hobby = 0
    hobby_cooking: Hobby = 0
    hobby_photography: Hobby = 0
    hobby_travel: Hobby = 0
    hobby_science_experiments: Hobby = 0
    hobby_volunteering: Hobby = 0
    hobby_debating: Hobby = 0
    hobby_robotics: Hobby = 0
    hobby_fashion: Hobby = 0
    hobby_business_trading: Hobby = 0

    grade_math: Score
    grade_science: Score
    grade_lang: Score
    grade_social: Score
    grade_cs: Score

    score_analytical: Score
    score_numeric: Score
    score_verbal: Score
    score_creative: Score
    score_social: Score