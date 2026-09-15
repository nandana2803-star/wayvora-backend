from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from app.utils.compat import install_pickle_compatibility

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models" / "careers"
MODEL_PATH = MODEL_DIR / "career_rank_models.pkl"
METADATA_PATH = MODEL_DIR / "career_metadata.json"
CAREER_DETAILS_PATH = MODEL_DIR / "career_details.json"

TARGET_COLUMNS_DEFAULT = ["top_career_1","top_career_2","top_career_3","top_career_4","top_career_5"]
HOBBY_COLUMNS = [
    "hobby_coding","hobby_gaming","hobby_reading","hobby_writing","hobby_music",
    "hobby_drawing_art","hobby_sports","hobby_cooking","hobby_photography",
    "hobby_travel","hobby_science_experiments","hobby_volunteering",
    "hobby_debating","hobby_robotics","hobby_fashion","hobby_business_trading",
]
GRADE_COLUMNS = ["grade_math","grade_science","grade_lang","grade_social","grade_cs"]
SCORE_COLUMNS = ["score_analytical","score_numeric","score_verbal","score_creative","score_social"]

HOBBY_LABELS = {
    "hobby_coding":"Coding","hobby_gaming":"Gaming","hobby_reading":"Reading",
    "hobby_writing":"Writing","hobby_music":"Music","hobby_drawing_art":"Drawing & Art",
    "hobby_sports":"Sports","hobby_cooking":"Cooking","hobby_photography":"Photography",
    "hobby_travel":"Travel","hobby_science_experiments":"Science Experiments",
    "hobby_volunteering":"Volunteering","hobby_debating":"Debating",
    "hobby_robotics":"Robotics","hobby_fashion":"Fashion","hobby_business_trading":"Business & Trading",
}
GRADE_LABELS = {
    "grade_math":"Mathematics","grade_science":"Science","grade_lang":"Language",
    "grade_social":"Social Science","grade_cs":"Computer Science",
}
SCORE_LABELS = {
    "score_analytical":"Analytical Thinking","score_numeric":"Numerical Ability",
    "score_verbal":"Verbal Ability","score_creative":"Creativity","score_social":"Social Skills",
}

CAREER_NAME_ALIASES = {

    # Technology
    "AI Engineer": "Machine Learning Engineer",
    "Cloud Engineer": "Cloud Solutions Architect",
    "Data Analyst": "Business Data Analyst",
    "Ethical Hacker": "Cybersecurity Analyst",
    "Mobile App Developer": "Mobile Application Developer",
    "Software Developer": "Software Engineer",
    "Software Tester / QA Engineer": "QA / Test Engineer",
    "Web Developer": "Full Stack Web Developer",

    # Education
    "Academic Counselor": "Academic Advisor",
    "College Lecturer": "College / University Professor",
    "Professor": "College / University Professor",
    "Online Course Instructor": "Online Tutor / EdTech Instructor",

    # Aviation / Marine
    "Commercial Pilot": "Commercial Airline Pilot",
    "Ship Captain": "Cruise Ship Captain",

    # Business / Management
    "Human Resource Manager": "Human Resources Manager",
    "Real Estate Manager": "Real Estate Property Manager",

    # Healthcare
    "Dietitian": "Clinical Nutritionist / Dietitian",
    "Medical Lab Technologist": "Medical Laboratory Technologist",
    "Nurse": "Registered Nurse",

    # Government / Law
    "Civil Services Officer": "IAS Officer (Civil Services)",
    "Legal Advisor": "Legal Advisor / In-House Counsel",

    # Policy
    "Policy Analyst": "Public Policy Analyst",

    # Creative
    "VFX Artist": "3D Modeler / VFX Artist",

    # Engineering
    "Electronics Engineer": "Electronics & Communication Engineer",

    # Other
    "Game Tester": "QA / Test Engineer",
}

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()

def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).lower()).strip()

class PredictionService:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        if not METADATA_PATH.exists():
            raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

        install_pickle_compatibility()
        self.model_bundle = joblib.load(MODEL_PATH)
        self.metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        self.career_details = self._load_details()
        self.details_lookup = self._build_details_lookup(self.career_details)
        self.feature_columns = self.metadata.get("feature_columns", [])
        self.target_columns = self.metadata.get("target_columns", TARGET_COLUMNS_DEFAULT)
        self.field_values = self.metadata.get("field_filter_values", [])
        if not self.feature_columns:
            raise ValueError("career_metadata.json contains no feature_columns.")

    def _load_details(self):
        if not CAREER_DETAILS_PATH.exists():
            return []
        data = json.loads(CAREER_DETAILS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            careers = data.get("careers")
            if isinstance(careers, list):
                return [x for x in careers if isinstance(x, dict)]
            return [x for x in data.values() if isinstance(x, dict)]
        return []

    def _build_details_lookup(self, details):
        lookup = {}
        for item in details:
            name = clean_text(item.get("Career_name", item.get("career_name", item.get("career", ""))))
            if name:
                lookup[normalize_name(name)] = item
        return lookup

    def _find_details(self, career):

        if not career:
            return None
        normalized = normalize_name(career)

        if normalized in self.details_lookup:
            return self.details_lookup[normalized]



        alias = CAREER_NAME_ALIASES.get(career)

        if alias:

            alias_normalized = normalize_name(alias)

            if alias_normalized in self.details_lookup:
                return self.details_lookup[alias_normalized]


        return None

    def build_input(self, field_filter, hobbies, grades, scores):
        row = {"field_filter": field_filter}
        selected = set(hobbies)
        for hobby in HOBBY_COLUMNS:
            row[hobby] = int(hobby in selected)
        for column in GRADE_COLUMNS:
            row[column] = float(grades[column])
        for column in SCORE_COLUMNS:
            row[column] = float(scores[column])
        input_data = pd.DataFrame([row])
        for column in self.feature_columns:
            if column not in input_data.columns:
                input_data[column] = np.nan
        return input_data[self.feature_columns]

    def predict(self, field_filter, hobbies, grades, scores):
        input_data = self.build_input(field_filter, hobbies, grades, scores)
        predictions = []
        used_careers = set()

        # This loop intentionally mirrors the Streamlit application's logic.
        for target_column in self.target_columns:
            rank_data = self.model_bundle.get(target_column)
            if not isinstance(rank_data, dict):
                continue

            availability_model = rank_data.get("availability_model")
            if availability_model is not None:
                try:
                    availability_prediction = availability_model.predict(input_data)[0]
                    if int(availability_prediction) != 1:
                        continue
                except Exception:
                    # Exact behavior of the existing Streamlit code.
                    pass
            elif rank_data.get("availability_constant") is not None:
                if int(rank_data["availability_constant"]) != 1:
                    continue

            career = ""
            career_model = rank_data.get("career_model")
            single_career = clean_text(rank_data.get("single_career"))

            if career_model is not None:
                try:
                    career = clean_text(career_model.predict(input_data)[0])
                except Exception:
                    career = ""
            elif single_career:
                career = single_career

            normalized = normalize_name(career)
            if not normalized or normalized in {"nan","none","null"}:
                continue
            if normalized in used_careers:
                continue

            used_careers.add(normalized)
            predictions.append({
                "rank": len(predictions) + 1,
                "career": career,
                "details": self._public_details(self._find_details(career)),
            })

        return predictions[:5]

    @staticmethod

    def _public_details(details):
        if not details:
            print("NO CAREER DETAILS FOUND")
            return None

        excluded_fields = {
        "id",
        "Career_name",
        "career_name",
        "career",
    }

        public_details = {}

        for key, value in details.items():

            if key in excluded_fields:
                continue

            public_details[key] = value


        return public_details

    def metadata_response(self):
        return {
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "field_filter_values": self.field_values,
            "hobbies": [{"value": x, "label": HOBBY_LABELS.get(x, x)} for x in HOBBY_COLUMNS],
            "grades": [{"value": x, "label": GRADE_LABELS.get(x, x)} for x in GRADE_COLUMNS],
            "scores": [{"value": x, "label": SCORE_LABELS.get(x, x)} for x in SCORE_COLUMNS],
        }
