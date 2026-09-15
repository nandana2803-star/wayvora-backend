from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import json
import re


class PredictionService:
    def __init__(self):
        # Locate the project root independently of the terminal location.
        self.project_root = Path(__file__).resolve().parents[2]
        self.model_dir = self.project_root / "models" / "courses"
        details_path = (
            self.project_root / "data" / "course_details.json"
        )

        with details_path.open(encoding="utf-8-sig") as file:
            course_details = json.load(file)
            self.course_details = course_details

        self.details_by_name = {}

        for details in course_details:
            key = self.normalize_course_name(details["degree_name"])
            self.details_by_name.setdefault(key, []).append(details)


        self.additional_details = {}

        additional_path = (
            self.project_root
            / "data"
            / "missing_course_details.json"
        )

        if additional_path.is_file():
            with additional_path.open(encoding="utf-8-sig") as file:
                additional_records = json.load(file)

            for record in additional_records:
                required_fields = (
                    "model_course_name",
                    "degree_name",
                    "program_level",
                    "field",
                    "eligible_stream_or_degree",
                    "source_url",
                )

                complete = all(
                    isinstance(record.get(key), str)
                    and record[key].strip()
                    for key in required_fields
                )

                if record.get("verified") is True and complete:
                    name = record["model_course_name"].strip()

                    if name in self.additional_details:
                        raise ValueError(
                            f"Duplicate additional course record: {name}"
                        )

                    self.additional_details[name] = record
        self.models = {}
        self.label_encoders = {}

        self.feature_columns = list(
            self.load_artifact("feature_columns.joblib")
        )

        self.field_mapping = self.load_artifact(
            "course_field_mapping.joblib"
        )

        for rank in range(1, 6):
            target = f"top_course_{rank}"

            self.models[target] = self.load_artifact(
                f"{target}_model.joblib"
            )

            self.label_encoders[target] = self.load_artifact(
                f"{target}_label_encoder.joblib"
            )

        print(
            f"CoursePick loaded {len(self.models)} models, "
            f"{len(self.label_encoders)} label encoders, and "
            f"{len(self.feature_columns)} input features."
        )

    def load_artifact(self, filename):
        path = self.model_dir / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Required CoursePick model file is missing: {path}"
            )

        return joblib.load(path)

    def model_status(self):
        return {
            "models_loaded": len(self.models),
            "label_encoders_loaded": len(self.label_encoders),
            "feature_count": len(self.feature_columns),
            "feature_columns": self.feature_columns,
        }
    def metadata(self):
        return {
            "fields": sorted(self.field_mapping.keys()),
            "hobbies": [
                {
                    "value": column,
                    "label": column.removeprefix("hobby_")
                    .replace("_", " ")
                    .title(),
                }
                for column in self.feature_columns
                if column.startswith("hobby_")
            ],
        }

    def predict(self, profile):
        candidate = pd.DataFrame(
            [profile],
            columns=self.feature_columns,
        )

        recommendations = []

        for rank in range(1, 6):
            target = f"top_course_{rank}"
            model = self.models[target]
            encoder = self.label_encoders[target]

            # Handles both flat and column-shaped prediction arrays.
            prediction = np.asarray(
                model.predict(candidate)
            ).reshape(-1)

            encoded_label = int(prediction[0])
            course = encoder.inverse_transform([encoded_label])[0]

            details = self.get_course_details(str(course))

            recommendations.append({
                "rank": rank,
                "course": str(course),
                "details": details,
                "details_available": details is not None,
            })

        return {
            "success": True,
            "recommendations": recommendations,
        }
    @staticmethod
    def normalize_course_name(name):
        name = name.casefold().strip()
        name = name.replace(".", "")

        degree_names = {
            "bsc": "bachelor of science",
            "msc": "master of science",
            "btech": "bachelor of technology",
            "mtech": "master of technology",
            "ba": "bachelor of arts",
            "ma": "master of arts",
            "bcom": "bachelor of commerce",
            "mcom": "master of commerce",
        }

        for abbreviation, full_name in degree_names.items():
            name = re.sub(
                rf"^{abbreviation}\b",
                full_name,
                name,
            )

        name = re.sub(r"\bin\b", " ", name)
        return " ".join(name.split())

    def get_course_details(self, course):
        if course in self.additional_details:
            return self.additional_details[course]
        aliases = {
            "BArch Architecture":
                "Bachelor of Architecture (B.Arch)",

            "BBA Marketing":
                "Bachelor of Business Administration in Marketing",

            "LLM Corporate Law":
                "Master of Laws (LLM) in Corporate Law",

            "LLM Law":
                "Master of Laws (LLM)",

            "MBA International Business":
                "Master of Business Administration (MBA) — International Business",

            "PhD Education":
                "Doctor of Philosophy (Ph.D) in Education",
        }

        # Use an explicit catalog name when we have established a match.
        if course in aliases:
            catalog_name = aliases[course]

            matches = [
                item
                for item in self.course_details
                if item.get("degree_name") == catalog_name
            ]

            return matches[0] if len(matches) == 1 else None

        # Preserve the existing lookup for other course names.
        key = self.normalize_course_name(course)
        matches = self.details_by_name.get(key, [])

        return matches[0] if len(matches) == 1 else None

    def catalog_status(self):
        predicted_courses = sorted({
            str(course)
            for encoder in self.label_encoders.values()
            for course in encoder.classes_
        })

        missing_courses = [
            course
            for course in predicted_courses
            if self.get_course_details(course) is None
        ]

        return {
            "total_predictable_courses": len(predicted_courses),
            "courses_with_details": (
                len(predicted_courses) - len(missing_courses)
            ),
            "courses_missing_details": len(missing_courses),
            "missing_courses": missing_courses,
        }

    def catalog_comparison(self):
        predicted_courses = sorted({
            str(course)
            for encoder in self.label_encoders.values()
            for course in encoder.classes_
        })

        prefixes = (
            "Graduate Certificate in ",
            "Graduate Diploma in ",
            "BArch ",
            "BBA ",
            "BEng ",
            "BFA ",
            "BSc ",
            "LLB ",
            "LLM ",
            "MBA ",
            "MSc ",
            "PhD ",
        )

        def subject_key(text):
            return " ".join(
                str(text).casefold().replace("&", "and").split()
            )

        comparison = []

        for course in predicted_courses:
            subject = course

            for prefix in prefixes:
                if course.startswith(prefix):
                    subject = course[len(prefix):]
                    break

            candidates = [
                {
                    "degree_name": item.get("degree_name"),
                    "program_level": item.get("program_level"),
                    "major": item.get("major"),
                }
                for item in self.course_details
                if subject_key(item.get("major", ""))
                == subject_key(subject)
            ]

            comparison.append({
                "predicted_course": course,
                "same_subject_candidates": candidates,
            })

        return comparison