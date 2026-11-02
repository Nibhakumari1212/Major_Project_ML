import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI()

model = joblib.load("fake_job_post_detector_model.pkl")


class Job(BaseModel):
    title: str
    company_profile: str = ""
    description: str = ""
    requirements: str = ""

    employment_type: str = "Unknown"
    required_experience: str = "Unknown"
    required_education: str = "Unknown"

    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0


@app.get("/")
def home():
    return FileResponse("index.html")


@app.post("/predict")
def predict(job: Job):

    text = (
        f"{job.title} "
        f"{job.company_profile} "
        f"{job.description} "
        f"{job.requirements}"
    )

    row = pd.DataFrame([{
        "text": text,
        "employment_type": job.employment_type,
        "required_experience": job.required_experience,
        "required_education": job.required_education,
        "telecommuting": job.telecommuting,
        "has_company_logo": job.has_company_logo,
        "has_questions": job.has_questions
    }])

    prob = float(model.predict_proba(row)[0][1])

    return {
        "fraudulent": int(prob > 0.5),
        "probability": round(prob, 2)
    }
