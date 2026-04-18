from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import uvicorn

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load("model.pkl")
FEATURES = [
    'crime_year', 'crime_month', 'crime_dow', 'crime_hour',
    'is_weekend', 'time_of_day', 'crime_type', 'is_domestic',
    'location_description', 'fbi_code', 'beat', 'district',
    'ward', 'community_area', 'crime_type_arrest_rate',
    'crime_type_total', 'district_arrest_rate', 'district_total',
    'hour_arrest_rate', 'dow_arrest_rate', 'is_high_leakage_type',
    'district_crime_type'
]

HIGH_LEAKAGE_TYPES = [
    'NARCOTICS', 'PROSTITUTION', 'GAMBLING',
    'LIQUOR LAW VIOLATION', 'PUBLIC INDECENCY',
    'CONCEALED CARRY LICENSE VIOLATION'
]

# ── Schema ────────────────────────────────────────────────────────────────────
class CrimeInput(BaseModel):
    crime_year: int = Field(..., example=2023)
    crime_month: int = Field(..., ge=1, le=12, example=6)
    crime_dow: int = Field(..., ge=0, le=6, example=2)
    crime_hour: int = Field(..., ge=0, le=23, example=14)
    crime_type: str = Field(..., example="THEFT")
    is_domestic: bool = Field(..., example=False)
    location_description: str = Field(..., example="STREET")
    fbi_code: str = Field(..., example="06")
    beat: int = Field(..., example=1121)
    district: float = Field(..., example=11.0)
    ward: Optional[float] = Field(None, example=28.0)
    community_area: Optional[float] = Field(None, example=26.0)
    crime_type_arrest_rate: float = Field(..., example=0.108)
    crime_type_total: int = Field(..., example=1811997)
    district_arrest_rate: float = Field(..., example=0.409)
    district_total: float = Field(..., example=540041.0)
    hour_arrest_rate: float = Field(..., example=0.245)
    dow_arrest_rate: float = Field(..., example=0.251)

class PredictionOutput(BaseModel):
    arrest_probability: float
    arrest_predicted: bool
    risk_level: str
    is_hard_case: bool
    model_version: str = "1.0.0"

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Chicago Crime Arrest Prediction API",
    description="""
    Predicts the probability of arrest for a reported crime in Chicago.
    
    Built on 8.5M crime records (2001-2026) with LightGBM.
    - Overall Test AUC: 0.865
    - Hard Cases AUC: 0.831
    - Time-based split: trained pre-2020, tested 2020+
    """,
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "message": "Chicago Crime Arrest Prediction API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionOutput)
def predict(crime: CrimeInput):
    try:
        # Build feature dict
        is_weekend = 1 if crime.crime_dow in [0, 6] else 0
        if crime.crime_hour in range(6, 12):
            time_of_day = "morning"
        elif crime.crime_hour in range(12, 18):
            time_of_day = "afternoon"
        elif crime.crime_hour in range(18, 22):
            time_of_day = "evening"
        else:
            time_of_day = "night"

        is_high_leakage = 1 if crime.crime_type.upper() in HIGH_LEAKAGE_TYPES else 0
        district_crime_type = f"{crime.district}_{crime.crime_type.upper()}"

        input_dict = {
            'crime_year': crime.crime_year,
            'crime_month': crime.crime_month,
            'crime_dow': crime.crime_dow,
            'crime_hour': crime.crime_hour,
            'is_weekend': is_weekend,
            'time_of_day': time_of_day,
            'crime_type': crime.crime_type.upper(),
            'is_domestic': crime.is_domestic,
            'location_description': crime.location_description.upper(),
            'fbi_code': crime.fbi_code,
            'beat': crime.beat,
            'district': crime.district,
            'ward': crime.ward if crime.ward else 0.0,
            'community_area': crime.community_area if crime.community_area else 0.0,
            'crime_type_arrest_rate': crime.crime_type_arrest_rate,
            'crime_type_total': crime.crime_type_total,
            'district_arrest_rate': crime.district_arrest_rate,
            'district_total': crime.district_total,
            'hour_arrest_rate': crime.hour_arrest_rate,
            'dow_arrest_rate': crime.dow_arrest_rate,
            'is_high_leakage_type': is_high_leakage,
            'district_crime_type': district_crime_type
        }

        # Convert categoricals
        cat_cols = ['crime_type', 'time_of_day', 'location_description',
                    'fbi_code', 'district_crime_type']
        df = pd.DataFrame([input_dict])
        for col in cat_cols:
            df[col] = df[col].astype('category')

        prob = model.predict_proba(df[FEATURES])[0][1]

        # Risk level
        if prob < 0.2:
            risk = "LOW"
        elif prob < 0.5:
            risk = "MEDIUM"
        else:
            risk = "HIGH"

        return PredictionOutput(
            arrest_probability=round(float(prob), 4),
            arrest_predicted=bool(prob >= 0.5),
            risk_level=risk,
            is_hard_case=is_high_leakage == 0
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)