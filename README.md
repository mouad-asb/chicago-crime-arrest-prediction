# Chicago Crime Arrest Prediction

As a crime series enthusiast, I once wondered whether it's possible to statistically predict if a crime would end in an arrest.
This project is my answer: an end-to-end MLOps pipeline that predicts the probability of arrest for a reported crime in Chicago, trained on **8.5 million crime records (2001–2026)**.

> **Stack:** PostgreSQL · dbt · LightGBM · FastAPI · Docker  
> **Test AUC:** 0.865 (overall) · 0.831 (difficult cases)  
> **Time-based split:** trained pre-2020, tested 2020+

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Statistical Analysis](#statistical-analysis)
- [Model](#model)
- [Results](#results)
- [Setup](#setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Further Work](#further-work)

---

## Overview

This project builds a binary classifier that predicts $P(\text{arrest} = 1 \mid \text{crime features})$, the probability that a reported crime results in an arrest.

The key design principle is that **every modeling decision is only based on statistical analysis**. Rather than fitting a model naively on the raw data, we first investigate four statistical properties of the dataset — temporal drift, class imbalance structure, spatial autocorrelation, and confounding, then use each finding to justify a concrete modeling choice.

The full pipeline runs in Docker with a single command:

```bash
docker-compose up
```

---

## Architecture

```
Raw Data (8.5M rows CSV)
        ↓
PostgreSQL (Docker)
        ↓
dbt transformation layer
    ├── stg_crimes     (cleaning, type casting, date parsing)
    └── feat_crimes    (feature engineering, historical aggregations)
        ↓
LightGBM (train.py)
        ↓
model.pkl
        ↓
FastAPI serving layer (api.py)
        ↓
Docker Compose (db + api containers)
```

**Tech stack:**

| Layer | Tool |
|---|---|
| Raw storage | PostgreSQL 15 |
| Transformation | dbt-postgres 1.7 |
| Modeling | LightGBM + scikit-learn |
| Explainability | SHAP |
| Serving | FastAPI + uvicorn |
| Containerization | Docker + docker-compose |

---

## Statistical Analysis

As mentioned in the Overview, before modeling four statistical properties of the data were investigated. Each finding directly informs a modeling decision.

### 1. Temporal Drift

**Hypothesis:** Arrest rates have changed significantly over time due to shifts in policing policy.

**Test:** Pearson correlation between year and arrest rate + visual inspection.

**Finding:** Arrest rate dropped from **31% in 2005 to 12% in 2023** — a 60% relative decline. The trend correlation is **−0.903** (p-value ≈ 0). Two structural breaks are visible: 2015–2016 (Ferguson effect, reduced proactive policing) and 2020 (COVID-19).

```math
r(\text{year}, \text{arrest rate}) = -0.903, \quad p \approx 0
```

**Modeling decision:** Time-based train/test split — train on pre-2020 data, test on 2020+. A random split would produce an artificially optimistic evaluation.

---

### 2. Class Imbalance Structure

**Hypothesis:** The class imbalance is driven by crime type.

**Test:** Kruskal-Wallis test across crime types + arrest rate breakdown.

**Finding:** Arrest rates range from **99.3% (narcotics)** to **5.7% (burglary)**. Six crime types have arrest rates above 95% — these are "consent crimes" where the arrest is the detection event. They represent **10.2% of the data** and are essentially pre-determined outcomes.

**Kruskal-Wallis:** H = large, p ≈ 0, arrest rates differ significantly across crime types.

**Modeling decision:** No random oversampling. Instead, `crime_type_arrest_rate` is included as a feature to let the model learn the structural imbalance. High-leakage crime types are flagged with `is_high_leakage_type` and the AUC is reported separately for more difficult cases.

---

### 3. Spatial Autocorrelation

**Hypothesis:** Arrest rates vary systematically by district, violating i.i.d assumptions.

**Test:** Chi-square test of independence between district and arrest.

**Finding:** Arrest rates range from **18.2% to 50%** across 25 districts (std = 0.077). Chi-square statistic = **150,668**, p ≈ 0, we conclude that district and arrest are strongly dependent.

**Modeling decision:** District is included as a feature. An interaction term `district x crime_type` is added to capture joint effects. District is flagged as a potential fairness concern because it may encode policing intensity rather than crime solvability.

---

### 4. Confounding

**Hypothesis:** The district effect on arrest rate is partially explained by crime type composition.

**Test:** Compare naive district arrest rates vs district arrest rates controlling for crime type.

**Finding:** District 11 has the highest naive arrest rate (40.9%) but does not appear in the top 5 for theft specifically. Its elevated overall rate is largely explained by heavy narcotics enforcement not by generally more effective policing. This is basically what confounding is.

**Modeling decision:** `crime_type_arrest_rate` per district is added as a separate feature to disentangle the crime type effect from the district effect.

---

## Model

**Algorithm:** LightGBM (gradient boosted trees)

Mainly chosen because it handles mixed feature types (categorical + numerical) natively, captures non-linear interactions between district, crime type, and time, and scales efficiently to 7M training rows.

**Features:**

| Feature | Source | Motivation |
|---|---|---|
| `crime_type` | raw | Primary driver of arrest rate |
| `crime_type_arrest_rate` | dbt aggregate | Historical base rate by crime type |
| `district_arrest_rate` | dbt aggregate | Historical base rate by district |
| `hour_arrest_rate` | dbt aggregate | Time-of-day policing patterns |
| `dow_arrest_rate` | dbt aggregate | Day-of-week policing patterns |
| `district_crime_type` | engineered | Interaction term (confounding finding) |
| `is_high_leakage_type` | engineered | Flag for near-deterministic crime types |
| `crime_year` | dbt | Captures temporal drift |
| `is_domestic` | raw | Strong predictor for certain crime types |
| `beat`, `district`, `ward` | raw | Spatial features |

**Hyperparameters:**

```python
LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
)
```

Hyperparameters were chosen to balance model complexity with generalization on a 7M-row training set. A low learning rate (0.05) with high n_estimators and early stopping allows convergence without overfitting. num_leaves=63 increases tree complexity relative to the LightGBM default (31), justified by the large training size. Row and column subsampling (0.8) combined with L1/L2 regularization further reduce variance, this is particularly important given the temporal distribution shift between train and test sets.

---

## Results

**Train/test split:** pre-2020 (train) / 2020+ (test)

| Split | Rows | Arrest Rate |
|---|---|---|
| Train | 7,054,949 | 27.5% |
| Test | 1,475,187 | 13.8% |

The difference in arrest rates between train and test confirms the temporal drift finding, the model is evaluated under a different policing regime.

**Evaluation:**

| Metric | Value |
|---|---|
| Train AUC | 0.910 |
| Test AUC (overall) | 0.865 |
| Test AUC (hard cases only) | 0.831 |
| Test Accuracy | 90% |
| Arrest Recall | 47% |
| No-Arrest Precision | 92% |

> **Note on diffcult cases AUC:** The overall AUC of 0.865 is somewhat inflated by crime types with near-deterministic arrest outcomes (narcotics, prostitution, gambling). Excluding these, the model achieves AUC **0.831** on more ambiguous cases, a more honest measure of what the model learned about policing patterns.

The 47% arrest recall reflects the temporal drift challenge: the model was trained on a world with 27.5% arrest rate and tested on a world with 13.8% arrest rate.

---

## Setup

### Prerequisites

- Docker Desktop
- Python 3.10+
- dbt-postgres 1.7

### 1. Clone the repo

```bash
git clone https://github.com/mouad-asb/chicago-crime-arrest-prediction.git
cd chicago-crime-arrest-prediction
```

### 2. Download the data

Download the Chicago Crime dataset from:
https://data.cityofchicago.org/api/views/ijzp-q8t2/rows.csv?accessType=DOWNLOAD

Place the CSV in the project root.

### 3. Start PostgreSQL

```bash
docker run --name chicago-crimes-db \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=admin \
  -e POSTGRES_DB=chicago_crimes \
  -p 5433:5432 -d postgres:15
```

### 4. Load raw data

```bash
pip install -r requirements.txt
python load_data.py
```

### 5. Run dbt transformations

```bash
cd chicago_crime_forecast
dbt run
dbt test
cd ..
```

### 6. Train the model

```bash
python train.py
```

### 7. Run the full stack with Docker

```bash
docker-compose up --build
```

API available at `http://localhost:8000/docs`

---

## API Reference

**POST** `/predict`

Predicts arrest probability for a reported crime.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "crime_year": 2023,
    "crime_month": 6,
    "crime_dow": 2,
    "crime_hour": 14,
    "crime_type": "THEFT",
    "is_domestic": false,
    "location_description": "STREET",
    "fbi_code": "06",
    "beat": 1121,
    "district": 11.0,
    "ward": 28.0,
    "community_area": 26.0,
    "crime_type_arrest_rate": 0.108,
    "crime_type_total": 1811997,
    "district_arrest_rate": 0.409,
    "district_total": 540041,
    "hour_arrest_rate": 0.245,
    "dow_arrest_rate": 0.251
  }'
```

**Response:**

```json
{
  "arrest_probability": 0.0193,
  "arrest_predicted": false,
  "risk_level": "LOW",
  "is_hard_case": true,
  "model_version": "1.0.0"
}
```

**GET** `/health` — returns model status

---

## Project Structure

```
chicago-crime-arrest-prediction/
├── chicago_crime_forecast/        # dbt project
│   ├── models/
│   │   ├── staging/
│   │   │   ├── stg_crimes.sql     # cleaning layer
│   │   │   └── schema.yml         # data quality tests
│   │   └── features/
│   │       ├── feat_crimes.sql    # feature engineering
│   │       └── schema.yml         # data quality tests
│   └── dbt_project.yml
├── api.py                         # FastAPI serving layer
├── train.py                       # model training pipeline
├── load_data.py                   # raw data ingestion
├── analysis.ipynb                 # statistical analysis notebook
├── model.pkl                      # trained model artifact
├── Dockerfile                     # API container
├── docker-compose.yml             # orchestrates db + api
└── requirements.txt
```

---

## Further Work

**Incremental dbt models** : the current `feat_crimes` model rebuilds all 8.5M rows on every `dbt run`. An incremental model would only process new crimes since the last run, making the pipeline production-ready for daily updates.

**Rolling historical features** : the current `crime_type_arrest_rate` is a static aggregate over all years. A rolling 6-month arrest rate would adapt to temporal drift dynamically, addressing the core challenge identified in the analysis.

**Fairness audit** : the model uses district as a feature, which may encode historical policing intensity rather than crime solvability. A formal fairness audit using demographic data at the community area level would be a natural extension.

**Migration to Databricks** : the current pipeline runs locally. Migrating the dbt layer to Databricks SQL and the training pipeline to MLflow would make this production-scale, handling the full historical dataset with incremental **daily** updates.

---

## References

- Chicago Data Portal: [City of Chicago Crime Data](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
- dbt Documentation: [docs.getdbt.com](https://docs.getdbt.com)
- LightGBM: Ke et al., 2017. *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*
- SHAP: Lundberg & Lee, 2017. *A Unified Approach to Interpreting Model Predictions*
