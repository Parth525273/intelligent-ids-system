# Intelligent Intrusion Detection System (IDS)

> Hybrid Deep Learning + XGBoost Network Security Intelligence  
> With Risk Scoring | Explainable AI | Attack Path Prediction | FastAPI

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Multiclass-green)
![FastAPI](https://img.shields.io/badge/FastAPI-REST-teal)

---

## Project Overview

This system detects and classifies network intrusions using a hybrid architecture:
- A pre-trained **Autoencoder** extracts latent feature representations
- A **Multiclass XGBoost Classifier** identifies attack types (DDoS, PortScan, BruteForce, Normal)
- A **Risk Scoring Engine** quantifies threat severity
- A **SHAP-based XAI Engine** explains every prediction
- An **Attack Path Predictor** maps threats to the Cyber Kill Chain
- A **FastAPI backend** exposes everything via REST API

Built on top of: [SecureNet-and-NIDS](https://github.com/Sumit4374/SecureNet-and-NIDS)

---

## Architecture
```
UNSW-NB15 Dataset
      ↓
Preprocessing (cleaner.py)
      ↓
Autoencoder → Latent Features
      ↓
XGBoost Multiclass Classifier
      ↓
┌─────────────────────────────────┐
│  Risk Engine | XAI | Path Pred  │
└─────────────────────────────────┘
      ↓
FastAPI → POST /predict
      ↓
React / Angular Dashboard
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras |
| Classification | XGBoost |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Data | Pandas, NumPy, Scikit-learn |
| Dataset | UNSW-NB15 |

---

## Project Structure
```
Intelligent-IDS-System/
├── src/
│   ├── data/               # Raw and processed datasets
│   ├── models/             # Saved .h5 and .joblib files
│   ├── preprocessing/      # cleaner.py
│   ├── core/               # risk_engine.py, xai_engine.py, path_predictor.py
│   └── api/                # FastAPI routes
├── notebooks/              # Experimentation only
├── main.py                 # FastAPI entry point
├── requirements.txt
└── README.md
```

---

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/intelligent-ids-system.git
cd intelligent-ids-system
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Dataset

Download [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place files in `src/data/`:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

---

## Running the API
```bash
uvicorn main:app --reload
# Open: http://127.0.0.1:8000/docs
```

### Example Request
```json
POST /predict
{
  "dur": 0.05,
  "sbytes": 5000,
  "dbytes": 1200,
  ...
}
```

### Example Response
```json
{
  "prediction": "ddos",
  "confidence": 0.94,
  "risk_score": 8.7,
  "top_features": ["dur", "sbytes", "dbytes"],
  "next_stage": "Service Disruption"
}
```

---

## Model Artifacts

| File | Description |
|---|---|
| `encoder_model.h5` | Pre-trained encoder for latent feature extraction |
| `autoencoder_xg_model.h5` | Full autoencoder for reconstruction error |
| `scaler.joblib` | Fitted StandardScaler |
| `xgb_multiclass.joblib` | New multiclass XGBoost (our addition) |

---

## Phases Completed

- [x] Phase 1 — Environment & repo setup
- [ ] Phase 2 — Data preprocessing pipeline
- [ ] Phase 3 — Hybrid multiclass model
- [ ] Phase 4 — Intelligence modules
- [ ] Phase 5 — FastAPI integration
- [ ] Phase 6 — Testing & deployment

---

## College Info

Vishwakarma Institute of Technology, Pune  
| BTech Second Year