# 🛡️ Intelligent Intrusion Detection System

A hybrid AI-powered Network Intrusion Detection System that detects, classifies, explains, and predicts cyberattacks in real time.

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Multiclass-green)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-teal)](https://fastapi.tiangolo.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-81.68%25-brightgreen)](#performance)

---

## What It Does

Most IDS tools only tell you *if* an attack happened. This system tells you:

- **What** type of attack — 6 attack categories
- **Why** it was flagged — SHAP feature explanations in plain English
- **How dangerous** — risk score from 0 to 10
- **What next** — predicted attacker's next move using Cyber Kill Chain

---

## Architecture

```
Network Traffic
      ↓
Preprocessing → 33 features (StandardScaler)
      ↓
┌─────────────────────────────────────────┐
│  Autoencoder   →  Reconstruction Error  │  Anomaly Detection
│  XGBoost       →  Attack Type + Conf.   │  Classification
└─────────────────────────────────────────┘
      ↓
Risk Engine  →  0–10 severity score
SHAP XAI     →  plain English explanation
Path Pred    →  next attack + Kill Chain
      ↓
FastAPI  →  JSON response  →  Dashboard + ARIA
```

---

## Attack Classes

| Class | Kill Chain Stage | Risk Weight |
|---|---|---|
| `reconnaissance` | Reconnaissance | 4/10 |
| `fuzzers` | Weaponization | 6/10 |
| `exploits` | Exploitation | 9/10 |
| `generic` | Command & Control | 7/10 |
| `dos` | Impact | 10/10 |
| `normal` | — | 0/10 |

---

## Performance

| Model | Accuracy |
|---|---|
| Binary XGBoost (Normal vs Attack) | ~95% |
| Multiclass XGBoost (6 classes) | **81.68%** |

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| generic | 1.00 | 0.98 | 0.99 |
| normal | 0.93 | 0.88 | 0.91 |
| reconnaissance | 0.85 | 0.81 | 0.83 |
| exploits | 0.85 | 0.54 | 0.66 |
| fuzzers | 0.59 | 0.67 | 0.63 |
| dos | 0.38 | 0.82 | 0.52 |

---

## Tech Stack

| Component | Technology |
|---|---|
| Anomaly Detection | TensorFlow / Keras — Autoencoder |
| Classification | XGBoost — Multiclass (200 estimators) |
| Explainability | SHAP TreeExplainer |
| Class Balancing | SMOTE — 69,750 samples per class |
| API | FastAPI + Uvicorn |
| AI Analyst | ARIA — OpenRouter (free) |
| Frontend | HTML + Chart.js |

---

## Installation

**Requirements:** Python 3.10+, Git

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/Intelligent-IDS-System.git
cd Intelligent-IDS-System

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Model Training

All models are trained using the provided notebook.

**Step 1 — Get the dataset**

Download both files from [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset):
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

**Step 2 — Run the notebook**

Open `notebooks/IDS_Training_Notebook.ipynb` in Google Colab,
upload both CSV files, and run all cells.
The notebook trains all models and downloads them automatically.

**Step 3 — Place model files**

After training, place the downloaded files in `src/models/`:

```
src/models/
├── encoder_model.h5
├── autoencoder_xg_model.h5
├── scaler.joblib
├── xgb_model.joblib
├── xgb_multiclass.joblib
└── le_label.joblib
```

---

## Configuration

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=sk-or-your-key-here
```

Get a free key at [openrouter.ai](https://openrouter.ai) — no credit card required.

---

## Running

```bash
uvicorn main:app --reload
```

| URL | Description |
|---|---|
| `http://127.0.0.1:8000/dashboard` | Main dashboard |
| `http://127.0.0.1:8000/docs` | Swagger API docs |
| `http://127.0.0.1:8000/health` | System status |

---

## API Reference

```
POST /predict       Single traffic record — full analysis
POST /analyze-csv   Batch CSV analysis (up to 200 rows)
POST /ai-chat       ARIA AI analyst Q&A
GET  /dashboard     HTML dashboard
GET  /health        System status
```

**Example response from `/predict`:**

```json
{
  "prediction": "dos",
  "confidence": 0.563,
  "risk_score": 6.66,
  "risk_level": "HIGH",
  "explanation": "DOS detected because: source is sending abnormally high traffic load",
  "top_features": ["sload", "rate", "smean"],
  "current_stage": "Impact",
  "next_attack": "EXPLOITS",
  "next_probability": 0.35,
  "recommended_actions": [
    "Enable rate limiting on all endpoints immediately",
    "Contact ISP for upstream traffic filtering"
  ]
}
```

---

## Dashboard Features

| Feature | Description |
|---|---|
| 🔴 Live Simulation | 6 attack type buttons for instant testing |
| 🎯 Kill Chain Sequence | Full 4-step attack sequence simulation |
| 📂 CSV Upload | Batch analysis — use files from `Data for testing/` |
| 📊 Charts | Attack distribution donut + risk score timeline |
| 🔍 Detail Panel | Slide-in panel with Kill Chain tracker and SHAP features |
| 🤖 ARIA | AI cybersecurity analyst chatbot |
| ⬇️ Export | Download all results as CSV |
| ☀️ Theme | Light / Dark mode toggle |

---

## Testing

Demo CSV files are provided in `Data for testing/` for immediate testing:

| File | Contents |
|---|---|
| `demo_dos.csv` | DoS + Normal traffic |
| `demo_exploits_recon.csv` | Exploits + Reconnaissance + Normal |
| `demo_fuzzers_generic.csv` | Fuzzers + Generic + Normal |
| `general_demo_traffic.csv` | Mixed traffic — all classes |

Upload any of these via the dashboard CSV upload button.

---

## Project Structure

```
Intelligent-IDS-System/
├── Data for testing/           ← Demo CSV files for testing
│   ├── demo_dos.csv
│   ├── demo_exploits_recon.csv
│   ├── demo_fuzzers_generic.csv
│   └── general_demo_traffic.csv
├── docs/                       ← Training plots and report
│   ├── 01_autoencoder_loss.png
│   ├── 02_binary_confusion_matrix.png
│   ├── 03_multiclass_confusion_matrix.png
│   ├── 04_reconstruction_error.png
│   ├── 05_smote_class_distribution.png
│   └── IDS_Training_Report.txt
├── notebooks/
│   └── IDS_Training_Notebook.ipynb   ← Train all models here
├── src/
│   ├── models/                 ← Place trained models here (not in repo)
│   ├── preprocessing/
│   │   └── cleaner.py
│   └── core/
│       ├── risk_engine.py
│       ├── xai_engine.py
│       └── path_predictor.py
├── templates/
│   └── index.html              ← Dashboard UI
├── main.py                     ← FastAPI server
├── requirements.txt
├── REFERENCE_MODEL_DETAILS.md  ← Complete technical documentation
└── README.md
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `jinja2 must be installed` | `pip install jinja2 python-multipart` |
| `ModuleNotFoundError: dotenv` | `pip install python-dotenv` |
| `FileNotFoundError: encoder_model.h5` | Run the training notebook first |
| `Port 8000 already in use` | `uvicorn main:app --reload --port 8001` |
| ARIA not responding | Check `OPENROUTER_API_KEY` in `.env` |

---

**Vishwakarma Institute of Technology, Pune — BTech CyberSecurity & IoT — 2026**
