# MODEL DETAILS: Intelligent IDS — Hybrid Autoencoder + XGBoost Pipeline

**Vishwakarma Institute of Technology, Pune**
**BTech Second Year — *CyberSecurity And IOT*
**Version:** 2.0.0 | **Date:** April 2026

---

## 1. Overview

This document describes the complete technical architecture of the Intelligent Intrusion Detection System (IDS). The system is a hybrid pipeline combining:

1. A **Feedforward Autoencoder** trained exclusively on normal traffic to learn reconstruction behavior and detect anomalies via reconstruction error.
2. A **Multiclass XGBoost Classifier** trained on 33 scaled features to classify 6 attack types.
3. A **Risk Scoring Engine** that combines model confidence, reconstruction error, and attack severity.
4. A **SHAP-based XAI Engine** that explains every prediction in human-readable form.
5. A **Markov Chain Attack Path Predictor** mapped to the Cyber Kill Chain framework.
6. A **FastAPI REST Backend** exposing all intelligence via a `/predict` endpoint.
7. A **Cybersecurity Dashboard** (HTML/JS) for live simulation and CSV upload analysis.

### 1.1 End-to-End Pipeline Flow

```
UNSW_NB15_training-set.csv (pre-labeled, with headers)
        ↓
Preprocessing (cleaner.py)
  - Drop identifiers (srcip, dstip, sport, dsport)
  - LabelEncode: proto, service, state
  - Replace inf/NaN with median
  - Correlation filter (threshold > 0.95) → 33 features
  - StandardScaler (fit on train only)
        ↓
┌──────────────────────────────────────────────┐
│  Autoencoder (trained on normal only)        │
│  → Reconstruction error = anomaly signal     │
│                                              │
│  XGBoost Multiclass (trained on 33 features) │
│  → 6-class attack prediction                 │
└──────────────────────────────────────────────┘
        ↓
Risk Engine + SHAP XAI + Markov Chain Path Predictor
        ↓
FastAPI POST /predict → JSON response
        ↓
HTML Dashboard (simulation + CSV upload)
```

---

## 2. Dataset

| Property | Value |
|---|---|
| Dataset | UNSW-NB15 (pre-labeled version) |
| Source | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| Files used | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| Total rows (after worms filter) | ~257,499 |
| Original features | 45 (with headers) |
| Features after correlation filter | 33 |

### 2.1 Attack Categories

| Class | Count | Kill Chain Stage |
|---|---|---|
| normal | 93,000 | None |
| generic | 58,871 | Command & Control |
| exploits | 44,525 | Exploitation |
| fuzzers | 24,246 | Weaponization |
| dos | 16,353 | Impact |
| reconnaissance | 13,987 | Reconnaissance |
| analysis | 2,677 | Dropped (<500 threshold) |
| backdoor | 2,329 | Dropped (<500 threshold) |
| shellcode | 1,511 | Dropped (<500 threshold) |
| worms | 174 | Dropped (<500 threshold) |

**Final training classes (6):** `dos`, `exploits`, `fuzzers`, `generic`, `normal`, `reconnaissance`

### 2.2 33 Features Used (after correlation filter)

```
dur, proto, service, state, spkts, dpkts, rate, sttl, dttl,
sload, dload, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb,
tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len,
ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_dst_sport_ltm,
is_ftp_login, ct_flw_http_mthd, ct_src_ltm, is_sm_ips_ports
```

**Dropped by correlation filter (>0.95):**
`sbytes, dbytes, dloss, sloss, dwin, ct_src_dport_ltm, ct_dst_src_ltm, ct_ftp_cmd, ct_srv_dst`

---

## 3. Preprocessing Pipeline

### 3.1 Steps (in order)

```python
# 1. Load pre-labeled CSV (headers present)
train_df = pd.read_csv('UNSW_NB15_training-set.csv')

# 2. Normalize attack_cat to lowercase
full_df['attack_cat'] = full_df['attack_cat'].str.strip().str.lower()

# 3. Filter classes with >= 500 samples (drops: worms, analysis, backdoor, shellcode)
valid_cats = cat_counts[cat_counts >= 500].index.tolist()

# 4. Encode labels with LabelEncoder → saves le_label.joblib (6 classes)

# 5. Drop identifier columns
DROP_COLS = ['id', 'attack_cat', 'label', 'srcip', 'dstip', 'sport', 'dsport']

# 6. Encode categorical columns
for col in ['proto', 'service', 'state']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].fillna('missing').astype(str))

# 7. Replace inf/-inf with NaN, fill NaN with median

# 8. Train/test split (75/25, stratified, random_state=42)

# 9. Correlation filter on X_train only (threshold > 0.95)
#    → drops 9 highly correlated features

# 10. StandardScaler fit on X_train, transform both splits
#     → saves scaler.joblib
```

### 3.2 Inference Preprocessing (cleaner.py)

At inference time, `preprocess_single(input_dict)` applies the same pipeline:
- Same 33 features in the same order
- Same LabelEncoder logic for proto/service/state
- Same scaler transform (loaded from scaler.joblib)

---

## 4. Model Architecture

### 4.1 Autoencoder

**Purpose:** Learn normal traffic patterns. High reconstruction error = anomaly.

| Layer | Units | Activation | Regularization |
|---|---|---|---|
| Input | 33 | — | — |
| Encoder Dense 1 | 64 | ReLU | L2(0.001) |
| Encoder Dense 2 | 32 | ReLU | L2(0.001) |
| Latent (bottleneck) | 16 | ReLU | L2(0.001) |
| Decoder Dense 1 | 32 | ReLU | L2(0.001) |
| Decoder Dense 2 | 64 | ReLU | L2(0.001) |
| Output | 33 | Linear | — |

**Training settings:**
- Optimizer: Adam (lr=1e-4)
- Loss: MSE
- Epochs: 100 (with EarlyStopping, patience=10)
- Batch size: 1024
- Trained on: Normal traffic only

**Anomaly threshold:** 95th percentile of reconstruction error on normal traffic

### 4.2 Binary XGBoost (Normal vs Attack)

**Purpose:** Fast binary classification using encoder latent features.

| Parameter | Value |
|---|---|
| Input | 16-dim encoder output |
| n_estimators | 100 |
| max_depth | 5 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | binary:logistic |
| SMOTE | Yes (train only) |

### 4.3 Multiclass XGBoost (6 Attack Types) — MAIN MODEL

**Purpose:** Classify exact attack type from 33 scaled features.

| Parameter | Value |
|---|---|
| Input | 33-dim scaled features (NOT encoder output) |
| Classes | 6 (dos, exploits, fuzzers, generic, normal, reconnaissance) |
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | multi:softprob |
| SMOTE | Yes — all 6 classes balanced to 69,750 samples each |

> **Note:** The multiclass XGBoost uses raw scaled 33-dim features directly,
> NOT the encoder latent features. The encoder is used separately for
> reconstruction error (anomaly scoring). This design choice was made because
> the 16-dim latent space was insufficient to separate all 6 classes well.

---

## 5. Model Performance

### 5.1 Binary XGBoost
- Accuracy: ~95%+

### 5.2 Multiclass XGBoost

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| dos | 0.38 | 0.82 | 0.52 | 4,088 |
| exploits | 0.85 | 0.54 | 0.66 | 11,131 |
| fuzzers | 0.59 | 0.67 | 0.63 | 6,062 |
| generic | 1.00 | 0.98 | 0.99 | 14,718 |
| normal | 0.93 | 0.88 | 0.91 | 23,250 |
| reconnaissance | 0.85 | 0.81 | 0.83 | 3,497 |
| **Overall Accuracy** | | | **81.68%** | 62,746 |

### 5.3 Reconstruction Error (Autoencoder)
- Anomaly threshold: ~0.088 (95th percentile of normal traffic errors)
- Mean error (Normal): ~0.030
- Mean error (Attack): ~0.222

---

## 6. Intelligence Modules

### 6.1 Risk Scoring Engine (risk_engine.py)

```
Risk = (Confidence × 0.5) + (ReconError_normalized × 0.3) + (AttackWeight_normalized × 0.2)
Final Score = Risk × 10  (scale 0-10)
```

| Attack | Weight |
|---|---|
| dos | 10 |
| exploits | 9 |
| generic | 7 |
| fuzzers | 6 |
| reconnaissance | 4 |
| normal | 0 |

| Score Range | Level |
|---|---|
| 8-10 | CRITICAL |
| 6-8 | HIGH |
| 4-6 | MEDIUM |
| 2-4 | LOW |
| 0-2 | SAFE |

### 6.2 SHAP Explainability Engine (xai_engine.py)

- Uses `shap.TreeExplainer` on the multiclass XGBoost model
- Input: 33-dim scaled feature vector
- SHAP output shape: (1, 33, 6)
- Returns top 3 features contributing to the prediction
- Generates human-readable explanation string per attack class

### 6.3 Attack Path Predictor (path_predictor.py)

**Markov Chain Transition Matrix** — maps current attack → most likely next attack:

| From \ To | dos | exploits | fuzzers | generic | reconnaissance |
|---|---|---|---|---|---|
| reconnaissance | 0.20 | **0.40** | 0.25 | 0.10 | 0.05 |
| fuzzers | 0.25 | **0.45** | 0.05 | 0.15 | 0.10 |
| exploits | 0.30 | 0.05 | 0.15 | **0.40** | 0.10 |
| generic | **0.45** | 0.25 | 0.15 | 0.05 | 0.10 |
| dos | 0.05 | **0.35** | 0.10 | 0.30 | 0.20 |

**Cyber Kill Chain Mapping:**

```
Reconnaissance → Weaponization (Fuzzers) → Exploitation (Exploits)
              → Command & Control (Generic) → Impact (DoS)
```

---

## 7. Saved Model Files

| File | Description | Size (approx) |
|---|---|---|
| `scaler.joblib` | StandardScaler fitted on 33 features | ~5 KB |
| `autoencoder_xg_model.h5` | Full autoencoder (33→16→33) | ~200 KB |
| `encoder_model.h5` | Encoder only (33→16 latent) | ~120 KB |
| `xgb_model.joblib` | Binary XGBoost (Normal vs Attack) | ~1 MB |
| `xgb_multiclass.joblib` | Multiclass XGBoost (6 classes) | ~5 MB |
| `le_label.joblib` | LabelEncoder (6 class names) | ~1 KB |

---

## 8. API Reference

### POST /predict

**Request body (33 features):**
```json
{
  "dur": 0.121478,
  "proto": "tcp",
  "service": "http",
  "state": "FIN",
  "spkts": 6,
  "dpkts": 4,
  "rate": 74.08,
  "sttl": 252,
  "source_ip": "192.168.1.100"
}
```

**Response:**
```json
{
  "prediction": "exploits",
  "confidence": 0.865,
  "risk_score": 8.54,
  "risk_level": "CRITICAL",
  "explanation": "EXPLOITS detected because: abnormal connection state...",
  "top_features": ["state", "trans_depth", "response_body_len"],
  "reconstruction_error": 0.421,
  "current_stage": "Exploitation",
  "next_attack": "GENERIC",
  "next_probability": 0.40,
  "recommended_actions": ["Isolate affected systems immediately", "..."],
  "attack_history_count": 3,
  "source_ip": "192.168.1.100"
}
```

### GET /health
Returns system status and loaded model info.

### POST /analyze-csv
Accepts UNSW-NB15 format CSV, analyzes up to 200 rows, returns batch results.

### GET /dashboard
Serves the HTML cybersecurity dashboard.

---

## 9. Project Structure

```
Intelligent-IDS-System/
├── src/
│   ├── models/
│   │   ├── encoder_model.h5
│   │   ├── autoencoder_xg_model.h5
│   │   ├── scaler.joblib
│   │   ├── xgb_model.joblib
│   │   ├── xgb_multiclass.joblib
│   │   └── le_label.joblib
│   ├── preprocessing/
│   │   └── cleaner.py
│   └── core/
│       ├── risk_engine.py
│       ├── xai_engine.py
│       └── path_predictor.py
├── docs/
│   ├── IDS_Training_Report.txt
│   ├── 01_autoencoder_loss.png
│   ├── 02_binary_confusion_matrix.png
│   ├── 03_multiclass_confusion_matrix.png
│   ├── 04_reconstruction_error.png
│   ├── 05_smote_class_distribution.png
│   └── Intelligent_IDS_Project_Plan.docx
├── notebooks/
│   └── IDS_Training_Notebook.ipynb
├── templates/
│   └── index.html
├── main.py
├── requirements.txt
├── README.md
└── REFERENCE_MODEL_DETAILS.md
```

---

## 10. Phases Completed

| Phase | Title | Status |
|---|---|---|
| 1 | Environment & Repository Setup | ✅ Complete |
| 2 | Data Preprocessing (33 features, correct headers) | ✅ Complete |
| 3 | Hybrid Model (81.68% accuracy, 6 classes) | ✅ Complete |
| 4 | Intelligence Modules (Risk + SHAP + Markov Chain) | ✅ Complete |
| 5 | FastAPI + HTML Dashboard | ✅ Complete |
| 6 | Testing & Documentation | ⏳ In Progress |

---

## 11. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the API server
uvicorn main:app --reload

# 3. Open dashboard
http://127.0.0.1:8000/dashboard

# 4. Swagger API docs
http://127.0.0.1:8000/docs
```