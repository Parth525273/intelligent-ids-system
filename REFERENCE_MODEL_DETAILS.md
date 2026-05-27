# Model Reference — Intelligent IDS System

**Vishwakarma Institute of Technology, Pune**
**BTech Second Year — CyberSecurity And IOT**
**Version:** 2.0.0 | **Date:** April 2026

---

## 1. Dataset

| Property | Value |
|---|---|
| Name | UNSW-NB15 (pre-labeled version) |
| Source | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| Files | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` |
| Total samples (after filtering) | 257,499 |
| Original features | 45 |
| Features after correlation filter | 33 |
| Train / Test split | 75% / 25% (stratified, random_state=42) |

### Class Distribution

| Class | Count | Kill Chain Stage | Status |
|---|---|---|---|
| normal | 93,000 | None | ✅ Used |
| generic | 58,871 | Command & Control | ✅ Used |
| exploits | 44,525 | Exploitation | ✅ Used |
| fuzzers | 24,246 | Weaponization | ✅ Used |
| dos | 16,353 | Impact | ✅ Used |
| reconnaissance | 13,987 | Reconnaissance | ✅ Used |
| analysis | 2,677 | — | ❌ Dropped (< 500) |
| backdoor | 2,329 | — | ❌ Dropped (< 500) |
| shellcode | 1,511 | — | ❌ Dropped (< 500) |
| worms | 174 | — | ❌ Dropped (< 500) |

### 33 Features Used

```
dur, proto, service, state, spkts, dpkts, rate, sttl, dttl,
sload, dload, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb,
tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len,
ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_dst_sport_ltm,
is_ftp_login, ct_flw_http_mthd, ct_src_ltm, is_sm_ips_ports
```

**Dropped by correlation filter (threshold > 0.95):**
`sbytes, dbytes, dloss, sloss, dwin, ct_src_dport_ltm, ct_dst_src_ltm, ct_ftp_cmd, ct_srv_dst`

---

## 2. Preprocessing

Steps applied in order — all statistical operations on training data only (no leakage):

1. Strip and normalize column names
2. Filter attack classes with >= 500 samples
3. LabelEncode target: `attack_cat` → integer (6 classes)
4. Drop identifier columns: `id, srcip, dstip, sport, dsport, attack_cat, label`
5. LabelEncode categoricals: `proto`, `service`, `state`
6. Replace `inf / -inf` with `NaN`, fill `NaN` with column median
7. Stratified train/test split (75/25, random_state=42)
8. Pearson correlation filter on `X_train` only (|corr| > 0.95) → removes 9 features
9. `StandardScaler` fit on `X_train`, transform both splits → saves `scaler.joblib`
10. SMOTE on training data only → balances all 6 classes to 69,750 samples each

---

## 3. Model Architecture

### 3.1 Autoencoder

Trained exclusively on normal traffic. High reconstruction error = anomaly.

| Layer | Units | Activation | Regularization |
|---|---|---|---|
| Input | 33 | — | — |
| Encoder Dense 1 | 64 | ReLU | L2(0.001) |
| Encoder Dense 2 | 32 | ReLU | L2(0.001) |
| Latent (bottleneck) | 16 | ReLU | L2(0.001) |
| Decoder Dense 1 | 32 | ReLU | L2(0.001) |
| Decoder Dense 2 | 64 | ReLU | L2(0.001) |
| Output | 33 | Linear | — |

**Training:**

| Setting | Value |
|---|---|
| Optimizer | Adam (lr = 1e-4) |
| Loss | Mean Squared Error |
| Max epochs | 100 |
| Early stopping | patience = 10, restore_best_weights = True |
| Batch size | 1024 |
| Training data | Normal traffic only (55,800 samples) |

**Anomaly threshold:** 95th percentile of reconstruction errors on normal training traffic (~0.088)

### 3.2 Binary XGBoost

Detects whether traffic is normal or attack (binary).

| Parameter | Value |
|---|---|
| Input | 16-dim encoder latent features |
| n_estimators | 100 |
| max_depth | 5 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | binary:logistic |
| SMOTE | Yes (train only) |

### 3.3 Multiclass XGBoost — Main Model

Classifies exact attack type across 6 categories.

| Parameter | Value |
|---|---|
| Input | 33-dim scaled features (not encoder output) |
| Classes | 6 |
| n_estimators | 200 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| objective | multi:softprob |
| eval_metric | mlogloss |
| SMOTE | Yes — 69,750 samples per class |

> **Design note:** XGBoost is trained on raw 33-dim scaled features, not the
> 16-dim encoder latent features. When trained on latent features, accuracy
> dropped to 69.82% because the latent space optimized for normal traffic
> reconstruction was insufficient to separate all 6 attack classes.
> Using raw scaled features restored accuracy to 81.68%.

---

## 4. Performance

### Binary XGBoost
- **Accuracy: ~95%+**

### Multiclass XGBoost

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| dos | 0.38 | 0.82 | 0.52 | 4,088 |
| exploits | 0.85 | 0.54 | 0.66 | 11,131 |
| fuzzers | 0.59 | 0.67 | 0.63 | 6,062 |
| generic | 1.00 | 0.98 | 0.99 | 14,718 |
| normal | 0.93 | 0.88 | 0.91 | 23,250 |
| reconnaissance | 0.85 | 0.81 | 0.83 | 3,497 |
| **Overall** | | | **81.68%** | 62,746 |

### Reconstruction Error (Autoencoder)

| Metric | Value |
|---|---|
| Anomaly threshold | 0.088 |
| Mean error — Normal | 0.030 |
| Mean error — Attack | 0.222 |

---

## 5. Intelligence Modules

### 5.1 Risk Scoring Engine

```
Risk = [(Confidence × 0.5) + (ReconError_normalized × 0.3) + (AttackWeight_normalized × 0.2)] × 10
```

| Attack | Severity Weight |
|---|---|
| dos | 10 |
| exploits | 9 |
| generic | 7 |
| fuzzers | 6 |
| reconnaissance | 4 |
| normal | 0 |

| Score | Level |
|---|---|
| 8 – 10 | CRITICAL |
| 6 – 8 | HIGH |
| 4 – 6 | MEDIUM |
| 2 – 4 | LOW |
| 0 – 2 | SAFE |

### 5.2 SHAP Explainability Engine

- Library: `shap.TreeExplainer` on multiclass XGBoost
- Input: 33-dim scaled feature vector
- Output shape: (1, 33, 6) — one SHAP value per feature per class
- Returns: top 3 features by SHAP magnitude for predicted class
- Output: human-readable explanation string

### 5.3 Markov Chain Attack Path Predictor

Transition matrix — P(next attack | current attack):

| Current → Next | dos | exploits | fuzzers | generic | reconnaissance |
|---|---|---|---|---|---|
| reconnaissance | 0.20 | **0.40** | 0.25 | 0.10 | 0.05 |
| fuzzers | 0.25 | **0.45** | 0.05 | 0.15 | 0.10 |
| exploits | 0.30 | 0.05 | 0.15 | **0.40** | 0.10 |
| generic | **0.45** | 0.25 | 0.15 | 0.05 | 0.10 |
| dos | 0.05 | **0.35** | 0.10 | 0.30 | 0.20 |

**Kill Chain mapping:**

| Attack | Kill Chain Stage |
|---|---|
| reconnaissance | Reconnaissance |
| fuzzers | Weaponization |
| exploits | Exploitation |
| generic | Command & Control |
| dos | Impact |
| normal | None |

Per-IP attack history tracked (last 10 events) for longitudinal threat analysis.

---

## 6. Saved Model Files

| File | Description |
|---|---|
| `scaler.joblib` | StandardScaler fitted on 33 training features |
| `encoder_model.h5` | Encoder only — 33 → 16 latent dims |
| `autoencoder_xg_model.h5` | Full autoencoder — 33 → 16 → 33 |
| `xgb_model.joblib` | Binary XGBoost (Normal vs Attack) |
| `xgb_multiclass.joblib` | Multiclass XGBoost — 6 classes (main model) |
| `le_label.joblib` | LabelEncoder — maps 6 class names to integers |

---

*Vishwakarma Institute of Technology, Pune | BTech CyberSecurity & IoT | 2026*
