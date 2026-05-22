import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# ── Constants ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# Exact 33 features after correlation filter (from notebook)
# Columns dropped by correlation filter:
# sbytes, dbytes, dloss, sloss, dwin, ct_src_dport_ltm, ct_dst_src_ltm, ct_ftp_cmd, ct_srv_dst
EXPECTED_FEATURES = [
    'dur', 'proto', 'service', 'state', 'spkts', 'dpkts',
    'rate', 'sttl', 'dttl', 'sload', 'dload',
    'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
    'ct_dst_sport_ltm', 'is_ftp_login',
    'ct_flw_http_mthd', 'ct_src_ltm', 'is_sm_ips_ports'
]

CATEGORICAL_COLS = ['proto', 'service', 'state']


# ── Load Scaler ─────────────────────────────────────────────
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")
    return joblib.load(SCALER_PATH)


# ── Main Cleaning Function ───────────────────────────────────
def clean_and_scale(df: pd.DataFrame) -> np.ndarray:
    df = df.copy()

    # Step 1 — Strip column name spaces
    df.columns = df.columns.str.strip()

    # Step 2 — Encode categorical columns using LabelEncoder
    # Must match notebook: fillna('missing') + LabelEncoder
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(
                df[col].fillna('missing').astype(str)
            )

    # Step 3 — Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Step 4 — Keep only expected features in correct order
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[EXPECTED_FEATURES]

    # Step 5 — Fill missing values with median (match notebook)
    for col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)

    # Step 6 — Scale using saved scaler
    scaler = load_scaler()
    scaled = scaler.transform(df)
    return scaled


# ── Single Row Preprocessing ─────────────────────────────────
def preprocess_single(input_dict: dict) -> np.ndarray:
    df = pd.DataFrame([input_dict])
    return clean_and_scale(df)


# ── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing cleaner.py with new 33-feature pipeline...\n")

    test_record = {
        'dur': 0.121478, 'proto': 'tcp', 'service': 'http',
        'state': 'FIN', 'spkts': 6, 'dpkts': 4,
        'rate': 74.08, 'sttl': 252, 'dttl': 62,
        'sload': 1950210.0, 'dload': 3163540.0,
        'sinpkt': 0.5, 'dinpkt': 0.3,
        'sjit': 10.5, 'djit': 8.2,
        'swin': 255, 'stcpb': 12345, 'dtcpb': 67890,
        'tcprtt': 0.001, 'synack': 0.0, 'ackdat': 0.0,
        'smean': 493, 'dmean': 800,
        'trans_depth': 1, 'response_body_len': 4800,
        'ct_srv_src': 150, 'ct_state_ttl': 4,
        'ct_dst_ltm': 100, 'ct_dst_sport_ltm': 1,
        'is_ftp_login': 0, 'ct_flw_http_mthd': 1,
        'ct_src_ltm': 1, 'is_sm_ips_ports': 0,
    }

    result = preprocess_single(test_record)
    print(f"Output shape : {result.shape}")
    print(f"Expected     : (1, 33)")
    print(f"First 5 vals : {result[0][:5].round(4)}")
    print(f"\n✅ cleaner.py working correctly!" if result.shape == (1, 33)
          else "\n❌ Shape mismatch — check EXPECTED_FEATURES list")