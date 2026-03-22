import pandas as pd
import numpy as np
import joblib
import os

# ── Constants ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# Exact 29 features the scaler was trained on
EXPECTED_FEATURES = [
    'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl',
    'service', 'Sload', 'Dload', 'Spkts', 'swin', 'stcpb', 'dtcpb',
    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Stime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'is_sm_ips_ports',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_srv_src', 'ct_dst_ltm'
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

    # Step 2 — Encode categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna('missing').astype(str)
            df[col] = pd.factorize(df[col])[0]

    # Step 3 — Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Step 4 — Keep only expected features in correct order
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[EXPECTED_FEATURES]

    # Step 5 — Fill missing values with 0
    df = df.fillna(0)

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
    print("Testing cleaner.py...")

    test_record = {
        'proto': 'udp',
        'state': 'CON',
        'dur': 0.001055,
        'sbytes': 132,
        'dbytes': 164,
        'sttl': 31,
        'dttl': 29,
        'service': '-',
        'Sload': 0.0,
        'Dload': 0.0,
        'Spkts': 1,
        'swin': 0,
        'stcpb': 0,
        'dtcpb': 0,
        'smeansz': 132,
        'dmeansz': 164,
        'trans_depth': 0,
        'res_bdy_len': 0,
        'Sjit': 0.0,
        'Djit': 0.0,
        'Stime': 1000000,
        'Sintpkt': 0.0,
        'Dintpkt': 0.0,
        'tcprtt': 0.0,
        'is_sm_ips_ports': 0,
        'ct_flw_http_mthd': 0,
        'is_ftp_login': 0,
        'ct_srv_src': 3,
        'ct_dst_ltm': 1,
    }

    result = preprocess_single(test_record)
    print(f"Input features: {len(test_record)}")
    print(f"Output shape: {result.shape}")
    print(f"First 5 scaled values: {result[0][:5]}")
    print("✅ cleaner.py working correctly!")