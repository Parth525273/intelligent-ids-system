# src/core/xai_engine.py
# Explainable AI Engine using real SHAP TreeExplainer

import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# Class names in order matching le_label encoding
CLASS_NAMES = ['bruteforce', 'ddos', 'normal', 'portscan']

# Each latent feature maps to original feature groups
LATENT_TO_ORIGINAL = {
    0: 'dur_sbytes_dbytes',
    1: 'proto_state_service',
    2: 'Sload_Dload_Spkts',
    3: 'sttl_dttl_tcprtt',
    4: 'ct_srv_src_ct_dst_ltm'
}

# Human readable explanations for each feature group
FEATURE_EXPLANATIONS = {
    'Sload_Dload_Spkts':      'abnormally high network load and packet count',
    'dur_sbytes_dbytes':      'unusual traffic duration and data transfer size',
    'proto_state_service':    'suspicious protocol type and connection state',
    'sttl_dttl_tcprtt':       'abnormal TTL values and response timing',
    'ct_srv_src_ct_dst_ltm':  'too many connections to same service from same source'
}

# Attack specific human readable messages
ATTACK_MESSAGES = {
    'ddos': {
        'Sload_Dload_Spkts':     'network is being flooded with abnormal traffic volume',
        'ct_srv_src_ct_dst_ltm': 'too many connections targeting same service',
        'proto_state_service':   'unusual protocol pattern detected in flood traffic',
        'dur_sbytes_dbytes':     'massive data transfer volume indicating flood attack',
        'sttl_dttl_tcprtt':      'abnormal packet timing consistent with DDoS flood'
    },
    'bruteforce': {
        'Sload_Dload_Spkts':     'repeated high-frequency connection attempts detected',
        'dur_sbytes_dbytes':     'many small identical packets suggesting credential stuffing',
        'proto_state_service':   'repeated failed authentication on specific service',
        'ct_srv_src_ct_dst_ltm': 'same source hammering same destination repeatedly',
        'sttl_dttl_tcprtt':      'abnormal response timing from repeated login attempts'
    },
    'portscan': {
        'Sload_Dload_Spkts':     'rapid low-volume probes across multiple ports',
        'dur_sbytes_dbytes':     'very short connections with minimal data — port probing',
        'proto_state_service':   'systematic scanning of multiple services detected',
        'sttl_dttl_tcprtt':      'abnormal TTL pattern consistent with port scanning',
        'ct_srv_src_ct_dst_ltm': 'same source probing many different services'
    },
    'normal': {
        'Sload_Dload_Spkts':     'normal traffic load within expected range',
        'dur_sbytes_dbytes':     'typical connection duration and data size',
        'proto_state_service':   'standard protocol usage detected',
        'sttl_dttl_tcprtt':      'normal TTL and timing values',
        'ct_srv_src_ct_dst_ltm': 'normal connection patterns observed'
    }
}


def load_model():
    model_path = os.path.join(MODEL_DIR, 'xgb_multiclass.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def get_top_features_shap(
    encoded_input: np.ndarray,
    predicted_class_idx: int = None,
    top_n: int = 3
) -> list:
    """
    Uses real SHAP TreeExplainer to get top features
    for THIS SPECIFIC prediction and class.

    Args:
        encoded_input: encoder output array shape (1, 5)
        predicted_class_idx: index of predicted class (0-3)
        top_n: number of top features to return

    Returns:
        list of dicts with feature name and importance score
    """
    import shap

    model = load_model()
    explainer = shap.TreeExplainer(model)

    # Get SHAP values shape: (1, 5, 4)
    shap_values = explainer.shap_values(encoded_input)
    shap_array = np.array(shap_values)

    # Use predicted class SHAP values
    if predicted_class_idx is not None and 0 <= predicted_class_idx <= 3:
        class_shap = np.abs(shap_array[0, :, predicted_class_idx])
    else:
        class_shap = np.mean(np.abs(shap_array[0]), axis=1)

    # Get top N indices
    top_indices = np.argsort(class_shap)[::-1][:top_n]

    top_features = []
    for idx in top_indices:
        feature_name = LATENT_TO_ORIGINAL.get(int(idx), f'latent_{idx+1}')
        importance = round(float(class_shap[idx]), 4)
        top_features.append({
            'feature': feature_name,
            'importance': importance,
            'latent_index': int(idx)
        })

    return top_features


def get_top_feature_names(
    encoded_input: np.ndarray,
    predicted_class_idx: int = None,
    top_n: int = 3
) -> list:
    """
    Returns just feature names as simple list.
    Used directly by FastAPI response.
    """
    features = get_top_features_shap(encoded_input, predicted_class_idx, top_n)
    return [f['feature'] for f in features]


def get_human_explanation(
    encoded_input: np.ndarray,
    prediction: str
) -> dict:
    """
    Returns complete human readable explanation.
    This is what gets shown to security analysts!

    Returns:
        dict with message, top_features, reasons
    """
    pred_lower = prediction.lower()
    class_idx = CLASS_NAMES.index(pred_lower) if pred_lower in CLASS_NAMES else None

    # Get top features with SHAP
    top_features = get_top_features_shap(encoded_input, class_idx, top_n=3)
    feature_names = [f['feature'] for f in top_features]

    # Build human readable reasons
    reasons = []
    for feature in feature_names:
        if pred_lower in ATTACK_MESSAGES:
            reason = ATTACK_MESSAGES[pred_lower].get(
                feature,
                FEATURE_EXPLANATIONS.get(feature, feature)
            )
            reasons.append(reason)

    # Build final message
    if pred_lower == 'normal':
        message = "Traffic appears normal. No suspicious patterns detected."
    else:
        message = (f"{prediction.upper()} detected because: "
                  + "; ".join(reasons))

    return {
        'prediction': prediction,
        'message': message,
        'top_features': feature_names,
        'reasons': reasons
    }


# ── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing xai_engine.py with real SHAP...")
    print()

    dummy_encoded = np.random.rand(1, 5)

    for attack in CLASS_NAMES:
        result = get_human_explanation(dummy_encoded, attack)
        print(f"{'='*60}")
        print(f"Prediction : {attack.upper()}")
        print(f"Message    : {result['message']}")
        print(f"Features   : {result['top_features']}")
        print(f"Reasons    : {result['reasons']}")
        print()

    print("✅ xai_engine.py working correctly!")