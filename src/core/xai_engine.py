# src/core/xai_engine.py
# Explainable AI Engine — Updated for 6 classes + 33 features

import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# 6 classes in alphabetical order (matches LabelEncoder output)
CLASS_NAMES = ['dos', 'exploits', 'fuzzers', 'generic', 'normal', 'reconnaissance']

# Top features from the 33-dim scaled feature set
# These are the most important features XGBoost learned
FEATURE_EXPLANATIONS = {
    'dur':               'connection duration',
    'proto':             'network protocol type',
    'service':           'target service',
    'state':             'connection state',
    'spkts':             'source packet count',
    'dpkts':             'destination packet count',
    'rate':              'packet rate',
    'sttl':              'source time-to-live',
    'dttl':              'destination time-to-live',
    'sload':             'source bits per second load',
    'dload':             'destination bits per second load',
    'sinpkt':            'source inter-packet arrival time',
    'dinpkt':            'destination inter-packet arrival time',
    'sjit':              'source jitter',
    'djit':              'destination jitter',
    'swin':              'source TCP window size',
    'stcpb':             'source TCP base sequence number',
    'dtcpb':             'destination TCP base sequence number',
    'tcprtt':            'TCP round-trip time',
    'synack':            'SYN to SYN-ACK time',
    'ackdat':            'SYN-ACK to ACK time',
    'smean':             'mean source packet size',
    'dmean':             'mean destination packet size',
    'trans_depth':       'transaction depth',
    'response_body_len': 'response body length',
    'ct_srv_src':        'connections to same service from source',
    'ct_state_ttl':      'connections with same state and TTL',
    'ct_dst_ltm':        'connections to destination recently',
    'ct_dst_sport_ltm':  'connections to destination port recently',
    'is_ftp_login':      'FTP login attempt',
    'ct_flw_http_mthd':  'HTTP method count',
    'ct_src_ltm':        'connections from source recently',
    'is_sm_ips_ports':   'same source/destination IPs and ports'
}

# Human readable messages per attack per feature
ATTACK_MESSAGES = {
    'dos': {
        'sload':         'source is sending abnormally high traffic load',
        'dload':         'destination is being flooded with traffic',
        'rate':          'packet rate is extremely high — flood detected',
        'spkts':         'massive number of packets from single source',
        'ct_srv_src':    'same service being targeted repeatedly',
        'sinpkt':        'packets arriving too fast — flooding pattern',
        'smean':         'uniform packet sizes indicating automated flood'
    },
    'exploits': {
        'state':         'abnormal connection state indicates exploitation',
        'trans_depth':   'deep transaction chain — possible exploit payload',
        'response_body_len': 'unusual response size from exploitation attempt',
        'tcprtt':        'abnormal round-trip time during exploit',
        'service':       'specific service being actively exploited',
        'ct_state_ttl':  'abnormal TTL pattern during exploitation'
    },
    'fuzzers': {
        'spkts':         'high packet count with random data patterns',
        'smean':         'random varying packet sizes — fuzzing detected',
        'dur':           'many short connections — fuzzing pattern',
        'rate':          'rapid connection attempts with malformed data',
        'sjit':          'irregular timing pattern consistent with fuzzing',
        'state':         'many incomplete connections from fuzzing'
    },
    'generic': {
        'ct_src_ltm':    'source making many connections recently',
        'ct_dst_ltm':    'destination receiving many connections recently',
        'proto':         'unusual protocol usage for generic attack',
        'sttl':          'abnormal TTL value — possible spoofing',
        'dttl':          'destination TTL anomaly detected',
        'service':       'service being accessed in unusual pattern'
    },
    'reconnaissance': {
        'ct_dst_sport_ltm': 'source scanning many destination ports',
        'ct_srv_src':    'same source probing many services',
        'dur':           'very short connections — port scanning pattern',
        'spkts':         'low packet count per connection — probe pattern',
        'state':         'many RST/REQ states — scanning behavior',
        'rate':          'systematic scan rate detected across ports'
    },
    'normal': {
        'sload':         'traffic load is within normal range',
        'rate':          'packet rate is normal',
        'dur':           'connection duration is typical',
        'state':         'connection state is normal',
        'proto':         'standard protocol usage detected'
    }
}


def load_model():
    model_path = os.path.join(MODEL_DIR, 'xgb_multiclass.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def get_top_features_shap(
    scaled_input: np.ndarray,
    predicted_class_idx: int = None,
    top_n: int = 3
) -> list:
    import shap

    model     = load_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)
    shap_array  = np.array(shap_values)

    feature_names = list(FEATURE_EXPLANATIONS.keys())

    # Shape is (n_samples, n_features, n_classes) = (1, 33, 6)
    if shap_array.ndim == 3:
        if predicted_class_idx is not None and predicted_class_idx < shap_array.shape[2]:
            class_shap = np.abs(shap_array[0, :, predicted_class_idx])
        else:
            class_shap = np.mean(np.abs(shap_array[0]), axis=1)
    elif shap_array.ndim == 2:
        class_shap = np.abs(shap_array[0])
    else:
        class_shap = np.abs(shap_array.flatten()[:len(feature_names)])

    top_indices  = np.argsort(class_shap)[::-1][:top_n]
    top_features = []
    for idx in top_indices:
        fname      = feature_names[int(idx)] if int(idx) < len(feature_names) else f'feature_{idx}'
        importance = round(float(class_shap[idx]), 4)
        top_features.append({
            'feature':    fname,
            'importance': importance,
            'index':      int(idx)
        })
    return top_features


def get_top_feature_names(
    scaled_input: np.ndarray,
    predicted_class_idx: int = None,
    top_n: int = 3
) -> list:
    """Returns just feature names as a list."""
    features = get_top_features_shap(scaled_input, predicted_class_idx, top_n)
    return [f['feature'] for f in features]


def get_human_explanation(
    scaled_input: np.ndarray,
    prediction: str
) -> dict:
    """
    Returns complete human readable explanation
    for security analysts.
    """
    pred_lower = prediction.lower()
    class_idx  = CLASS_NAMES.index(pred_lower) if pred_lower in CLASS_NAMES else None

    top_features  = get_top_features_shap(scaled_input, class_idx, top_n=3)
    feature_names = [f['feature'] for f in top_features]

    reasons = []
    for feature in feature_names:
        attack_msgs = ATTACK_MESSAGES.get(pred_lower, {})
        reason = attack_msgs.get(
            feature,
            FEATURE_EXPLANATIONS.get(feature, feature)
        )
        reasons.append(reason)

    if pred_lower == 'normal':
        message = "Traffic appears normal. No suspicious patterns detected."
    else:
        message = f"{prediction.upper()} detected because: " + "; ".join(reasons)

    return {
        'prediction':  prediction,
        'message':     message,
        'top_features': feature_names,
        'reasons':     reasons
    }


if __name__ == "__main__":
    print("Testing xai_engine.py — 6 classes, 33 features\n")
    dummy_input = np.random.rand(1, 33)
    for attack in CLASS_NAMES:
        result = get_human_explanation(dummy_input, attack)
        print(f"{'='*55}")
        print(f"Prediction : {attack.upper()}")
        print(f"Message    : {result['message'][:80]}...")
        print(f"Features   : {result['top_features']}")
    print("\n✅ xai_engine.py updated for 6 classes + 33 features!")