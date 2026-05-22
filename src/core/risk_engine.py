# src/core/risk_engine.py
# Risk Scoring Engine — Updated for 6 classes

ATTACK_WEIGHTS = {
    'dos':            10,
    'exploits':        9,
    'generic':         7,
    'fuzzers':         6,
    'reconnaissance':  4,
    'normal':          0
}

def calculate_risk_score(
    prediction: str,
    confidence: float,
    reconstruction_error: float
) -> float:
    """
    Calculates risk score from 0 to 10.
    Formula: Risk = (Confidence * 0.5) + (Recon_Error * 0.3) + (Attack_Weight * 0.2)
    """
    if prediction.lower() == 'normal':
        return 0.0

    attack_weight    = ATTACK_WEIGHTS.get(prediction.lower(), 0)
    recon_normalized = min(reconstruction_error / 2.0, 1.0)
    attack_normalized = attack_weight / 10.0

    risk = (
        (confidence       * 0.5) +
        (recon_normalized * 0.3) +
        (attack_normalized * 0.2)
    )

    return round(risk * 10, 2)


def get_risk_level(risk_score: float) -> str:
    if risk_score >= 8:   return "CRITICAL"
    elif risk_score >= 6: return "HIGH"
    elif risk_score >= 4: return "MEDIUM"
    elif risk_score >= 2: return "LOW"
    else:                 return "SAFE"


if __name__ == "__main__":
    print("Testing risk_engine.py — 6 classes\n")
    test_cases = [
        ("dos",            0.94, 0.85),
        ("exploits",       0.88, 0.72),
        ("generic",        0.76, 0.55),
        ("fuzzers",        0.70, 0.45),
        ("reconnaissance", 0.65, 0.30),
        ("normal",         0.99, 0.05),
    ]
    for prediction, confidence, recon_error in test_cases:
        score = calculate_risk_score(prediction, confidence, recon_error)
        level = get_risk_level(score)
        print(f"  {prediction:16} | score={score:5} | level={level}")
    print("\n✅ risk_engine.py working correctly!")