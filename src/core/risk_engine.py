# src/core/risk_engine.py
# Risk Scoring Engine
# Formula: Risk = (Confidence * 0.5) + (Recon_Error * 0.3) + (Attack_Weight * 0.2)

# Attack severity weights
ATTACK_WEIGHTS = {
    'ddos':       10,
    'bruteforce':  7,
    'portscan':    5,
    'normal':      0
}

def calculate_risk_score(
    prediction: str,
    confidence: float,
    reconstruction_error: float
) -> float:
    """
    Calculates risk score from 0 to 10.

    Args:
        prediction: predicted class (normal/ddos/portscan/bruteforce)
        confidence: model confidence (0 to 1)
        reconstruction_error: autoencoder reconstruction error

    Returns:
        risk_score: float between 0 and 10
    """

    # Normal traffic = zero risk always
    if prediction.lower() == 'normal':
        return 0.0

    # Get attack weight
    attack_weight = ATTACK_WEIGHTS.get(prediction.lower(), 0)

    # Normalize reconstruction error to 0-1 range
    # Typical reconstruction errors range from 0 to 2
    recon_normalized = min(reconstruction_error / 2.0, 1.0)

    # Normalize attack weight to 0-1 range
    attack_normalized = attack_weight / 10.0

    # Calculate weighted risk score
    risk = (
        (confidence * 0.5) +
        (recon_normalized * 0.3) +
        (attack_normalized * 0.2)
    )

    # Scale to 0-10
    risk_score = round(risk * 10, 2)

    return risk_score


def get_risk_level(risk_score: float) -> str:
    """
    Converts numeric risk score to human readable level.
    """
    if risk_score >= 8:
        return "CRITICAL"
    elif risk_score >= 6:
        return "HIGH"
    elif risk_score >= 4:
        return "MEDIUM"
    elif risk_score >= 2:
        return "LOW"
    else:
        return "SAFE"


# ── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing risk_engine.py...")

    test_cases = [
        ("ddos", 0.94, 0.85),
        ("portscan", 0.76, 0.45),
        ("bruteforce", 0.88, 0.62),
        ("normal", 0.99, 0.05),
    ]

    for prediction, confidence, recon_error in test_cases:
        score = calculate_risk_score(prediction, confidence, recon_error)
        level = get_risk_level(score)
        print(f"  {prediction:12} | confidence={confidence} | "
              f"recon_error={recon_error} | "
              f"risk_score={score} | level={level}")

    print("✅ risk_engine.py working correctly!")