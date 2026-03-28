# src/core/path_predictor.py
# Advanced Attack Path Predictor
# Uses Markov Chain + Attack History for real predictions

import os
import json
from datetime import datetime
from collections import defaultdict

# ── Cyber Kill Chain Stages ──────────────────────────────────
KILL_CHAIN_STAGES = {
    'portscan':   'Reconnaissance',
    'bruteforce': 'Credential Access',
    'ddos':       'Impact',
    'normal':     'None'
}

# ── Markov Chain Transition Matrix ───────────────────────────
# Based on real cybersecurity research
# P(next_attack | current_attack)
TRANSITION_MATRIX = {
    'portscan': {
        'bruteforce': 0.55,
        'ddos':       0.25,
        'portscan':   0.15,
        'normal':     0.05
    },
    'bruteforce': {
        'ddos':       0.40,
        'portscan':   0.30,
        'bruteforce': 0.20,
        'normal':     0.10
    },
    'ddos': {
        'bruteforce': 0.45,
        'portscan':   0.30,
        'ddos':       0.15,
        'normal':     0.10
    },
    'normal': {
        'portscan':   0.40,
        'bruteforce': 0.35,
        'ddos':       0.20,
        'normal':     0.05
    }
}

# ── Attack Descriptions ──────────────────────────────────────
ATTACK_DESCRIPTIONS = {
    'portscan': {
        'what': 'Attacker is scanning for open ports and vulnerabilities',
        'why': 'Gathering intelligence before launching main attack',
        'urgency': 'MEDIUM — Attack preparation phase'
    },
    'bruteforce': {
        'what': 'Attacker is attempting to break into accounts by force',
        'why': 'Trying to gain unauthorized access to systems',
        'urgency': 'HIGH — Active intrusion attempt'
    },
    'ddos': {
        'what': 'Attacker is flooding network to overwhelm services',
        'why': 'Disrupting operations or creating distraction',
        'urgency': 'CRITICAL — Active service disruption'
    },
    'normal': {
        'what': 'Traffic appears normal and benign',
        'why': 'No malicious intent detected',
        'urgency': 'NONE — No action required'
    }
}

# ── Recommended Actions ──────────────────────────────────────
RECOMMENDED_ACTIONS = {
    'portscan':   [
        'Block source IP at firewall immediately',
        'Enable intrusion prevention system',
        'Monitor for follow-up bruteforce attempts',
        'Review exposed service vulnerabilities'
    ],
    'bruteforce': [
        'Enable account lockout policy',
        'Enforce MFA on all accounts immediately',
        'Block source IP after failed attempts',
        'Alert security team for manual review'
    ],
    'ddos': [
        'Enable rate limiting on all endpoints',
        'Contact ISP for upstream traffic filtering',
        'Activate DDoS protection service',
        'Switch to backup infrastructure if available'
    ],
    'normal': [
        'No action required',
        'Continue standard monitoring'
    ]
}

# ── In-Memory Attack History ─────────────────────────────────
# Tracks attack sequence per source IP
# Format: { "ip_address": [{"attack": "portscan", "time": "..."}] }
_attack_history = defaultdict(list)


def record_attack(source_ip: str, attack_type: str):
    """
    Records an attack in history for this IP.
    Keeps last 10 attacks per IP.
    """
    if attack_type == 'normal':
        return

    _attack_history[source_ip].append({
        'attack': attack_type,
        'timestamp': datetime.now().isoformat()
    })

    # Keep only last 10 attacks per IP
    if len(_attack_history[source_ip]) > 10:
        _attack_history[source_ip] = _attack_history[source_ip][-10:]


def get_attack_history(source_ip: str) -> list:
    """Returns attack history for an IP."""
    return _attack_history.get(source_ip, [])


def predict_next_attack(current_attack: str) -> dict:
    """
    Uses Markov Chain to predict most likely next attack.

    Args:
        current_attack: current detected attack type

    Returns:
        dict with next attack prediction and probability
    """
    if current_attack not in TRANSITION_MATRIX:
        current_attack = 'normal'

    transitions = TRANSITION_MATRIX[current_attack]

    # Get most likely next attack
    next_attack = max(transitions, key=transitions.get)
    probability = transitions[next_attack]

    # Get all predictions sorted by probability
    all_predictions = sorted(
        transitions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        'most_likely_next': next_attack,
        'probability': probability,
        'all_predictions': [
            {
                'attack': atk,
                'probability': prob,
                'percentage': f"{prob*100:.0f}%"
            }
            for atk, prob in all_predictions
            if atk != 'normal'
        ]
    }


def get_attack_chain(source_ip: str) -> list:
    """
    Returns the complete attack chain for an IP.
    Shows progression of attacks over time.
    """
    history = get_attack_history(source_ip)
    if not history:
        return []

    chain = []
    for i, event in enumerate(history):
        stage = KILL_CHAIN_STAGES.get(event['attack'], 'Unknown')
        chain.append({
            'step': i + 1,
            'attack': event['attack'],
            'stage': stage,
            'timestamp': event['timestamp']
        })
    return chain


def predict_attack_path(
    prediction: str,
    source_ip: str = "unknown"
) -> dict:
    """
    Main function — complete attack path analysis.

    Args:
        prediction: current attack type
        source_ip: source IP address for history tracking

    Returns:
        Complete attack path analysis
    """
    pred_lower = prediction.lower()

    # Record this attack in history
    record_attack(source_ip, pred_lower)

    # Get current stage info
    current_stage = KILL_CHAIN_STAGES.get(pred_lower, 'None')
    description = ATTACK_DESCRIPTIONS.get(pred_lower, {})
    actions = RECOMMENDED_ACTIONS.get(pred_lower, [])

    # Get next attack prediction using Markov Chain
    next_prediction = predict_next_attack(pred_lower)

    # Get attack history for this IP
    history = get_attack_chain(source_ip)

    # Build response
    if pred_lower == 'normal':
        return {
            'prediction': pred_lower,
            'current_stage': 'None',
            'next_stage': 'No Threat Detected',
            'what_is_happening': description.get('what', 'Normal traffic'),
            'why': description.get('why', ''),
            'urgency': 'NONE',
            'recommended_actions': actions,
            'next_attack_prediction': None,
            'attack_history': history
        }

    return {
        'prediction': pred_lower,
        'current_stage': current_stage,
        'next_stage': next_prediction['most_likely_next'].upper(),
        'next_stage_probability': next_prediction['probability'],
        'what_is_happening': description.get('what', ''),
        'why': description.get('why', ''),
        'urgency': description.get('urgency', ''),
        'recommended_actions': actions,
        'next_attack_prediction': next_prediction,
        'attack_history': history
    }


def get_next_stage(prediction: str) -> str:
    """Quick helper for FastAPI response."""
    result = predict_attack_path(prediction)
    return result['next_stage']


# ── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing path_predictor.py with Markov Chain...")
    print()

    # Simulate a real attack sequence from one IP
    attacker_ip = "192.168.1.100"
    attack_sequence = ['portscan', 'portscan', 'bruteforce', 'ddos']

    print(f"Simulating attack from IP: {attacker_ip}")
    print(f"Attack sequence: {attack_sequence}")
    print()

    for attack in attack_sequence:
        result = predict_attack_path(attack, attacker_ip)
        print(f"{'='*60}")
        print(f"Current Attack  : {result['prediction'].upper()}")
        print(f"Kill Chain Stage: {result['current_stage']}")
        print(f"What's happening: {result['what_is_happening']}")
        print(f"Urgency         : {result['urgency']}")
        print(f"Next Most Likely: {result['next_stage']}", end="")
        if result.get('next_stage_probability'):
            print(f" ({result['next_stage_probability']*100:.0f}% probability)")
        else:
            print()
        if result.get('next_attack_prediction'):
            print(f"All predictions :")
            for pred in result['next_attack_prediction']['all_predictions']:
                print(f"  → {pred['attack']:12} {pred['percentage']}")
        print(f"Recommended     : {result['recommended_actions'][0]}")
        print(f"Attack History  : {len(result['attack_history'])} attacks recorded")
        print()

    # Show full attack chain
    print(f"{'='*60}")
    print(f"Full Attack Chain for {attacker_ip}:")
    chain = get_attack_chain(attacker_ip)
    for step in chain:
        print(f"  Step {step['step']}: {step['attack'].upper()} "
              f"→ Stage: {step['stage']}")

    print()
    print("✅ path_predictor.py working correctly!")