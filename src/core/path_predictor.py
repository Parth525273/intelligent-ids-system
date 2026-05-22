# src/core/path_predictor.py
# Attack Path Predictor — Markov Chain + Cyber Kill Chain
# Updated for 6 classes: dos, exploits, fuzzers, generic, normal, reconnaissance

import os
from datetime import datetime
from collections import defaultdict

# ── Cyber Kill Chain Stages ──────────────────────────────────
KILL_CHAIN_STAGES = {
    'reconnaissance': 'Reconnaissance',
    'fuzzers':        'Weaponization',
    'exploits':       'Exploitation',
    'generic':        'Command & Control',
    'dos':            'Impact',
    'normal':         'None'
}

# ── Markov Chain Transition Matrix ───────────────────────────
# P(next_attack | current_attack)
TRANSITION_MATRIX = {
    'reconnaissance': {
        'exploits':       0.40,
        'fuzzers':        0.25,
        'dos':            0.20,
        'generic':        0.10,
        'reconnaissance': 0.05,
        'normal':         0.00
    },
    'fuzzers': {
        'exploits':       0.45,
        'dos':            0.25,
        'generic':        0.15,
        'reconnaissance': 0.10,
        'fuzzers':        0.05,
        'normal':         0.00
    },
    'exploits': {
        'generic':        0.40,
        'dos':            0.30,
        'fuzzers':        0.15,
        'reconnaissance': 0.10,
        'exploits':       0.05,
        'normal':         0.00
    },
    'generic': {
        'dos':            0.45,
        'exploits':       0.25,
        'fuzzers':        0.15,
        'reconnaissance': 0.10,
        'generic':        0.05,
        'normal':         0.00
    },
    'dos': {
        'exploits':       0.35,
        'generic':        0.30,
        'reconnaissance': 0.20,
        'fuzzers':        0.10,
        'dos':            0.05,
        'normal':         0.00
    },
    'normal': {
        'reconnaissance': 0.40,
        'fuzzers':        0.25,
        'exploits':       0.20,
        'generic':        0.10,
        'dos':            0.05,
        'normal':         0.00
    }
}

# ── Attack Descriptions ──────────────────────────────────────
ATTACK_DESCRIPTIONS = {
    'reconnaissance': {
        'what':    'Attacker is scanning network to discover open ports and services',
        'why':     'Gathering intelligence before launching the main attack',
        'urgency': 'MEDIUM — Attack preparation phase detected'
    },
    'fuzzers': {
        'what':    'Attacker is sending malformed/random data to find vulnerabilities',
        'why':     'Probing for software bugs that can be exploited',
        'urgency': 'HIGH — Active vulnerability probing'
    },
    'exploits': {
        'what':    'Attacker is actively exploiting known vulnerabilities',
        'why':     'Attempting to gain unauthorized access or execute code',
        'urgency': 'CRITICAL — Active exploitation in progress'
    },
    'generic': {
        'what':    'Generic attack pattern detected — possible C&C communication',
        'why':     'Attacker may have established foothold and is maintaining access',
        'urgency': 'HIGH — Possible system compromise'
    },
    'dos': {
        'what':    'Attacker is flooding the network to overwhelm services',
        'why':     'Disrupting operations or creating distraction for another attack',
        'urgency': 'CRITICAL — Active service disruption'
    },
    'normal': {
        'what':    'Traffic appears normal and benign',
        'why':     'No malicious intent detected',
        'urgency': 'NONE — No action required'
    }
}

# ── Recommended Actions ──────────────────────────────────────
RECOMMENDED_ACTIONS = {
    'reconnaissance': [
        'Block source IP at firewall immediately',
        'Enable intrusion prevention system (IPS)',
        'Monitor for follow-up exploit attempts',
        'Review all exposed service vulnerabilities'
    ],
    'fuzzers': [
        'Enable input validation on all endpoints',
        'Block source IP immediately',
        'Review application logs for crash patterns',
        'Patch all known vulnerabilities immediately'
    ],
    'exploits': [
        'Isolate affected systems immediately',
        'Block source IP at firewall and upstream',
        'Alert security team for incident response',
        'Check for lateral movement in the network'
    ],
    'generic': [
        'Inspect all outbound traffic for C&C patterns',
        'Block suspicious IPs at perimeter firewall',
        'Run full malware scan on affected systems',
        'Review authentication logs for anomalies'
    ],
    'dos': [
        'Enable rate limiting on all endpoints immediately',
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
_attack_history = defaultdict(list)


def record_attack(source_ip: str, attack_type: str):
    """Records an attack in history for this IP. Keeps last 10."""
    if attack_type == 'normal':
        return

    _attack_history[source_ip].append({
        'attack': attack_type,
        'timestamp': datetime.now().isoformat()
    })

    if len(_attack_history[source_ip]) > 10:
        _attack_history[source_ip] = _attack_history[source_ip][-10:]


def get_attack_history(source_ip: str) -> list:
    """Returns attack history for an IP."""
    return _attack_history.get(source_ip, [])


def predict_next_attack(current_attack: str) -> dict:
    """Uses Markov Chain to predict most likely next attack."""
    if current_attack not in TRANSITION_MATRIX:
        current_attack = 'normal'

    transitions = TRANSITION_MATRIX[current_attack]
    next_attack  = max(transitions, key=transitions.get)
    probability  = transitions[next_attack]

    all_predictions = sorted(
        transitions.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        'most_likely_next': next_attack,
        'probability':      probability,
        'all_predictions': [
            {
                'attack':      atk,
                'probability': prob,
                'percentage':  f"{prob*100:.0f}%"
            }
            for atk, prob in all_predictions
            if atk != 'normal'
        ]
    }


def get_attack_chain(source_ip: str) -> list:
    """Returns complete attack chain for an IP."""
    history = get_attack_history(source_ip)
    if not history:
        return []

    chain = []
    for i, event in enumerate(history):
        stage = KILL_CHAIN_STAGES.get(event['attack'], 'Unknown')
        chain.append({
            'step':      i + 1,
            'attack':    event['attack'],
            'stage':     stage,
            'timestamp': event['timestamp']
        })
    return chain


def predict_attack_path(prediction: str, source_ip: str = "unknown") -> dict:
    """Main function — complete attack path analysis."""
    pred_lower = prediction.lower()

    # Fallback for unknown class
    if pred_lower not in KILL_CHAIN_STAGES:
        pred_lower = 'normal'

    record_attack(source_ip, pred_lower)

    current_stage = KILL_CHAIN_STAGES.get(pred_lower, 'None')
    description   = ATTACK_DESCRIPTIONS.get(pred_lower, {})
    actions       = RECOMMENDED_ACTIONS.get(pred_lower, [])
    next_pred     = predict_next_attack(pred_lower)
    history       = get_attack_chain(source_ip)

    if pred_lower == 'normal':
        return {
            'prediction':            pred_lower,
            'current_stage':         'None',
            'next_stage':            'No Threat Detected',
            'what_is_happening':     description.get('what', 'Normal traffic'),
            'why':                   description.get('why', ''),
            'urgency':               'NONE',
            'recommended_actions':   actions,
            'next_attack_prediction': None,
            'attack_history':        history
        }

    return {
        'prediction':             pred_lower,
        'current_stage':          current_stage,
        'next_stage':             next_pred['most_likely_next'].upper(),
        'next_stage_probability': next_pred['probability'],
        'what_is_happening':      description.get('what', ''),
        'why':                    description.get('why', ''),
        'urgency':                description.get('urgency', ''),
        'recommended_actions':    actions,
        'next_attack_prediction': next_pred,
        'attack_history':         history
    }


def get_next_stage(prediction: str) -> str:
    """Quick helper for FastAPI response."""
    return predict_attack_path(prediction)['next_stage']


# ── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing path_predictor.py — 6 class version\n")

    attacker_ip     = "192.168.1.100"
    attack_sequence = ['reconnaissance', 'fuzzers', 'exploits', 'dos']

    for attack in attack_sequence:
        result = predict_attack_path(attack, attacker_ip)
        print(f"{'='*55}")
        print(f"Current : {result['prediction'].upper()}")
        print(f"Stage   : {result['current_stage']}")
        print(f"Urgency : {result['urgency']}")
        print(f"Next    : {result['next_stage']} "
              f"({result.get('next_stage_probability', 0)*100:.0f}%)")
        print(f"Action  : {result['recommended_actions'][0]}")

    print(f"\n{'='*55}")
    print("Full attack chain:")
    for step in get_attack_chain(attacker_ip):
        print(f"  Step {step['step']}: {step['attack'].upper()} → {step['stage']}")

    print("\n✅ path_predictor.py updated for 6 classes!")