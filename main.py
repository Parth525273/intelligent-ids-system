# main.py
# Intelligent IDS System — FastAPI Entry Point
# Updated for 6 classes + 33 features

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import io
import warnings
import os
from dotenv import load_dotenv
import pathlib
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / '.env')


warnings.filterwarnings('ignore')

# ── Load Models at Startup ───────────────────────────────────
print("Loading models...")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'src', 'models')

try:
    from tensorflow.keras.models import load_model as keras_load_model
    encoder_model = keras_load_model(
        os.path.join(MODEL_DIR, 'encoder_model.h5'), compile=False
    )
    autoencoder_model = keras_load_model(
        os.path.join(MODEL_DIR, 'autoencoder_xg_model.h5'), compile=False
    )
    print("✅ Encoder and Autoencoder loaded!")
except Exception as e:
    print(f"❌ Error loading Keras models: {e}")
    raise

try:
    scaler    = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, 'xgb_multiclass.joblib'))
    le_label  = joblib.load(os.path.join(MODEL_DIR, 'le_label.joblib'))
    print("✅ Scaler, XGBoost, LabelEncoder loaded!")
except Exception as e:
    print(f"❌ Error loading joblib models: {e}")
    raise

# ── Import Intelligence Modules ──────────────────────────────
from src.preprocessing.cleaner import preprocess_single
from src.core.risk_engine       import calculate_risk_score, get_risk_level
from src.core.xai_engine        import get_human_explanation
from src.core.path_predictor    import predict_attack_path

# 6 classes we trained on
VALID_CLASSES = ['dos', 'exploits', 'fuzzers', 'generic', 'normal', 'reconnaissance']

print("✅ All modules imported!")

# ── FastAPI App ──────────────────────────────────────────────
app = FastAPI(
    title="Intelligent IDS System",
    description="Hybrid Autoencoder + XGBoost IDS with Risk Scoring, XAI and Attack Path Prediction",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

class AIQuestion(BaseModel):
    question: str

# ── Input Schema ─────────────────────────────────────────────
# Matches exact 33 features after correlation filter
class NetworkTraffic(BaseModel):
    dur:              float = 0.001055
    proto:            str   = "udp"
    service:          str   = "-"
    state:            str   = "CON"
    spkts:            int   = 1
    dpkts:            int   = 1
    rate:             float = 0.0
    sttl:             int   = 31
    dttl:             int   = 29
    sload:            float = 0.0
    dload:            float = 0.0
    sinpkt:           float = 0.0
    dinpkt:           float = 0.0
    sjit:             float = 0.0
    djit:             float = 0.0
    swin:             int   = 0
    stcpb:            int   = 0
    dtcpb:            int   = 0
    tcprtt:           float = 0.0
    synack:           float = 0.0
    ackdat:           float = 0.0
    smean:            int   = 132
    dmean:            int   = 164
    trans_depth:      int   = 0
    response_body_len:int   = 0
    ct_srv_src:       int   = 3
    ct_state_ttl:     int   = 0
    ct_dst_ltm:       int   = 1
    ct_dst_sport_ltm: int   = 1
    is_ftp_login:     int   = 0
    ct_flw_http_mthd: int   = 0
    ct_src_ltm:       int   = 1
    is_sm_ips_ports:  int   = 0
    source_ip: Optional[str] = "unknown"

    class Config:
        json_schema_extra = {
            "example": {
                "dur": 0.121478, "proto": "tcp", "service": "http",
                "state": "FIN", "spkts": 6, "dpkts": 4,
                "rate": 74.08, "sttl": 252, "dttl": 62,
                "sload": 195021.0, "dload": 316354.0,
                "sinpkt": 20.5, "dinpkt": 18.3,
                "sjit": 10.5, "djit": 8.2,
                "swin": 255, "stcpb": 12345, "dtcpb": 67890,
                "tcprtt": 0.05, "synack": 0.0, "ackdat": 0.0,
                "smean": 493, "dmean": 800,
                "trans_depth": 1, "response_body_len": 4800,
                "ct_srv_src": 15, "ct_state_ttl": 4,
                "ct_dst_ltm": 10, "ct_dst_sport_ltm": 1,
                "is_ftp_login": 0, "ct_flw_http_mthd": 1,
                "ct_src_ltm": 1, "is_sm_ips_ports": 0,
                "source_ip": "192.168.1.100"
            }
        }


# ── Helper: Run full prediction pipeline ─────────────────────
def run_pipeline(row_dict: dict, source_ip: str = "unknown") -> dict:
    """
    Full pipeline: preprocess → encode → predict → explain.
    Used by both /predict and /analyze-csv endpoints.
    """
    # Step 1: Preprocess → 33-dim scaled features
    scaled_input = preprocess_single(row_dict)

    # Step 2: Predict attack type (XGBoost on scaled features)
    pred_num    = xgb_model.predict(scaled_input)[0]
    prediction  = le_label.inverse_transform([pred_num])[0]

    # Fallback if prediction not in our 6 classes
    if prediction not in VALID_CLASSES:
        prediction = 'normal'

    confidence  = float(xgb_model.predict_proba(scaled_input)[0].max())

    # Step 3: Reconstruction error (autoencoder on scaled features)
    reconstructed = autoencoder_model.predict(scaled_input, verbose=0)
    recon_error   = float(np.mean((reconstructed - scaled_input) ** 2))

    # Step 4: Risk score
    risk_score = calculate_risk_score(prediction, confidence, recon_error)
    risk_level = get_risk_level(risk_score)

    # Step 5: XAI explanation (SHAP on scaled features)
    explanation = get_human_explanation(scaled_input, prediction)

    # Step 6: Attack path prediction
    path = predict_attack_path(prediction, source_ip)

    return {
        "prediction":           prediction,
        "confidence":           round(confidence, 4),
        "risk_score":           risk_score,
        "risk_level":           risk_level,
        "explanation":          explanation["message"],
        "top_features":         explanation["top_features"],
        "reasons":              explanation["reasons"],
        "reconstruction_error": round(recon_error, 6),
        "current_stage":        path["current_stage"],
        "next_attack":          path["next_stage"],
        "next_probability":     round(path.get("next_stage_probability", 0), 2),
        "recommended_actions":  path["recommended_actions"],
        "attack_history_count": len(path["attack_history"]),
        "source_ip":            source_ip
    }


# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message":   "Intelligent IDS System API",
        "version":   "2.0.0",
        "classes":   VALID_CLASSES,
        "docs":      "/docs",
        "dashboard": "/dashboard",
        "predict":   "/predict"
    }


@app.get("/health")
def health():
    return {
        "status":        "running",
        "models_loaded": True,
        "classes":       VALID_CLASSES,
        "features":      33,
        "encoder":       "encoder_model.h5",
        "classifier":    "xgb_multiclass.joblib"
    }


@app.post("/predict")
def predict(traffic: NetworkTraffic):
    """
    Main prediction endpoint.
    Takes 33 network traffic features, returns full security analysis.
    """
    try:
        source_ip    = traffic.source_ip or "unknown"
        traffic_dict = traffic.dict()
        traffic_dict.pop('source_ip', None)

        result = run_pipeline(traffic_dict, source_ip)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard")
def dashboard(request: Request):
    """Serve the HTML dashboard."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    """
    Batch CSV analysis endpoint.
    Accepts UNSW_NB15 testing-set CSV format.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Sample max 200 rows
        if len(df) > 200:
            step = len(df) // 200
            df   = df.iloc[::step].head(200)

        # Columns to remove before prediction
        DROP_COLS = [
            'id', 'label', 'Label', 'attack_cat',
            'srcip', 'dstip', 'sport', 'dsport',
            'sbytes', 'dbytes', 'sloss', 'dloss', 'dwin',
            'ct_src_dport_ltm', 'ct_dst_src_ltm',
            'ct_ftp_cmd', 'ct_srv_dst'
        ]

        results = []
        for _, row in df.iterrows():
            try:
                row_dict = row.to_dict()

                # Save metadata before dropping
                source_ip    = str(row_dict.get('srcip', 'unknown'))
                actual_label = str(row_dict.get('attack_cat', 'unknown'))

                # Drop non-feature columns
                for col in DROP_COLS:
                    row_dict.pop(col, None)

                result = run_pipeline(row_dict, source_ip)
                result['actual_label'] = actual_label
                results.append(result)

            except Exception:
                continue

        return {"results": results, "total": len(results)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run Server ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

    
@app.post("/predict")
def predict(traffic: NetworkTraffic):
    try:
        source_ip    = traffic.source_ip or "unknown"
        traffic_dict = traffic.dict()
        traffic_dict.pop('source_ip', None)
        result = run_pipeline(traffic_dict, source_ip)
        return result
    except Exception as e:
        import traceback
        print("=" * 50)
        print(traceback.format_exc())
        print("=" * 50)
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/ai-chat")
async def ai_chat(body: AIQuestion):
    try:
        import httpx
        OR_KEY = os.getenv("OPENROUTER_API_KEY", "")

        if not OR_KEY:
            return {"response": "API key not configured."}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OR_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Intelligent IDS"
                },
                json={
                    "model": "openrouter/free",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"You are ARIA, a cybersecurity AI analyst in an IDS dashboard at VIT Pune. Answer concisely in 2-4 sentences. Question: {body.question}"
                        }
                    ],
                    "max_tokens": 300
                },
                timeout=15.0
            )
            data = response.json()

        print("OpenRouter response:", data)

        if 'error' in data:
            print("Error:", data['error'])
            return {"response": f"API Error: {data['error'].get('message', 'Unknown error')}"}

        answer = data['choices'][0]['message']['content']
        return {"response": answer}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"response": f"Error: {str(e)}"}
    
    