"""FastAPI service: validated /predict endpoint + dashboard at /."""
import json, logging, joblib, pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, create_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")
BASE = Path(__file__).parent
model = joblib.load(BASE / "model/model.joblib")
meta = json.loads((BASE / "model/meta.json").read_text())

# Input validation: schema built from training-data ranges (gt 0, <= 2x observed max).
Features = create_model(
    "Features",
    **{n: (float, Field(..., gt=0, le=r["max"] * 2, description=f"{n} (> 0)"))
       for n, r in meta["features"].items()})

class Prediction(BaseModel):
    label: str
    benign_probability: float
    malignant_probability: float

app = FastAPI(title="Tumor Classifier API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/meta")
def get_meta():
    return meta

@app.post("/predict", response_model=Prediction)
def predict(x: Features):
    try:
        df = pd.DataFrame([x.model_dump()])[list(meta["features"])]
        p = float(model.predict_proba(df)[0, 1])
    except Exception as e:                       # never leak internals
        log.exception("prediction failed")
        raise HTTPException(500, "Prediction failed") from e
    log.info("predict benign_p=%.3f", p)
    return Prediction(label="benign" if p >= 0.5 else "malignant",
                      benign_probability=round(p, 4),
                      malignant_probability=round(1 - p, 4))

app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

@app.get("/")
def index():
    return FileResponse(BASE / "static/index.html")
