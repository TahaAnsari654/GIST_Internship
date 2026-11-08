# ML Model Deployment: Tumor Classifier (FastAPI + Dashboard)

## 1. Pipeline overview
Data (sklearn Breast Cancer, 10 mean features) -> Train/test split (80/20, stratified)
-> Pipeline(StandardScaler + RandomForest) -> RandomizedSearchCV (5-fold, F1)
-> Evaluate -> Save `model/model.joblib` + `model/meta.json`
-> FastAPI (`app.py`) loads artefacts -> validated `/predict` -> Dashboard (`static/index.html`)
-> Tests (pytest) -> Docker.

## 2. Run
    pip install -r requirements.txt
    python train.py                 # trains, tunes, saves artefacts
    pytest -q tests                 # API + validation tests
    uvicorn app:app --reload        # open http://127.0.0.1:8000  (docs: /docs)

## 3. API
| Method | Path | Purpose |
|---|---|---|
| GET | /health | liveness check |
| GET | /meta | feature ranges + model metrics (dashboard uses this) |
| POST | /predict | body: 10 floats -> label + probabilities |

Example:
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"mean_radius":14,"mean_texture":19,"mean_perimeter":92,"mean_area":650,"mean_smoothness":0.096,"mean_compactness":0.104,"mean_concavity":0.089,"mean_concave_points":0.048,"mean_symmetry":0.181,"mean_fractal_dimension":0.063}'

## 4. Input validation (two layers)
- Server: Pydantic schema, each field required, numeric, `> 0` and `<= 2 x training max`.
  Bad input returns HTTP 422 with the exact field error. Internal errors return a generic 500.
- Client: the dashboard checks the same ranges before sending and shows per-field messages.

## 5. Model results (test set)
Accuracy ~0.956, F1 ~0.965, ROC-AUC ~0.99, CV F1 ~0.948.

## 6. Docker
    docker build -t ml-api . && docker run -p 8000:8000 ml-api

## 7. Limitations / next steps
Educational model, not for medical use. Add auth, rate limiting, model versioning,
drift monitoring, and deploy on Render/Railway/AWS.
