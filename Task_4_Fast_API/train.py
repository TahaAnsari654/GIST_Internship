"""Train + tune a model and save artefacts (model.joblib, meta.json)."""
import json, joblib, numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

data = load_breast_cancer(as_frame=True)
X = data.data.filter(like="mean").copy()              # 10 "mean" features
X.columns = [c.replace(" ", "_") for c in X.columns]
y = data.target                                       # 1 = benign, 0 = malignant

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([("scaler", StandardScaler()),
                 ("clf", RandomForestClassifier(random_state=42))])
params = {"clf__n_estimators": [100, 200, 400],
          "clf__max_depth": [None, 5, 10, 20],
          "clf__min_samples_split": [2, 5, 10],
          "clf__min_samples_leaf": [1, 2, 4]}
search = RandomizedSearchCV(pipe, params, n_iter=20, cv=5, scoring="f1",
                            random_state=42, n_jobs=-1)
search.fit(X_tr, y_tr)
best = search.best_estimator_

pred, proba = best.predict(X_te), best.predict_proba(X_te)[:, 1]
metrics = {"accuracy": accuracy_score(y_te, pred), "f1": f1_score(y_te, pred),
           "roc_auc": roc_auc_score(y_te, proba), "cv_f1": search.best_score_}
print(classification_report(y_te, pred, target_names=["malignant", "benign"]))
print("Best params:", search.best_params_, "\nMetrics:", metrics)

joblib.dump(best, "model/model.joblib")
meta = {"features": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                         "mean": float(X[c].mean())} for c in X.columns},
        "classes": {"0": "malignant", "1": "benign"},
        "best_params": search.best_params_,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()}}
json.dump(meta, open("model/meta.json", "w"), indent=2)
