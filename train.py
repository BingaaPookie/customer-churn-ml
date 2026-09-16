from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data_utils import load_customer_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURES

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = load_customer_data(DATA_PATH)
X = df[FEATURES].copy()
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_FEATURES),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CATEGORICAL_FEATURES),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1, min_samples_leaf=2),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=120, learning_rate=0.05, max_depth=2),
}

results = {}
best_name = None
best_f1 = -1

for name, estimator in models.items():
    pipe = Pipeline([( "preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    results[name] = metrics
    print(f"{name}: {metrics}")
    if metrics["f1"] > best_f1:
        best_f1 = metrics["f1"]
        best_name = name
        joblib.dump(pipe, MODEL_DIR / "best_model.joblib")

metadata = {"best_model": best_name, "features": FEATURES, "target": "Churn", "classes": [0, 1]}
(MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
(MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

print(f"\nBest model by F1: {best_name}")
print(f"Saved: {MODEL_DIR / 'best_model.joblib'}")
