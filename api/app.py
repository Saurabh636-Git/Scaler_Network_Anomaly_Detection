import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from joblib import load


ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
SCHEMA_PATH = ARTIFACTS_DIR / "schema.json"


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}. Train it first with train_model.py")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema artifact not found at {SCHEMA_PATH}. Train it first with train_model.py")
    model = load(MODEL_PATH)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return model, schema


def coerce_records_to_dataframe(
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    expected_columns: List[str]
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = data if isinstance(data, list) else [data]
    df = pd.DataFrame.from_records(records)
    # Add any missing expected columns as NaN
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan
    # Restrict to expected feature columns only
    df = df[expected_columns]
    return df


app = Flask(__name__)
model, schema = load_artifacts()


@app.get("/")
def index() -> Any:
    return jsonify({
        "message": "Network Anomaly Detection API",
        "endpoints": {
            "health": "/health",
            "version": "/version",
            "predict": "/predict (POST)"
        }
    })


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/version")
def version() -> Any:
    return jsonify({
        "model_path": str(MODEL_PATH),
        "schema_path": str(SCHEMA_PATH),
        "features": schema.get("features", []),
        "target": schema.get("target"),
        "classes": schema.get("classes"),
    })


@app.post("/predict")
def predict() -> Any:
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400

    if payload is None:
        return jsonify({"error": "Empty JSON payload"}), 400

    expected_features: List[str] = schema.get("features", [])
    if not expected_features:
        return jsonify({"error": "Model schema missing 'features'"}), 500

    try:
        X = coerce_records_to_dataframe(payload, expected_features)
    except Exception as e:
        return jsonify({"error": f"Failed to build input frame: {e}"}), 400

    # Predict class labels
    try:
        y_pred = model.predict(X)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    # Predict probabilities if available
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
    except Exception:
        proba = None

    classes = schema.get("classes")
    label_map = None
    if isinstance(classes, dict):
        # classes saved as mapping of str(int_label)->string_name
        label_map = {int(k): v for k, v in classes.items()}

    results: List[Dict[str, Any]] = []
    for i, pred in enumerate(y_pred):
        record: Dict[str, Any] = {"prediction": int(pred)}
        if label_map is not None and int(pred) in label_map:
            record["label"] = label_map[int(pred)]
        if proba is not None:
            # Return probability of positive class if binary, else full vector
            if proba.shape[1] == 2:
                record["probability_attack"] = float(proba[i, 1])
                record["probability_normal"] = float(proba[i, 0])
            else:
                record["probabilities"] = [float(p) for p in proba[i]]
        results.append(record)

    return jsonify({"results": results})


if __name__ == "__main__":
    # For local dev only; use a WSGI server for production
    app.run(host="0.0.0.0", port=8000, debug=True)


