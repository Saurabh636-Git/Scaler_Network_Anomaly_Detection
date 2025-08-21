import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = Path("Network_anomaly_data.csv")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # Infer target
    label_col = None
    for c in df.columns:
        if c in {"attack", "label", "target"}:
            label_col = c
            break
    if label_col is None:
        raise SystemExit("Could not locate a target column (attack/label/target). Update train_model.py.")

    y_raw = df[label_col]
    # Binary target: Normal vs Attack
    y = (y_raw.astype(str).str.lower() != "normal").astype(int)

    X_cols = [c for c in df.columns if c != label_col]
    X = df[X_cols].copy()

    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.number)]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced")
    pipe = Pipeline(steps=[("prep", preprocess), ("clf", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)

    dump(pipe, ARTIFACTS_DIR / "model.joblib")

    # Save schema for the API
    classes = {0: "Normal", 1: "Attack"}
    schema = {
        "features": X_cols,
        "target": label_col,
        "classes": {str(k): v for k, v in classes.items()}
    }
    (ARTIFACTS_DIR / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    print("Saved artifacts to:", ARTIFACTS_DIR.resolve())


if __name__ == "__main__":
    main()


