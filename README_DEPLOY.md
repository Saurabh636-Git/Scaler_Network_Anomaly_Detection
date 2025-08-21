## Flask Deployment (Block 4)

### 1) Train and export artifacts
```
python train_model.py
```
This creates `artifacts/model.joblib` and `artifacts/schema.json`.

### 2) Install API dependencies (prefer a venv)
```
python -m venv comp_new
comp_new\Scripts\activate
pip install -r api/requirements.txt
```

### 3) Run the API
```
python api/app.py
```
The service listens on http://localhost:8000

### 4) Test endpoints

Health:
```
GET http://localhost:8000/health
```

Model info:
```
GET http://localhost:8000/version
```

Predict (single record):
```
POST http://localhost:8000/predict
Content-Type: application/json

{
  "duration": 0,
  "protocol_type": "tcp",
  "service": "http",
  "flag": "SF",
  "src_bytes": 181,
  "dst_bytes": 5450
}
```

Predict (batch):
```
POST http://localhost:8000/predict
Content-Type: application/json

[
  {"duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF", "src_bytes": 181, "dst_bytes": 5450},
  {"duration": 2, "protocol_type": "udp", "service": "domain_u", "flag": "S0", "src_bytes": 0, "dst_bytes": 0}
]
```

Notes:
- The API auto-fills any missing expected columns as null; the pipeline imputers will handle them.
- For production, run behind a WSGI server (e.g., gunicorn/uvicorn) and add input validation/auth.


