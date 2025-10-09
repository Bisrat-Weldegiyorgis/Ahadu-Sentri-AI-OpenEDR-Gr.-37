# Ahadu Sentri-AI: OpenEDR Automation System

An open-source prototype EDR pipeline that streams endpoint events, extracts
features, runs a trained ML model for detection, and lets analysts take actions
via a Streamlit dashboard.

## Folder Structure

```
ahadu_sentri_ai/
├── backend/
│   ├── detection_engine.py
│   ├── model_utils.py
│   ├── log_utils.py
│   ├── rf_pipeline.pkl           # Place your trained pipeline here (from Colab)
│   ├── detections_log.json       # Auto-created on first run
│   └── requirements.txt
├── dashboard/
│   └── dashboard.py
├── data/
│   └── sample_events.json
└── README.md
```

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
cd ahadu_sentri_ai/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Model Setup

Export the trained RandomForest pipeline from Google Colab as `rf_pipeline.pkl`.
Make sure the pipeline contains its preprocessing (e.g., `ColumnTransformer`,
encoders) so the same steps run at inference.

Copy it into `ahadu_sentri_ai/backend/rf_pipeline.pkl`.

Alternatively, you can point the engine to a model path:

```bash
export SENTRI_MODEL_PATH=/absolute/path/to/rf_pipeline.pkl
```

## Run the Detection Engine

Run with random simulated events:

```bash
python ahadu_sentri_ai/backend/detection_engine.py --sleep 1.5
```

Run with a sample events file and generate 10 records:

```bash
python ahadu_sentri_ai/backend/detection_engine.py \
  --source file \
  --source-file ahadu_sentri_ai/data/sample_events.json \
  --num-events 10 --sleep 0
```

The engine writes records to `ahadu_sentri_ai/backend/detections_log.json`.

## Run the Streamlit Dashboard

In a separate terminal (same virtualenv), run:

```bash
streamlit run ahadu_sentri_ai/dashboard/dashboard.py
```

The dashboard will read from the same log file and show a live feed.
Use the buttons (ALLOW / QUARANTINE / BLOCK) to override actions; these updates
are written back to the JSON log.

## Feature Mismatch Handling

If you see an error like:

- "X has 45 features, ColumnTransformer expects 42"

Place the exact training pipeline (including its preprocessors) into
`rf_pipeline.pkl`. The backend introspects `feature_names_in_` to construct the
precise input DataFrame in the correct order. If names are not available, it
falls back to a robust numeric subset. Ensure your training pipeline exposes
input names (preferred) or accepts numeric arrays accordingly.

## Notes

- The log file is guarded by a file lock to avoid corruption when both backend
  and dashboard access it.
- You can change the log file path via env var `SENTRI_LOG_PATH` or dashboard
  sidebar.
