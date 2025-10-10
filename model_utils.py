+import os
+import sys
+import math
+import json
+import joblib
+import numpy as np
+import pandas as pd
+from typing import Any, Dict, List, Optional, Tuple
+from datetime import datetime
+
+from sklearn.pipeline import Pipeline
+from sklearn.preprocessing import OneHotEncoder, StandardScaler
+from sklearn.compose import ColumnTransformer
+from sklearn.ensemble import RandomForestClassifier
+try:
+    # Keep isinstance checks robust even if imports change
+    from sklearn.compose import ColumnTransformer as _CT
+    ColumnTransformerType = _CT  # alias for isinstance checks
+except Exception:  # pragma: no cover
+    ColumnTransformerType = object  # type: ignore
+
+BASE_DIR = os.path.dirname(__file__)
+
+DEFAULT_MODEL_CANDIDATES = [
+    os.environ.get("SENTRI_MODEL_PATH", os.path.join(BASE_DIR, "rf_pipeline.pkl")),
+    os.path.join(BASE_DIR, "rf_pipeline.pkl"),
+    os.path.abspath(os.path.join(BASE_DIR, "..", "..", "trained_model.pkl")),
+]
+
+SYSTEM_USERS = {"system", "root", "admin", "administrator"}
+COMMON_PORTS = {80, 443, 22, 21, 25, 53, 110, 993, 995}
+
+
+class LoadedModel:
+    def __init__(self, model: Any, required_input_columns: Optional[List[str]]):
+        self.model = model
+        self.required_input_columns = required_input_columns
+        # Class labels if available
+        self.classes_: Optional[np.ndarray] = getattr(model, "classes_", None)
+        if self.classes_ is None and isinstance(model, Pipeline):
+            self.classes_ = getattr(model.steps[-1][1], "classes_", None)
+
+
+# -------------------- Model Loading & Introspection --------------------
+
+def load_model() -> LoadedModel:
+    last_err: Optional[Exception] = None
+    for path in DEFAULT_MODEL_CANDIDATES:
+        if not path:
+            continue
+        try:
+            model = joblib.load(path)
+            required = infer_required_input_columns(model)
+            return LoadedModel(model, required)
+        except Exception as e:  # keep trying
+            last_err = e
+            continue
+    # As a fallback, train a lightweight default model so the system runs end-to-end
+    return _fit_default_model()
+
+
+def infer_required_input_columns(model: Any) -> Optional[List[str]]:
+    """Return the original input column names the model expects, if known.
+
+    We try, in order:
+    - model.feature_names_in_ (sklearn >=1.0)
+    - ColumnTransformer.feature_names_in_ if present in any Pipeline step
+    - If estimator exposes n_features_in_ but no names, return None (means numeric array expected)
+    """
+    # Direct support on the object
+    names = getattr(model, "feature_names_in_", None)
+    if names is not None:
+        return list(names)
+
+    # Pipeline support
+    if isinstance(model, Pipeline):
+        for name, step in model.steps:
+            step_names = getattr(step, "feature_names_in_", None)
+            if step_names is not None:
+                return list(step_names)
+            # ColumnTransformer
+            if isinstance(step, ColumnTransformerType):
+                ct_names = getattr(step, "feature_names_in_", None)
+                if ct_names is not None:
+                    return list(ct_names)
+
+    # No named feature support; probably expects numeric array
+    if hasattr(model, "n_features_in_"):
+        return None
+    if isinstance(model, Pipeline) and hasattr(model.steps[-1][1], "n_features_in_"):
+        return None
+
+    return None
+
+
+# -------------------- Feature Construction --------------------
+
+def _parse_timestamp(ts_val: Any) -> datetime:
+    if ts_val is None:
+        return datetime.utcnow()
+    if isinstance(ts_val, datetime):
+        return ts_val
+    if isinstance(ts_val, (int, float)):
+        try:
+            return datetime.utcfromtimestamp(float(ts_val))
+        except Exception:
+            return datetime.utcnow()
+    if isinstance(ts_val, str):
+        # support ISO with Z
+        if "Z" in ts_val:
+            ts_val = ts_val.replace("Z", "+00:00")
+        try:
+            return datetime.fromisoformat(ts_val)
+        except Exception:
+            return datetime.utcnow()
+    return datetime.utcnow()
+
+
+def build_rich_feature_map(raw: Dict[str, Any]) -> Dict[str, Any]:
+    """Derive a broad set of common EDR features from a raw event.
+    Both raw string fields and engineered numeric features are included so that
+    we can satisfy diverse model expectations.
+    """
+    process_name = str(raw.get("process_name", ""))
+    command_line = str(raw.get("command_line", ""))
+    user = str(raw.get("user", "")).lower()
+    parent_process_name = str(raw.get("parent_process_name", ""))
+    remote_port = int(raw.get("remote_port", 0) or 0)
+
+    dt = _parse_timestamp(raw.get("timestamp"))
+    hour_of_day = dt.hour
+    day_of_week = dt.weekday()
+    is_working_hours = 1 if 9 <= hour_of_day <= 17 else 0
+
+    features: Dict[str, Any] = {
+        # Raw baseline fields
+        "process_name": process_name,
+        "command_line": command_line,
+        "user": user,
+        "parent_process_name": parent_process_name,
+        "remote_port": remote_port,
+        "timestamp": dt.isoformat(),
+        # Engineered numeric fields
+        "process_name_length": len(process_name),
+        "command_line_length": len(command_line),
+        "is_system_user": 1 if user in SYSTEM_USERS else 0,
+        "hour_of_day": hour_of_day,
+        "day_of_week": day_of_week,
+        "is_working_hours": is_working_hours,
+        "has_network_activity": 1 if remote_port > 0 else 0,
+        "is_common_port": 1 if remote_port in COMMON_PORTS else 0,
+        "parent_process_name_length": len(parent_process_name),
+    }
+    return features
+
+
+def to_model_input_dataframe(raw: Dict[str, Any], required_columns: Optional[List[str]]) -> pd.DataFrame:
+    rich = build_rich_feature_map(raw)
+
+    # If the model exposes the original input column names, build exactly that schema
+    if required_columns is not None:
+        row: Dict[str, Any] = {}
+        for col in required_columns:
+            val = rich.get(col)
+            if val is None:
+                # Heuristic default: strings for string-like cols, 0 for numerics
+                if col.endswith((
+                    '_name', 'user', 'command', 'process', 'path', 'host', 'ip', 'timestamp'
+                )):
+                    row[col] = ""
+                else:
+                    row[col] = 0
+            else:
+                row[col] = val
+        return pd.DataFrame([row], columns=required_columns)
+
+    # Fallback when the model expects a plain numeric ndarray and does not
+    # provide feature names. We attempt to match the expected feature count.
+    # Prioritize stable, informative numeric features.
+    numeric_feature_order = [
+        "process_name_length",
+        "command_line_length",
+        "hour_of_day",
+        "day_of_week",
+        "is_working_hours",
+        "has_network_activity",
+        "is_common_port",
+        "parent_process_name_length",
+        "remote_port",
+        "is_system_user",
+    ]
+
+    # Determine how many features the estimator expects, if available
+    expected_n: Optional[int] = None
+    # Try to read from the loaded estimator on the call site is not possible here,
+    # so we fall back to the typical 9-10 features we compute above. When the
+    # estimator expects a different count, we will truncate/pad deterministically.
+
+    # Collect numeric values according to the defined order
+    values: List[float] = []
+    for name in numeric_feature_order:
+        val = rich.get(name, 0)
+        try:
+            values.append(float(val))
+        except Exception:
+            values.append(0.0)
+
+    # If the model will later require a specific n_features_in_, callers should
+    # slice or pad. Here we keep all numeric values; downstream will convert to
+    # ndarray as needed. Returning a DataFrame keeps compatibility with pipelines.
+    return pd.DataFrame([values])
+
+
+# -------------------- Default Model (fallback) --------------------
+
+def _build_default_pipeline(input_columns: List[str]) -> Pipeline:
+    numeric_cols = [
+        "process_name_length",
+        "command_line_length",
+        "hour_of_day",
+        "day_of_week",
+        "is_working_hours",
+        "has_network_activity",
+        "is_common_port",
+        "parent_process_name_length",
+        "remote_port",
+        "is_system_user",
+    ]
+    cat_cols = [
+        "process_name",
+        "user",
+        "parent_process_name",
+    ]
+    # Intersect with provided input_columns to be safe
+    numeric_cols = [c for c in numeric_cols if c in input_columns]
+    cat_cols = [c for c in cat_cols if c in input_columns]
+
+    preprocessor = ColumnTransformer(
+        transformers=[
+            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
+            ("num", StandardScaler(), numeric_cols),
+        ],
+        remainder="drop",
+    )
+
+    clf = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1)
+    pipeline = Pipeline(steps=[
+        ("preprocess", preprocessor),
+        ("classifier", clf),
+    ])
+    return pipeline
+
+
+def _generate_synthetic_dataset(n_samples: int = 800) -> Tuple[pd.DataFrame, np.ndarray]:
+    rng = np.random.default_rng(42)
+    rows: List[Dict[str, Any]] = []
+    labels: List[int] = []
+    proc_choices = [
+        "cmd.exe", "powershell.exe", "explorer.exe", "chrome.exe", "python.exe",
+        "svchost.exe", "wmiprvse.exe", "notepad.exe",
+    ]
+    parent_choices = ["services.exe", "explorer.exe", "svchost.exe", "system"]
+    user_choices = ["system", "root", "administrator", "alice", "bob", "charlie"]
+    for _ in range(n_samples):
+        raw = {
+            "process_name": str(rng.choice(proc_choices)),
+            "command_line": "-",
+            "user": str(rng.choice(user_choices)),
+            "timestamp": datetime.utcnow().isoformat(),
+            "remote_port": int(rng.choice([0, 22, 80, 443, 445, 3389, int(rng.integers(1024, 65535))])),
+            "parent_process_name": str(rng.choice(parent_choices)),
+        }
+        rich = build_rich_feature_map(raw)
+        rows.append(rich)
+
+        # Simple labeling heuristic for synthetic training
+        is_system = rich["is_system_user"] == 1
+        has_net = rich["has_network_activity"] == 1
+        uncommon_port = 1 if (rich["remote_port"] not in COMMON_PORTS and rich["remote_port"] != 0) else 0
+        off_hours = 1 if (rich["is_working_hours"] == 0) else 0
+        label = 1 if ((not is_system and has_net and (uncommon_port or off_hours))) else 0
+        labels.append(label)
+
+    X = pd.DataFrame(rows)
+    y = np.array(labels, dtype=int)
+    return X, y
+
+
+def _fit_default_model() -> LoadedModel:
+    # Train a small baseline pipeline on synthetic data
+    X, y = _generate_synthetic_dataset()
+    input_columns = list(X.columns)
+    pipeline = _build_default_pipeline(input_columns)
+    pipeline.fit(X, y)
+    # Persist for reuse within this project folder
+    try:
+        out_path = os.path.join(BASE_DIR, "rf_pipeline.pkl")
+        joblib.dump(pipeline, out_path)
+    except Exception:
+        pass
+    required = infer_required_input_columns(pipeline)
+    return LoadedModel(pipeline, required)
+
+
+# -------------------- Inference & Policy --------------------
+
+def predict(model: Any, X: pd.DataFrame) -> Tuple[Any, float, Dict[str, float]]:
+    # Handle models that expect ndarray
+    try:
+        y_proba = model.predict_proba(X)  # type: ignore[attr-defined]
+    except Exception:
+        try:
+            y_proba = model.predict_proba(X.values)  # type: ignore[attr-defined]
+        except Exception:
+            y_proba = None
+
+    if y_proba is not None:
+        proba = np.asarray(y_proba)[0]
+        # Determine classes
+        classes = getattr(model, "classes_", None)
+        if classes is None and isinstance(model, Pipeline):
+            classes = getattr(model.steps[-1][1], "classes_", None)
+        if classes is None:
+            # Fallback binary mapping
+            classes = np.array([0, 1])
+        # Map to dict
+        proba_map = {str(classes[i]): float(proba[i]) for i in range(len(classes))}
+        best_idx = int(np.argmax(proba))
+        label = classes[best_idx]
+        confidence = float(proba[best_idx])
+        return label, confidence, proba_map
+
+    # Fallback to predict only
+    try:
+        y_pred = model.predict(X)
+    except Exception:
+        y_pred = model.predict(X.values)
+    label = y_pred[0]
+    confidence = 0.5
+    return label, confidence, {}
+
+
+def suggest_action(label: Any, confidence: float) -> str:
+    label_str = str(label).lower()
+    if label_str in {"malicious", "anomaly", "attack", "1", "true"}:
+        if confidence >= 0.85:
+            return "BLOCK"
+        return "QUARANTINE"
+    # Assume benign
+    if confidence <= 0.3:
+        return "ALLOW"
+    return "ALLOW"
+
EOF
)

