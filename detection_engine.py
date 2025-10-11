import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable
from uuid import uuid4

import numpy as np

# Local imports
BASE_DIR = os.path.dirname(__file__)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from log_utils import append_record, ensure_log_file, get_log_path
from model_utils import load_model, to_model_input_dataframe, suggest_action

PROCESS_NAMES = [
    "cmd.exe", "powershell.exe", "explorer.exe", "chrome.exe", "python.exe",
    "svchost.exe", "wmiprvse.exe", "notepad.exe", "winword.exe", "msiexec.exe"
]
PARENT_PROCESSES = [
    "services.exe", "explorer.exe", "system", "wininit.exe", "taskhostw.exe"
]
USERS = ["system", "root", "administrator", "alice", "bob", "charlie", "svc_account"]
COMMON_PORTS = [0, 80, 443, 22, 445, 3389, 53]


def generate_random_event() -> Dict[str, Any]:
    proc = random.choice(PROCESS_NAMES)
    parent = random.choice(PARENT_PROCESSES)
    user = random.choice(USERS)
    port = random.choice(COMMON_PORTS + [random.randint(1024, 65535) for _ in range(3)])

    base_time = datetime.utcnow() - timedelta(minutes=random.randint(0, 60))

    return {
        "process_name": proc,
        "command_line": f"{proc} /c echo test_{random.randint(1000,9999)}",
        "user": user,
        "timestamp": base_time.isoformat() + "Z",
        "remote_port": port,
        "parent_process_name": parent,
        "host": random.choice(["host-a", "host-b", "host-c"]),
    }


def iter_events_from_file(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("events", [])
    for item in data:
        yield item


def main() -> None:
    parser = argparse.ArgumentParser(description="Ahadu Sentri-AI detection engine")
    parser.add_argument("--model-path", default=None, help="Path to rf_pipeline.pkl (optional)")
    parser.add_argument("--log-path", default=None, help="Path to detections_log.json (optional)")
    parser.add_argument("--num-events", type=int, default=0, help="If >0, run for N events then exit")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between events")
    parser.add_argument("--source", default="random", choices=["random", "file"], help="Event source")
    parser.add_argument(
        "--source-file",
        default=os.path.join(os.path.dirname(BASE_DIR), "data", "sample_events.json"),
        help="Path to sample events JSON when --source=file",
    )

    args = parser.parse_args()

    if args.model_path:
        os.environ["SENTRI_MODEL_PATH"] = args.model_path

    loaded = load_model()

    log_path = ensure_log_file(args.log_path)
    print(f"[Engine] Using log file: {log_path}")

    if args.source == "file":
        event_iter = iter_events_from_file(args.source_file)
    else:
        def _gen():
            while True:
                yield generate_random_event()
        event_iter = _gen()

    produced = 0
    for raw_event in event_iter:
        X = to_model_input_dataframe(raw_event, loaded.required_input_columns)
        try:
            from model_utils import predict as model_predict
            label, confidence, proba_map = model_predict(loaded.model, X)
        except Exception as e:
            print(f"[Engine] Prediction failed: {e}")
            label, confidence, proba_map = "unknown", 0.0, {}

        action = suggest_action(label, confidence)

        record = {
            "id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": raw_event,
            "model_input": X.to_dict(orient="records")[0],
            "predicted_label": str(label),
            "confidence_score": float(confidence),
            "probabilities": proba_map,
            "suggested_action": action,
            "user_override_action": None,
        }
        append_record(record, path=log_path)
        print(
            f"[Engine] {record['id']} label={record['predicted_label']} "
            f"conf={record['confidence_score']:.2f} action={record['suggested_action']}"
        )

        produced += 1
        if args.num_events and produced >= args.num_events:
            break
        time.sleep(max(0.0, args.sleep))


if __name__ == "__main__":
    main()
