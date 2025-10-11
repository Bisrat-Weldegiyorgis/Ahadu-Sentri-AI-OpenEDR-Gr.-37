import json
import os
import time
from typing import Dict, List, Optional
from uuid import uuid4
from filelock import FileLock, Timeout

BASE_DIR = os.path.dirname(__file__)
DEFAULT_LOG_PATH = os.environ.get("SENTRI_LOG_PATH", os.path.join(BASE_DIR, "detections_log.json"))
LOCK_TIMEOUT_SECONDS = int(os.environ.get("SENTRI_LOCK_TIMEOUT", "5"))


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def ensure_log_file(path: Optional[str] = None) -> str:
    log_path = path or DEFAULT_LOG_PATH
    _ensure_parent_dir(log_path)
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump([], f)
    return log_path


def _read_json_safely(path: str) -> List[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if isinstance(data, list):
                return data
            # If file accidentally contains JSON Lines, parse line by line
            lines = [json.loads(line) for line in content.splitlines() if line.strip()]
            return lines
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        # Corrupt file; back it up and start fresh
        backup_path = path + f".bak-{int(time.time())}"
        try:
            os.replace(path, backup_path)
        except Exception:
            pass
        return []


def _write_json_safely(path: str, records: List[Dict]) -> None:
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    os.replace(temp_path, path)


def read_all(path: Optional[str] = None) -> List[Dict]:
    log_path = ensure_log_file(path)
    lock = FileLock(log_path + ".lock")
    try:
        with lock.acquire(timeout=LOCK_TIMEOUT_SECONDS):
            return _read_json_safely(log_path)
    except Timeout:
        # Best-effort fallback without lock
        return _read_json_safely(log_path)


def append_record(record: Dict, path: Optional[str] = None) -> Dict:
    log_path = ensure_log_file(path)
    lock = FileLock(log_path + ".lock")
    if "id" not in record:
        record["id"] = str(uuid4())
    if "timestamp" not in record:
        record["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        with lock.acquire(timeout=LOCK_TIMEOUT_SECONDS):
            records = _read_json_safely(log_path)
            records.append(record)
            _write_json_safely(log_path, records)
    except Timeout:
        # Last resort: try without lock (risk of race)
        records = _read_json_safely(log_path)
        records.append(record)
        _write_json_safely(log_path, records)
    return record


def update_record(record_id: str, updates: Dict, path: Optional[str] = None) -> bool:
    log_path = ensure_log_file(path)
    lock = FileLock(log_path + ".lock")
    try:
        with lock.acquire(timeout=LOCK_TIMEOUT_SECONDS):
            records = _read_json_safely(log_path)
            updated = False
            for rec in records:
                if rec.get("id") == record_id:
                    rec.update(updates)
                    rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    updated = True
                    break
            if updated:
                _write_json_safely(log_path, records)
            return updated
    except Timeout:
        # Give up
        return False


def get_log_path() -> str:
    return DEFAULT_LOG_PATH
