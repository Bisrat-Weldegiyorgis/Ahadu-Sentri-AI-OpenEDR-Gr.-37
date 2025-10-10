import json
import os
import time
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.append(BACKEND_DIR)

from log_utils import read_all, update_record, get_log_path

st.set_page_config(page_title="Ahadu Sentri-AI Dashboard", layout="wide")

st.title("Ahadu Sentri-AI: OpenEDR Automation System")

# Sidebar controls
refresh_secs = st.sidebar.slider("Auto-refresh (seconds)", min_value=2, max_value=30, value=5, step=1)
log_path = st.sidebar.text_input("Log file path", get_log_path())

# Tabs
live_tab, history_tab, stats_tab = st.tabs(["Live Feed", "History", "Analytics"])


def load_records() -> List[Dict]:
    try:
        return read_all(path=log_path)
    except Exception:
        return []


with live_tab:
    st.subheader("Live Detections")
    records = load_records()
    if not records:
        st.info("No detections yet. Start the backend engine to stream events.")
    else:
        # Show newest first
        records = sorted(records, key=lambda r: r.get("timestamp", ""), reverse=True)

        for rec in records[:50]:
            with st.container(border=True):
                cols = st.columns([4, 2, 2, 3])
                with cols[0]:
                    st.write(f"ID: `{rec.get('id')}`")
                    st.write(f"Time: {rec.get('timestamp')}")
                    st.write(f"Host: {rec.get('event',{}).get('host','-')}")
                    st.write(f"Process: {rec.get('event',{}).get('process_name','-')}")
                    st.caption(rec.get("event",{}).get("command_line",""))
                with cols[1]:
                    st.metric("Prediction", rec.get("predicted_label", "-"))
                    st.progress(min(1.0, max(0.0, float(rec.get("confidence_score", 0.0)))), text="Confidence")
                with cols[2]:
                    st.write("Suggested Action:")
                    st.code(rec.get("suggested_action", "-"))
                    st.write("User Override:")
                    st.code(rec.get("user_override_action", "-"))
                with cols[3]:
                    st.write("Set Action:")
                    c1, c2, c3 = st.columns(3)
                    record_id = rec.get("id")
                    if c1.button("ALLOW", key=f"allow-{record_id}"):
                        update_record(record_id, {"user_override_action": "ALLOW"}, path=log_path)
                        st.experimental_rerun()
                    if c2.button("QUARANTINE", key=f"quarantine-{record_id}"):
                        update_record(record_id, {"user_override_action": "QUARANTINE"}, path=log_path)
                        st.experimental_rerun()
                    if c3.button("BLOCK", key=f"block-{record_id}"):
                        update_record(record_id, {"user_override_action": "BLOCK"}, path=log_path)
                        st.experimental_rerun()

with history_tab:
    st.subheader("Detection History")
    records = load_records()
    if not records:
        st.info("No history available yet.")
    else:
        df = pd.DataFrame(records)
        # Flatten some nested fields
        if "event" in df.columns:
            event_df = pd.json_normalize(df["event"]).add_prefix("event.")
            df = pd.concat([df.drop(columns=["event"]), event_df], axis=1)
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

with stats_tab:
    st.subheader("Analytics")
    records = load_records()
    if not records:
        st.info("No stats yet.")
    else:
        df = pd.DataFrame(records)
        # Label counts
        if "predicted_label" in df.columns:
            counts = df["predicted_label"].value_counts().reset_index()
            counts.columns = ["label", "count"]
            st.bar_chart(data=counts, x="label", y="count")
        # Confidence distribution
        if "confidence_score" in df.columns:
            st.line_chart(df["confidence_score"]) 

# Auto refresh
st.sidebar.caption("Dashboard refreshes automatically.")
st_autorefresh = st.sidebar.empty()
if refresh_secs:
    st_autorefresh.write(f"Auto-refresh every {refresh_secs} sec")
    st.runtime.legacy_caching.clear_cache()
    time.sleep(0.01)
    st.experimental_rerun()
