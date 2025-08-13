from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd

from .utils import scan_runs_flat


def render_models_tab():
    st.subheader("Models – Featured")
    runs = scan_runs_flat("experiments")
    if not runs:
        st.info("No experiments found. Train a model or check the Pipelines tab.")
        return

    # Filters
    ds_values = sorted({r.get("dataset", "") for r in runs})
    pl_values = sorted({r.get("pipeline", "") for r in runs})
    ds_filter = st.selectbox("Dataset", options=["All"] + ds_values, index=0)
    pl_filter = st.selectbox("Pipeline", options=["All"] + pl_values, index=0)

    filtered = []
    for r in runs:
        if ds_filter != "All" and r.get("dataset") != ds_filter:
            continue
        if pl_filter != "All" and r.get("pipeline") != pl_filter:
            continue
        # Use last stage metrics as summary
        stages = r.get("stages", [])
        last = stages[-1] if stages else {}
        metrics = last.get("metrics") or {}
        filtered.append({
            "dataset": r.get("dataset"),
            "pipeline": r.get("pipeline"),
            "backbone": r.get("backbone"),
            "run_dir": r.get("path"),
            "roc": last.get("roc"),
            "weights": last.get("weights"),
            "metrics": metrics,
        })

    if not filtered:
        st.info("No runs found for selection.")
        return

    # sort by Test EER ascending (best first), fallback to Dev EER
    def score_key(c):
        m = c["metrics"]
        te = m.get("test_eer")
        de = m.get("dev_eer")
        return (te if te is not None else (de if de is not None else 1.0))

    filtered.sort(key=score_key)
    # show as cards in grid
    cols = st.columns(3)
    for i, c in enumerate(filtered[:12]):
        with cols[i % 3]:
            st.markdown(f"**{c['dataset']} · {c['pipeline']} · {c['backbone']}**")
            m = c.get("metrics")
            st.caption(f"Dev EER: {m.get('dev_eer'):.4f} | Test EER: {m.get('test_eer'):.4f} | AUC: {m.get('test_auc'):.4f}" if m else "Metrics N/A")
            if c.get("roc"):
                st.image(c["roc"], use_container_width=True)
            key_open = f"open_{i}"
            key_verify = f"verify_{i}"
            b1, b2 = st.columns([1, 1])
            with b1:
                if st.button("Open", key=key_open):
                    st.session_state["lib_open_path"] = str(c["run_dir"])  # reuse Library detail
                    st.info("Open selected. Click the 'Library' tab above to view details.")
            with b2:
                if st.button("Verify", key=key_verify):
                    st.session_state["v_weights"] = c.get("weights") or ""
                    st.info("Verify selected. Click the 'Verify (Cosine)' tab above.")
