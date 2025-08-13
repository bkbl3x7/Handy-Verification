from typing import Optional, Callable
import subprocess, sys, os
from pathlib import Path
from pathlib import Path

import streamlit as st
import pandas as pd
import torch
from torch.utils.data import DataLoader

from handnet import find_device
from .pipelines import _load_all_from, DATASETS_DIR, PIPELINES_DIR


def run_stage_with_cli(args_list, log_placeholder):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = args_list[:]
    if len(cmd) >= 1 and cmd[0] == sys.executable:
        cmd.insert(1, "-u")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(root),
        env=env,
    )
    logs = []
    if proc.stdout is not None:
        for line in proc.stdout:
            logs.append(line.rstrip())
            log_placeholder.text("\n".join(logs[-200:]))
    proc.wait()
    return proc.returncode


def render_train_eval_tab():
    if 'is_training' not in st.session_state:
        st.session_state.is_training = False
    if 'stop_training' not in st.session_state:
        st.session_state.stop_training = False

    st.subheader("Train / Evaluate (Advanced)")
    disabled = st.session_state.is_training
    datasets = _load_all_from(DATASETS_DIR)
    pipelines = _load_all_from(PIPELINES_DIR)
    with st.expander("Select Dataset and Pipeline", expanded=True):
        ds_name = st.selectbox("Dataset", options=list(datasets.keys()) or ["(none)"] , key="sel_ds", disabled=disabled)
        pl_name = st.selectbox("Pipeline", options=list(pipelines.keys()) or ["(none)"] , key="sel_pl", disabled=disabled)
        exp_root = "experiments"

    col_l, col_r = st.columns([2, 3])
    with col_r:
        st.write("Logs")
        ph_logs = st.empty()
    with col_l:
        if st.session_state.is_training:
            if st.button("Stop", key="btn_stop"):
                st.session_state.stop_training = True
        else:
            if st.button("Run Pipeline", key="btn_train"):
                st.session_state.is_training = True
                st.session_state.stop_training = False
                # Validate
                if not datasets or ds_name not in datasets or not pipelines or pl_name not in pipelines:
                    st.error("Please define a dataset and pipeline in the Pipelines tab.")
                else:
                    ds = datasets[ds_name]
                    pl = pipelines[pl_name]
                    from datetime import datetime
                    import re, json as _json
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    def slugify(s: str) -> str:
                        return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)
                    run_slug = f"{timestamp}_{slugify(ds_name)}_{slugify(pl_name)}"
                    base_dir = Path("experiments") / run_slug
                    base_dir.mkdir(parents=True, exist_ok=True)
                    manifest = {
                        # Keep names for UI convenience, but snapshot full configs to avoid name drift
                        "dataset": ds_name,
                        "pipeline": pl_name,
                        "dataset_cfg": ds,
                        "pipeline_cfg": pl,
                        "backbone": pl.get("backbone", "resnet18"),
                        "seed": int(pl.get("seed", 42)),
                        "created_at": timestamp,
                        "stages": []
                    }
                    (base_dir / "manifest.json").write_text(_json.dumps(manifest, indent=2))

                    init_from = ""
                    for stage in pl.get("stages", []):
                        args = [
                            sys.executable, "scripts/train_handnet.py",
                            "--data_dir", ds.get("data_dir", ""),
                            "--csv_path", ds.get("csv_path", ""),
                            "--aspect", str(ds.get("aspect", "palmar")),
                            "--img_size", str(ds.get("img_size", 224)),
                            "--min_images", str(ds.get("min_images", 3)),
                            "--backbone", pl.get("backbone", "resnet18"),
                            "--seed", str(pl.get("seed", 42)),
                            "--loss", stage.get("loss", "ce"),
                            "--epochs", str(stage.get("epochs", 10)),
                            "--exp_root", "experiments",
                            "--exp_name", ds_name,
                            "--pipeline", pl_name,
                            "--stage", stage.get("name", "stage"),
                            "--run_slug", run_slug,
                        ]
                        if ds.get("clean_only", False):
                            args.append("--clean_only")
                        bs = stage.get("batch_size")
                        if bs is not None:
                            args += ["--batch_size", str(bs)]
                        pk = stage.get("pk")
                        if pk:
                            args += ["--pk", str(pk)]
                        if init_from:
                            args += ["--init_from", init_from]

                        st.write(f"Running stage: {stage.get('name')}…")
                        rc = run_stage_with_cli(args, ph_logs)
                        if rc != 0:
                            st.error(f"Stage {stage.get('name')} failed with code {rc}")
                            break
                        # Update manifest with this stage's metrics and paths (flat files named after run_slug)
                        stage_name = stage.get("name")
                        metrics = {}
                        try:
                            with open(base_dir / f"{run_slug}.json", "r") as f:
                                metrics = _json.load(f)
                        except Exception:
                            pass
                        manifest["stages"].append({
                            "name": stage_name,
                            "loss": stage.get("loss"),
                            "epochs": stage.get("epochs"),
                            "batch_size": stage.get("batch_size"),
                            "pk": stage.get("pk"),
                            "dir": str(base_dir),
                            "metrics": metrics,
                            "weights": str(base_dir / f"{run_slug}.pt"),
                            "roc": str(base_dir / f"{run_slug}.png")
                        })
                        (base_dir / "manifest.json").write_text(_json.dumps(manifest, indent=2))
                        init_from = str(base_dir / f"{run_slug}.pt")
                    st.success("Pipeline completed.")
                st.session_state.is_training = False
                st.session_state.stop_training = False
    # Logs panel is created above; streaming output will appear there
