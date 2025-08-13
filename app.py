import os
import json
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional

import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from handnet import (
    find_device,
    set_seed,
    ensure_min_images_per_subject,
    filter_by_aspect,
    filter_clean,
    stratified_subject_split,
    build_image_index,
    HandDataset,
    make_transforms,
    make_backbone,
    LinearClassifier,
    train_one_epoch,
    evaluate,
    evaluate_and_save,
)


st.set_page_config(page_title="HandNet Trainer", layout="wide")
st.title("Hand Verification – HandNet UI")

tab_models, tab_library, tab_train, tab_verify = st.tabs([
    "Models",
    "Library",
    "Train / Eval",
    "Verify (Cosine)",
])


def run_training_or_eval(
    data_dir: str,
    csv_path: str,
    aspect: str,
    clean_only: bool,
    min_images: int,
    img_size: int,
    batch_size: int,
    epochs: int,
    seed: int,
    backbone: str,
    loss_name: str,
    device_arg: str,
    out_dir: str,
    eval_only: bool,
    weights_path: str = "",
    epoch_callback: Optional[Callable[[int, float, float], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
):
    set_seed(seed)
    device = find_device(device_arg)
    st.write(f"Using device: {device}")

    df = pd.read_csv(csv_path)
    df = filter_by_aspect(df, aspect)
    if clean_only:
        df = filter_clean(df)
    df = ensure_min_images_per_subject(df, min_images)

    img_index = build_image_index(data_dir)
    st.write(f"Indexed {len(img_index)} images under {data_dir}")
    st.write(f"Rows after filtering: {len(df)}; Subjects: {df['id'].nunique()}")

    subjects = sorted(df['id'].unique().tolist())
    train_sub, dev_sub, test_sub = stratified_subject_split(subjects, ratios=(0.7, 0.15, 0.15), seed=seed)
    df_train = df[df['id'].isin(train_sub)].copy()
    df_dev = df[df['id'].isin(dev_sub)].copy()
    df_test = df[df['id'].isin(test_sub)].copy()
    st.write(f"Split → train/dev/test subjects: {len(train_sub)}/{len(dev_sub)}/{len(test_sub)}")

    train_tf, eval_tf = make_transforms(img_size, aug=True)

    id_to_idx = {sid: i for i, sid in enumerate(sorted(df_train['id'].unique()))}

    df_train = df_train
    df_dev = df_dev
    df_test = df_test

    train_ds = HandDataset(df_train[["imageName", "id"]], img_index, transform=train_tf)
    dev_ds = HandDataset(df_dev[["imageName", "id"]], img_index, transform=eval_tf)
    test_ds = HandDataset(df_test[["imageName", "id"]], img_index, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    embed_dim = 128
    model = make_backbone(backbone, embed_dim=embed_dim).to(device)

    clf = None
    if loss_name == "ce":
        clf = LinearClassifier(embed_dim, num_classes=len(id_to_idx)).to(device)
        params = list(model.parameters()) + list(clf.parameters())
    else:
        params = list(model.parameters())

    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    run_dir = Path(out_dir) / f"{backbone}_{loss_name}_{aspect}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if eval_only:
        assert weights_path, "Weights path is required for evaluation."
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state['model'])
        if 'clf' in state and clf is not None and state['clf'] is not None:
            clf.load_state_dict(state['clf'])
        evaluate_and_save(model, dev_loader, test_loader, run_dir)
        st.success(f"Evaluation complete. Artifacts saved to: {run_dir}")
        return

    best_eer = 1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        loss_val = train_one_epoch(model, clf, train_loader, opt, device, loss_type=loss_name)
        dev_eer = evaluate(model, dev_loader)
        if epoch_callback:
            epoch_callback(epoch, float(loss_val), float(dev_eer))
        else:
            st.write(f"Epoch {epoch:02d} | loss={loss_val:.4f} | dev EER={dev_eer*100:.2f}%")
        # early stop trigger from UI
        if should_stop and should_stop():
            st.warning("Training stopped by user.")
            break
        if dev_eer < best_eer:
            best_eer = dev_eer
            best_state = {
                'model': model.state_dict(),
                'clf': clf.state_dict() if clf is not None else None,
                'epoch': epoch,
                'dev_eer': dev_eer,
            }
            torch.save(best_state, run_dir / "best.pt")

    if best_state is None:
        st.warning("No improvement recorded; saving last state.")
        best_state = {
            'model': model.state_dict(),
            'clf': clf.state_dict() if clf is not None else None,
            'epoch': epochs,
            'dev_eer': float('nan'),
        }
        torch.save(best_state, run_dir / "best.pt")
    else:
        model.load_state_dict(best_state['model'])

    evaluate_and_save(model, dev_loader, test_loader, run_dir)
    st.success(f"Done. Artifacts in: {run_dir}")


if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'stop_training' not in st.session_state:
    st.session_state.stop_training = False

with tab_train:
    st.subheader("Train / Evaluate (Advanced)")
    disabled = st.session_state.is_training
    with st.expander("Train Settings", expanded=True):
        data_dir = st.text_input("Data directory", value="data/Hands", key="t_data", disabled=disabled)
        csv_path = st.text_input("CSV path", value="data/HandInfo.csv", key="t_csv", disabled=disabled)
        cols = st.columns(3)
        with cols[0]:
            aspect = st.selectbox("Aspect", options=["palmar", "dorsal", "any"], index=0, key="t_aspect", disabled=disabled)
            clean_only = st.checkbox("Clean-only", value=False, key="t_clean", disabled=disabled)
            min_images = st.number_input("Min images/subject", 1, 20, 3, key="t_minimg", disabled=disabled)
        with cols[1]:
            img_size = st.number_input("Image size", 64, 1024, 224, step=16, key="t_img", disabled=disabled)
            batch_size = st.number_input("Batch size", 1, 1024, 64, step=1, key="t_bs", disabled=disabled)
            epochs = st.number_input("Epochs", 1, 200, 15, key="t_epochs", disabled=disabled)
        with cols[2]:
            seed = st.number_input("Seed", 0, 999999, 42, key="t_seed", disabled=disabled)
            backbone = st.selectbox("Backbone", options=["resnet18", "handnet"], index=0, key="t_bb", disabled=disabled)
            loss_name = st.selectbox("Loss", options=["ce", "triplet"], index=0, key="t_loss", disabled=disabled)
        device_arg = st.selectbox("Device", options=["auto", "cpu", "cuda", "mps"], index=0, key="t_dev", disabled=disabled)
        out_dir = st.text_input("Output dir", value="runs", key="t_out", disabled=disabled)

    ph_status = st.empty()
    ph_prog = st.progress(0) if st.session_state.is_training else st.empty()
    col_l, col_r = st.columns([3, 2])
    with col_l:
        ph_logs = st.empty()
        ph_chart_loss = st.empty()
        ph_chart_eer = st.empty()
    with col_r:
        if st.session_state.is_training:
            if st.button("Stop", key="btn_stop"):
                st.session_state.stop_training = True
        else:
            if st.button("Start Training", key="btn_train"):
                st.session_state.is_training = True
                st.session_state.stop_training = False
                st.session_state.loss_hist = []
                st.session_state.eer_hist = []

                def epoch_cb(ep: int, loss_v: float, eer_v: float):
                    st.session_state.loss_hist.append(loss_v)
                    st.session_state.eer_hist.append(eer_v)
                    ph_prog.progress(ep / max(1, int(epochs)))
                    ph_logs.text("\n".join([
                        f"Epoch {i+1:02d} | loss={l:.4f} | dev EER={e*100:.2f}%"
                        for i, (l, e) in enumerate(zip(st.session_state.loss_hist, st.session_state.eer_hist))
                    ]))
                    # charts
                    ph_chart_loss.line_chart(pd.DataFrame({"loss": st.session_state.loss_hist}))
                    ph_chart_eer.line_chart(pd.DataFrame({"dev_eer": st.session_state.eer_hist}))

                def should_stop():
                    return st.session_state.stop_training

                # Run training
                run_training_or_eval(
                    data_dir,
                    csv_path,
                    aspect,
                    clean_only,
                    int(min_images),
                    int(img_size),
                    int(batch_size),
                    int(epochs),
                    int(seed),
                    backbone,
                    loss_name,
                    device_arg,
                    out_dir,
                    eval_only=False,
                    epoch_callback=epoch_cb,
                    should_stop=should_stop,
                )
                st.session_state.is_training = False
                st.session_state.stop_training = False

# (Removed separate Eval tab; Evaluate is available within Train / Eval tab)
    st.subheader("Evaluate")

def scan_experiments(root: str) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    """Scan experiments directory into a nested mapping:
    {exp_name: {backbone: {stage: [run_dirs...]}}}
    """
    tree: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}
    root_path = Path(root)
    if not root_path.exists():
        return tree
    for exp_name in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        bb_map: Dict[str, Dict[str, List[Path]]] = {}
        for bb in sorted([p for p in exp_name.iterdir() if p.is_dir()]):
            stage_map: Dict[str, List[Path]] = {}
            for stage in sorted([p for p in bb.iterdir() if p.is_dir()]):
                runs = sorted([p for p in stage.iterdir() if p.is_dir()], reverse=True)
                if runs:
                    stage_map[stage.name] = runs
            if stage_map:
                bb_map[bb.name] = stage_map
        if bb_map:
            tree[exp_name.name] = bb_map
    return tree


def load_run_info(run_dir: Path) -> Dict[str, Any]:
    """Load metrics, args, and artifact paths for a run directory."""
    info: Dict[str, Any] = {"path": str(run_dir)}
    mfp = run_dir / "metrics.json"
    afp = run_dir / "args.json"
    rfp = run_dir / "roc.png"
    sfp = run_dir / "scores.csv"
    bfp = run_dir / "best.pt"
    if mfp.exists():
        try:
            info["metrics"] = json.loads(mfp.read_text())
        except Exception:
            info["metrics"] = None
    if afp.exists():
        try:
            info["args"] = json.loads(afp.read_text())
        except Exception:
            info["args"] = None
    info["roc_path"] = str(rfp) if rfp.exists() else ""
    info["scores_path"] = str(sfp) if sfp.exists() else ""
    info["weights_path"] = str(bfp) if bfp.exists() else ""
    return info
with tab_library:
    st.subheader("Library – All Models")
    root = st.text_input("Experiments root", value="experiments", key="lib_root")
    tree = scan_experiments(root)
    # Aggregate runs into a flat table
    rows = []
    for exp_name, bb_map in tree.items():
        for bb, stage_map in bb_map.items():
            for stage, runs in stage_map.items():
                for r in runs:
                    info = load_run_info(r)
                    metrics = info.get("metrics") or {}
                    rows.append({
                        "Experiment": exp_name,
                        "Backbone": bb,
                        "Stage": stage,
                        "Run": r.name,
                        "Dev EER": metrics.get("dev_eer"),
                        "Test EER": metrics.get("test_eer"),
                        "AUC": metrics.get("test_auc"),
                        "Path": str(r),
                    })
    if rows:
        df_runs = pd.DataFrame(rows)
        sort_col = st.selectbox("Sort by", options=["Test EER", "Dev EER", "AUC", "Run"], index=0)
        ascending = st.checkbox("Ascending", value=True)
        df_runs = df_runs.sort_values(by=sort_col, ascending=ascending)
        st.dataframe(df_runs, use_container_width=True)
        default_sel = st.session_state.get("lib_open_path", "(none)")
        options = ["(none)"] + df_runs["Path"].tolist()
        try:
            idx = options.index(default_sel)
        except ValueError:
            idx = 0
        sel = st.selectbox("Open run", options=options, index=idx)
        st.session_state["lib_open_path"] = sel
        if sel != "(none)":
            run_dir = Path(sel)
            info = load_run_info(run_dir)
            cols = st.columns(2)
            with cols[0]:
                st.write("Metrics")
                st.json(info.get("metrics", {}))
                st.write("Args")
                st.json(info.get("args", {}))
            with cols[1]:
                st.write("ROC Curve")
                if info.get("roc_path"):
                    st.image(info["roc_path"], use_container_width=True)
                # Score histogram
                sp = info.get("scores_path")
                if sp:
                    try:
                        sdf = pd.read_csv(sp)
                        st.write("Score distribution")
                        st.bar_chart(sdf["score"])
                    except Exception:
                        pass
            dl_cols = st.columns(3)
            with dl_cols[0]:
                if info.get("weights_path"):
                    with open(info["weights_path"], "rb") as f:
                        st.download_button("Download best.pt", f, file_name="best.pt")
            with dl_cols[1]:
                if info.get("scores_path"):
                    with open(info["scores_path"], "rb") as f:
                        st.download_button("Download scores.csv", f, file_name="scores.csv")
            with dl_cols[2]:
                if info.get("roc_path"):
                    with open(info["roc_path"], "rb") as f:
                        st.download_button("Download roc.png", f, file_name="roc.png")


def scan_experiments(root: str) -> Dict[str, Dict[str, Dict[str, List[Path]]]]:
    tree: Dict[str, Dict[str, Dict[str, List[Path]]]] = {}
    root_path = Path(root)
    if not root_path.exists():
        return tree
    for exp_name in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        bb_map: Dict[str, Dict[str, List[Path]]] = {}
        for bb in sorted([p for p in exp_name.iterdir() if p.is_dir()]):
            stage_map: Dict[str, List[Path]] = {}
            for stage in sorted([p for p in bb.iterdir() if p.is_dir()]):
                runs = sorted([p for p in stage.iterdir() if p.is_dir()], reverse=True)
                if runs:
                    stage_map[stage.name] = runs
            if stage_map:
                bb_map[bb.name] = stage_map
        if bb_map:
            tree[exp_name.name] = bb_map
    return tree




with tab_models:
    st.subheader("Models – Featured")
    root = st.text_input("Experiments root", value="experiments", key="feat_root")
    tree = scan_experiments(root)
    if not tree:
        st.info("No experiments found. Train a model or check the root path.")
    else:
        exp_opts = ["All"] + sorted(tree.keys())
        pick_exp = st.selectbox("Category", options=exp_opts)
        # Gather candidates
        candidates = []
        for exp_name, bb_map in tree.items():
            if pick_exp != "All" and exp_name != pick_exp:
                continue
            for bb, stage_map in bb_map.items():
                for stage, runs in stage_map.items():
                    for r in runs:
                        info = load_run_info(r)
                        metrics = info.get("metrics") or {}
                        candidates.append({
                            "exp": exp_name,
                            "bb": bb,
                            "stage": stage,
                            "run": r,
                            "metrics": metrics,
                            "roc": info.get("roc_path"),
                            "weights": info.get("weights_path"),
                        })
        if not candidates:
            st.info("No runs found for selection.")
        else:
            # sort by Test EER ascending (best first), fallback to Dev EER
            def score_key(c):
                m = c["metrics"]
                te = m.get("test_eer")
                de = m.get("dev_eer")
                return (te if te is not None else (de if de is not None else 1.0))

            candidates.sort(key=score_key)
            # show as cards in grid
            cols = st.columns(3)
            for i, c in enumerate(candidates[:12]):
                with cols[i % 3]:
                    st.markdown(f"**{c['exp']} · {c['bb']} · {c['stage']}**")
                    m = c["metrics"]
                    st.caption(f"Dev EER: {m.get('dev_eer'):.4f} | Test EER: {m.get('test_eer'):.4f} | AUC: {m.get('test_auc'):.4f}" if m else "Metrics N/A")
                    if c.get("roc"):
                        st.image(c["roc"], use_container_width=True)
                    key_open = f"open_{i}"
                    key_verify = f"verify_{i}"
                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("Open", key=key_open):
                            st.session_state["lib_open_path"] = str(c["run"])  # reuse Library detail
                            st.info("Open this run in Library tab to see full details.")
                    with b2:
                        if st.button("Verify", key=key_verify):
                            st.session_state["v_weights"] = c.get("weights") or ""
                            st.success("Verify tab prefilled with this model.")


def load_model_for_verify(weights_path: str, device: torch.device):
    """Load model from a .pt checkpoint, inferring backbone and img_size from saved args.

    Returns (model, args_dict).
    """
    assert weights_path, "Weights path is required"
    state = torch.load(weights_path, map_location=device)
    args = state.get('args', {}) or {}
    backbone = args.get('backbone', 'resnet18')
    model = make_backbone(backbone, embed_dim=128).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, args


def embed_images(model, pil_images: List[Image.Image], img_size: int, device: torch.device):
    _, eval_tf = make_transforms(img_size, aug=False)
    xs = [eval_tf(im.convert('RGB')) for im in pil_images]
    xb = torch.stack(xs, 0).to(device)
    with torch.no_grad():
        z = model(xb)
    return z


with tab_verify:
    st.subheader("Verify (cosine similarity)")
    v_weights = st.text_input("Model weights (.pt)", value=st.session_state.get("v_weights", ""), key="vwt")

    enroll_files = st.file_uploader("Enroll images (1-5)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    probe_file = st.file_uploader("Probe image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    if st.button("Compute Similarity"):
        if not v_weights:
            st.error("Please provide a model weights (.pt) file.")
        elif not enroll_files or probe_file is None:
            st.error("Please upload at least one enroll image and one probe image.")
        else:
            device = find_device("auto")
            model, ck_args = load_model_for_verify(v_weights, device)
            img_size = int(ck_args.get('img_size', 224))

            enroll_imgs = [Image.open(f).convert('RGB') for f in enroll_files]
            probe_img = Image.open(probe_file).convert('RGB')

            z_enroll = embed_images(model, enroll_imgs, img_size, device)
            z_probe = embed_images(model, [probe_img], img_size, device)

            z_mean = z_enroll.mean(0, keepdim=True)
            score = F.cosine_similarity(z_mean, z_probe).item()
            st.metric("Cosine similarity", f"{score:.4f}")
