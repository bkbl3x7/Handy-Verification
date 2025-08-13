from typing import Dict, Any, List
from pathlib import Path
import json
import yaml
import streamlit as st

# --------------------------------------------------------------------
# Paths & setup
# --------------------------------------------------------------------
DATASETS_DIR = Path("configs/datasets")
PIPELINES_DIR = Path("configs/pipelines")
OPTIONS_PATH = Path("configs/options.json")
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
PIPELINES_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Minimal helpers (same as your original)
# --------------------------------------------------------------------
def _load_all_from(dirpath: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in sorted(dirpath.glob("*.y*ml")):
        try:
            data = yaml.safe_load(p.read_text()) or {}
            out[p.stem] = data
        except Exception:
            continue
    return out

def _save_to(dirpath: Path, name: str, data: Dict[str, Any]):
    text = yaml.safe_dump(data, sort_keys=False)
    (dirpath / f"{name}.yaml").write_text(text)

# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
def render_pipelines_tab():
    st.subheader("Pipelines – Datasets and Recipes")
    st.caption("Create and edit datasets/pipelines with simple forms. Files are saved per item.")

    # Make primary buttons red; use for Delete.
    st.markdown(
        """
        <style>
          button[data-testid="baseButton-primary"]{
            background-color:#dc2626 !important; /* red-600 */
            border-color:#dc2626 !important;
            color:#ffffff !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Load UI options (inline—no extra helpers)
    if OPTIONS_PATH.exists():
        try:
            options = json.loads(OPTIONS_PATH.read_text())
        except Exception:
            options = {}
    else:
        options = {}
    backbones: List[str] = options.get("backbones", ["resnet18", "handnet"])
    losses: List[str] = options.get("losses", ["ce", "triplet"])
    aspects: List[str] = options.get("aspects", ["palmar", "dorsal", "any"])
    pk_presets: List[str] = options.get("pk_presets", ["16x4", "8x8", "32x2", ""])

    # One clear choice instead of two giant columns
    choice = st.radio("What would you like to work on?", ["Datasets", "Pipelines"], horizontal=True)

    # ============================ DATASETS =============================
    if choice == "Datasets":
        st.markdown("### Datasets")

        datasets = _load_all_from(DATASETS_DIR)
        ds_names = list(datasets.keys())
        ds_sel = st.selectbox("Select dataset", options=["(new)"] + ds_names, key="ds_sel")

        if ds_sel == "(new)":
            st.markdown("**New dataset**")
            new_name = st.text_input("Name (filename without .yaml)", key="ds_new")
            with st.container():
                data_dir = st.text_input("data_dir", value="data/Hands", key="ds_dir")
                csv_path = st.text_input("csv_path", value="data/HandInfo.csv", key="ds_csv")
                try:
                    aspect_idx = aspects.index("palmar")
                except Exception:
                    aspect_idx = 0
                aspect = st.selectbox("aspect", aspects, index=aspect_idx, key="ds_aspect")
                clean_only = st.checkbox("clean_only", value=True, key="ds_clean")
                img_size = int(st.number_input("img_size", 64, 1024, 224, step=16, key="ds_img"))
                min_images = int(st.number_input("min_images", 1, 20, 3, key="ds_min"))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save dataset"):
                    if new_name:
                        _save_to(DATASETS_DIR, new_name, {
                            "data_dir": data_dir,
                            "csv_path": csv_path,
                            "aspect": aspect,
                            "clean_only": clean_only,
                            "img_size": img_size,
                            "min_images": min_images,
                        })
                        st.success(f"Saved configs/datasets/{new_name}.yaml")
                    else:
                        st.warning("Please provide a dataset name.")
            with c2:
                st.button("Delete dataset", type="primary", disabled=True)

        else:
            st.markdown(f"**Edit dataset:** `{ds_sel}`")
            cfg = datasets.get(ds_sel, {})
            with st.container():
                data_dir = st.text_input("data_dir", value=cfg.get("data_dir", ""), key="eds_dir")
                csv_path = st.text_input("csv_path", value=cfg.get("csv_path", ""), key="eds_csv")
                try:
                    aspect_idx = aspects.index(cfg.get("aspect", "palmar"))
                except Exception:
                    aspect_idx = 0
                aspect = st.selectbox("aspect", aspects, index=aspect_idx, key="eds_aspect")
                clean_only = st.checkbox("clean_only", value=bool(cfg.get("clean_only", False)), key="eds_clean")
                img_size = int(st.number_input("img_size", 64, 1024, int(cfg.get("img_size", 224)), step=16, key="eds_img"))
                min_images = int(st.number_input("min_images", 1, 20, int(cfg.get("min_images", 3)), key="eds_min"))

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save dataset"):
                    _save_to(DATASETS_DIR, ds_sel, {
                        "data_dir": data_dir,
                        "csv_path": csv_path,
                        "aspect": aspect,
                        "clean_only": clean_only,
                        "img_size": img_size,
                        "min_images": min_images,
                    })
                    st.success("Saved.")
            with c2:
                if st.button("Delete dataset", type="primary"):
                    (DATASETS_DIR / f"{ds_sel}.yaml").unlink(missing_ok=True)
                    st.warning("Deleted. Refresh to update list.")

    # ============================ PIPELINES ============================
    else:
        st.markdown("### Pipelines")

        pipelines = _load_all_from(PIPELINES_DIR)
        pl_names = list(pipelines.keys())
        pl_sel = st.selectbox("Select pipeline", options=["(new)"] + pl_names, key="pl_sel")

        # keep this state so the UI doesn't jump around
        if "stage_count" not in st.session_state:
            st.session_state.stage_count = 2

        if pl_sel == "(new)":
            st.markdown("**New pipeline**")
            new_name = st.text_input("Name (filename without .yaml)", key="pl_new")

            # Basic settings
            backbone = st.selectbox("backbone", backbones, index=0, key="pl_bb")
            seed = int(st.number_input("seed", 0, 999_999, 42, key="pl_seed"))

            st.divider()
            st.markdown("#### Stages")
            st.session_state.stage_count = int(st.number_input("Number of stages", 1, 6, st.session_state.stage_count, key="pl_nstages"))
            stages: List[Dict[str, Any]] = []

            # Indented/organized stages via expanders
            for i in range(st.session_state.stage_count):
                name_default = "A_ce" if i == 0 else ("B_triplet_pk16x4" if i == 1 else f"stage_{i+1}")
                with st.expander(f"Stage {i+1}", expanded=True if i < 2 else False):
                    name = st.text_input(f"s{i+1}.name", value=name_default, key=f"pl_s{i}_name")
                    # pick sensible default index inline
                    if i == 0:
                        loss_idx = 0
                    else:
                        try:
                            loss_idx = losses.index("triplet")
                        except Exception:
                            loss_idx = 0
                    loss = st.selectbox(f"s{i+1}.loss", losses, index=loss_idx, key=f"pl_s{i}_loss")
                    epochs = int(st.number_input(f"s{i+1}.epochs", 1, 500, (10 if i == 0 else 30), key=f"pl_s{i}_ep"))
                    batch_size = st.number_input(f"s{i+1}.batch_size (optional)", 1, 4096, 64, key=f"pl_s{i}_bs")
                    try:
                        pk_idx = pk_presets.index("16x4" if i == 1 else "")
                    except Exception:
                        pk_idx = len(pk_presets) - 1
                    pk = st.selectbox(f"s{i+1}.pk (optional)", pk_presets, index=pk_idx, key=f"pl_s{i}_pk")

                    stage_cfg: Dict[str, Any] = {"name": name, "loss": loss, "epochs": epochs}
                    if batch_size:
                        stage_cfg["batch_size"] = int(batch_size)
                    if pk:
                        stage_cfg["pk"] = pk
                    stages.append(stage_cfg)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save pipeline"):
                    if new_name:
                        _save_to(PIPELINES_DIR, new_name, {"backbone": backbone, "seed": seed, "stages": stages})
                        st.success(f"Saved configs/pipelines/{new_name}.yaml")
                    else:
                        st.warning("Please provide a pipeline name.")
            with c2:
                st.button("Delete pipeline", type="primary", disabled=True)

        else:
            st.markdown(f"**Edit pipeline:** `{pl_sel}`")
            cfg = pipelines.get(pl_sel, {})

            # Basic settings
            try:
                bb_idx = backbones.index(cfg.get("backbone", "resnet18"))
            except Exception:
                bb_idx = 0
            backbone = st.selectbox("backbone", backbones, index=bb_idx, key="epl_bb")
            seed = int(st.number_input("seed", 0, 999_999, int(cfg.get("seed", 42)), key="epl_seed"))

            st.divider()
            st.markdown("#### Stages")
            stages_cfg = cfg.get("stages", [])
            default_count = len(stages_cfg) or 1
            st.session_state.stage_count = int(st.number_input("Number of stages", 1, 6, default_count, key="epl_nstages"))

            stages: List[Dict[str, Any]] = []
            for i in range(st.session_state.stage_count):
                prev = stages_cfg[i] if i < len(stages_cfg) else {}
                with st.expander(f"Stage {i+1}", expanded=True if i < 2 else False):
                    name = st.text_input(f"s{i+1}.name", value=str(prev.get("name", f"stage_{i+1}")), key=f"epl_s{i}_name")
                    try:
                        loss_idx = losses.index(prev.get("loss", "ce"))
                    except Exception:
                        loss_idx = 0
                    loss = st.selectbox(f"s{i+1}.loss", losses, index=loss_idx, key=f"epl_s{i}_loss")
                    epochs = int(st.number_input(f"s{i+1}.epochs", 1, 500, int(prev.get("epochs", 10)), key=f"epl_s{i}_ep"))
                    batch_size = st.number_input(f"s{i+1}.batch_size (optional)", 1, 4096, int(prev.get("batch_size", 64)), key=f"epl_s{i}_bs")
                    try:
                        pk_idx = pk_presets.index(prev.get("pk", ""))
                    except Exception:
                        pk_idx = len(pk_presets) - 1
                    pk = st.selectbox(f"s{i+1}.pk (optional)", pk_presets, index=pk_idx, key=f"epl_s{i}_pk")

                    stage_cfg: Dict[str, Any] = {"name": name, "loss": loss, "epochs": epochs}
                    if batch_size:
                        stage_cfg["batch_size"] = int(batch_size)
                    if pk:
                        stage_cfg["pk"] = pk
                    stages.append(stage_cfg)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save pipeline"):
                    _save_to(PIPELINES_DIR, pl_sel, {"backbone": backbone, "seed": seed, "stages": stages})
                    st.success("Saved.")
            with c2:
                if st.button("Delete pipeline", type="primary"):
                    (PIPELINES_DIR / f"{pl_sel}.yaml").unlink(missing_ok=True)
                    st.warning("Deleted. Refresh to update list.")
