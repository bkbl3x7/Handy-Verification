from pathlib import Path
import streamlit as st
import pandas as pd

from .utils import scan_runs_flat


def render_library_tab():
    st.subheader("Library – All Models")
    runs = scan_runs_flat("experiments")
    rows = []
    for r in runs:
        stages = r.get("stages", [])
        last = stages[-1] if stages else {}
        metrics = last.get("metrics") or {}
        rows.append({
            "Dataset": r.get("dataset"),
            "Pipeline": r.get("pipeline"),
            "Backbone": r.get("backbone"),
            "Run": Path(r.get("path")).name,
            "Dev EER": metrics.get("dev_eer"),
            "Test EER": metrics.get("test_eer"),
            "AUC": metrics.get("test_auc"),
            "Path": r.get("path"),
        })

    if rows:
        # Default sort (no UI controls): by Test EER ascending
        df_runs = pd.DataFrame(rows).sort_values(by="Test EER", ascending=True)
        st.dataframe(df_runs.drop(columns=["Path"]), use_container_width=True)

        # Open run selector
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

            # --- Actions: compact, top-left (Download + red Delete) ---
            # Style primary buttons as red (we'll make Delete primary).
            st.markdown(
                """
                <style>
                button[data-testid="baseButton-primary"]{
                    background-color:#dc2626 !important;  /* red-600 */
                    border-color:#dc2626 !important;
                    color:#ffffff !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            col_dl, col_del, _spacer = st.columns([0.18, 0.18, 0.64], gap="small")
            with col_dl:
                import io, zipfile
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in run_dir.rglob("*"):
                        if p.is_file():
                            zf.write(p, arcname=p.relative_to(run_dir))
                buf.seek(0)
                # No use_container_width so it stays compact
                st.download_button(
                    label="Download",
                    data=buf,
                    file_name=f"{run_dir.name}.zip",
                    mime="application/zip",
                )

            with col_del:
                # Primary makes it red via CSS above
                if st.button("Delete", type="primary", key="delete_exp_btn"):
                    import shutil
                    try:
                        shutil.rmtree(run_dir)
                        st.success("Experiment deleted.")
                        st.session_state["lib_open_path"] = "(none)"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
            # -----------------------------------------------------------

            # Details
            import json as _json
            info = {}
            try:
                info = _json.loads((run_dir / "manifest.json").read_text())
            except Exception:
                pass

            cols = st.columns(2)
            with cols[0]:
                st.write("Metrics")
                stages = info.get("stages", [])
                if stages:
                    st.json(stages[-1].get("metrics", {}))
                st.write("Run")
                st.json({k: info.get(k) for k in ["backbone", "seed", "created_at"]})
                st.write("Dataset (snapshot)")
                st.json(info.get("dataset_cfg", {}))
                st.write("Pipeline (snapshot)")
                st.json(info.get("pipeline_cfg", {}))
            with cols[1]:
                st.write("ROC Curve")
                if stages and stages[-1].get("roc"):
                    st.image(stages[-1]["roc"], use_container_width=True)
                sp = run_dir / f"{run_dir.name}.csv"
                if sp and sp.exists():
                    try:
                        sdf = pd.read_csv(sp)
                        st.write("Score distribution")
                        st.bar_chart(sdf["score"])
                    except Exception:
                        pass
    else:
        st.info("No experiments found. Train a model or check the Pipelines tab.")
