import sys
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path so absolute imports work when running this file directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from ui.models import render_models_tab
from ui.library import render_library_tab
from ui.train_eval import render_train_eval_tab
from ui.verify import render_verify_tab
from ui.pipelines import render_pipelines_tab
from ui.docs import render_docs_tab


def main():
    st.set_page_config(page_title="HandNet Trainer", layout="wide")
    st.title("Hand Verification – HandNet UI")

    tab_models, tab_library, tab_pipes, tab_train, tab_verify, tab_docs = st.tabs([
        "Models",
        "Library",
        "Pipelines",
        "Train / Eval",
        "Verify (Cosine)",
        "Docs",
    ])

    with tab_models:
        render_models_tab()
    with tab_library:
        render_library_tab()
    with tab_pipes:
        render_pipelines_tab()
    with tab_train:
        render_train_eval_tab()
    with tab_verify:
        render_verify_tab()
    with tab_docs:
        render_docs_tab()


if __name__ == "__main__":
    main()
