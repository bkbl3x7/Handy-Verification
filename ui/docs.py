import streamlit as st


def render_docs_tab():
    st.title("Hand Verification App – Guide")

    st.markdown(
        """
        ## Overview
        This app streamlines training, evaluating, browsing, and verifying hand verification models.
        It wraps a simple PyTorch pipeline and provides a friendly UI for common workflows.

        ### Key Concepts
        - Dataset: A reusable configuration that tells the app where images and metadata live, and how to filter them.
        - Pipeline (Recipe): A sequence of 1–N training stages (e.g., CE pretrain, then Triplet with a P×K sampler).
        - Run (Experiment): A concrete execution of a Pipeline on a Dataset. Each run lives in `experiments/<datetime>_<dataset>_<pipeline>/`.
        - Manifest: A JSON summary of a run (`manifest.json`) with dataset/pipeline/backbone/seed, and per-stage metrics and artifacts.

        ### Metrics & Terms
        - EER (Equal Error Rate): The point where false accept rate equals false reject rate. Lower is better.
        - AUC (Area Under the ROC Curve): Probability the model ranks a genuine pair higher than an impostor pair. Higher is better.
        - ROC (Receiver Operating Characteristic): Curve of TPR vs FPR across thresholds.
        - CE (Cross-Entropy): Classification loss over subject IDs.
        - Triplet Loss: Metric learning loss that pulls together same-ID embeddings and pushes apart different-ID embeddings.
        - P×K Sampler: For Triplet, batches contain P identities with K images each (e.g., 16×4) to ensure good positives/negatives.

        ## Tabs & Workflows
        ### Models
        Featured runs with quick filters.
        - Filter by Dataset and Pipeline.
        - Card actions:
          - Open: Preselect in Library to view details.
          - Verify: Prefill Verify tab with this run's weights.

        ### Library
        Browse all runs (sortable table by clicking headers).
        - Select a run using "Open run".
        - Actions (top-left):
          - Download experiment: ZIP the whole run folder.
          - Delete experiment: Permanently remove the run folder.
        - Details: Shows last-stage metrics, run arguments, ROC image, and score histogram.

        ### Pipelines
        Define and manage:
        - Datasets (one YAML per dataset under `configs/datasets/`).
        - Pipelines (one YAML per pipeline under `configs/pipelines/`).
        The UI uses dropdowns and inputs (backed by `configs/options.json`) so users do not need to edit YAML directly.

        Dataset fields
        - data_dir: Root folder containing images (subfolders allowed).
        - csv_path: CSV with at least `imageName` and `id` columns.
        - aspect: Filter (palmar/dorsal/any) if present in CSV.
        - clean_only: Drop rows flagged for accessories/nailPolish/irregularities if present in CSV.
        - img_size: Square image size for preprocessing and models.
        - min_images: Minimum number of images per subject to include.

        Pipeline fields
        - backbone: `resnet18` or `handnet` (from options.json).
        - seed: Random seed for reproducibility.
        - stages: Each stage has `name`, `loss` (ce/triplet), `epochs`, optional `batch_size`, optional `pk` (e.g., 16x4) for Triplet.

        ### Train / Eval
        - Select a Dataset + Pipeline.
        - Click "Run Pipeline" to execute each stage sequentially.
        - Live logs stream for each stage.
        - Output goes to `experiments/<datetime>_<dataset>_<pipeline>/` with artifacts named `{experiment}.pt/.json/.png/.csv`.
        - `manifest.json` summarizes per-stage metrics and artifacts.

        ### Verify (Cosine)
        - Provide a run weights file (.pt) and upload 1–5 enroll images plus a probe image.
        - The app infers backbone and uses the run's img_size; it returns the cosine similarity.

        ## Files & Layout
        - Experiments: `experiments/<datetime>_<dataset>_<pipeline>/`
          - `{experiment}.pt`: best weights of the latest stage
          - `{experiment}.json`: metrics of the latest stage
          - `{experiment}.png`: ROC of the latest stage
          - `{experiment}.csv`: score pairs of the latest stage
          - `manifest.json`: single source of truth for the run (snapshots dataset/pipeline configs, run info, and per-stage metrics/artifacts)
        - Configs:
          - `configs/options.json`: Allowed dropdown values for backbones/losses/aspects/P×K.
          - `configs/datasets/*.yaml`: One YAML file per dataset.
          - `configs/pipelines/*.yaml`: One YAML file per pipeline recipe.

        ## Tips
        - Start with `resnet18` + CE pretrain then Triplet with P×K for better verification.
        - Keep datasets clean with `clean_only` and a reasonable `min_images` to reduce label noise.
        - Use the Library to compare runs and prune older experiments.
        """
    )
