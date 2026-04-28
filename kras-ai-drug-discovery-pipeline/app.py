from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PIPELINE_PATH = BASE_DIR / "pipeline.py"
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = RESULTS_DIR / "data"
PLOTS_DIR = RESULTS_DIR / "plots"
MOLS_DIR = RESULTS_DIR / "molecules"
SUMMARY_FILE = RESULTS_DIR / "run_summary.json"
TOP_HITS_FILE = RESULTS_DIR / "top_hits.csv"


def ensure_results_dirs() -> None:
    for path in [RESULTS_DIR, DATA_DIR, PLOTS_DIR, MOLS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def save_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    required_columns = {"smiles", "pIC50"}
    if not required_columns.issubset(df.columns):
        raise ValueError("Uploaded CSV must contain 'smiles' and 'pIC50' columns.")

    df = df.loc[:, ["smiles", "pIC50"]].copy()
    df.to_csv(DATA_DIR / "cleaned_data.csv", index=False)
    return df


def run_pipeline(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPYCACHEPREFIX"] = "/tmp"
    command = [sys.executable, str(PIPELINE_PATH), *args]
    return subprocess.run(
        command,
        cwd=BASE_DIR,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def read_summary() -> dict:
    if not SUMMARY_FILE.exists():
        return {}
    return json.loads(SUMMARY_FILE.read_text())


def render_status_box(summary: dict) -> None:
    if not summary:
        st.info("No pipeline run detected yet.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Status", summary.get("status", "unknown"))
    col2.metric("Training Molecules", summary.get("n_training_molecules", 0))
    col3.metric("Generated Candidates", summary.get("n_generated_candidates", 0))

    if "n_docked" in summary:
        st.caption(f"Docked molecules: {summary['n_docked']}")


def render_results() -> None:
    summary = read_summary()
    render_status_box(summary)

    if summary:
        with st.expander("Run Summary", expanded=False):
            st.json(summary)

    if TOP_HITS_FILE.exists():
        st.subheader("Top Hits")
        hits_df = pd.read_csv(TOP_HITS_FILE)
        st.dataframe(hits_df, use_container_width=True)
        st.download_button(
            "Download Top Hits CSV",
            data=TOP_HITS_FILE.read_bytes(),
            file_name="top_hits.csv",
            mime="text/csv",
        )

    plot_files = [
        PLOTS_DIR / "parity_random.png",
        PLOTS_DIR / "parity_scaffold.png",
        PLOTS_DIR / "distribution.png",
    ]
    available_plots = [path for path in plot_files if path.exists()]
    if available_plots:
        st.subheader("Plots")
        cols = st.columns(len(available_plots))
        for col, plot_path in zip(cols, available_plots):
            col.image(str(plot_path), caption=plot_path.name, use_container_width=True)

    mol_images = sorted(MOLS_DIR.glob("mol_*.png"))[:12]
    if mol_images:
        st.subheader("Top Molecule Images")
        cols = st.columns(3)
        for idx, image_path in enumerate(mol_images):
            cols[idx % 3].image(str(image_path), caption=image_path.name, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="KRAS Drug Pipeline", page_icon="🧬", layout="wide")
    ensure_results_dirs()

    st.title("KRAS Drug Discovery Pipeline")
    st.caption("Streamlit wrapper for the KRAS training, screening, and docking pipeline.")

    with st.sidebar:
        st.header("Run Settings")
        target = st.text_input("Target Query", value="KRAS")
        max_mols = st.number_input("Max Training Molecules", min_value=50, max_value=5000, value=400, step=50)
        max_analogs = st.number_input("Max BRICS Analogs", min_value=10, max_value=5000, value=400, step=10)
        max_dock = st.number_input("Max Molecules to Dock", min_value=1, max_value=200, value=20, step=1)
        cache_only = st.checkbox("Use only local cleaned_data.csv", value=False)
        vina_exhaustiveness = st.number_input("Vina Exhaustiveness", min_value=1, max_value=64, value=8, step=1)

    st.subheader("Input Dataset")
    uploaded_file = st.file_uploader(
        "Optional: upload a cleaned training dataset CSV with smiles and pIC50 columns",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            uploaded_df = save_uploaded_dataset(uploaded_file)
            st.success(f"Saved uploaded dataset to {DATA_DIR / 'cleaned_data.csv'}")
            st.dataframe(uploaded_df.head(20), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    dataset_path = DATA_DIR / "cleaned_data.csv"
    if dataset_path.exists():
        st.caption(f"Current local dataset: {dataset_path}")

    st.subheader("Run Pipeline")
    if st.button("Run KRAS Pipeline", type="primary", use_container_width=True):
        if not PIPELINE_PATH.exists():
            st.error(f"Missing pipeline file: {PIPELINE_PATH}")
            st.stop()

        cli_args = [
            "--target",
            str(target),
            "--max-mols",
            str(max_mols),
            "--max-analogs",
            str(max_analogs),
            "--max-dock",
            str(max_dock),
            "--vina-exhaustiveness",
            str(vina_exhaustiveness),
        ]
        if cache_only:
            cli_args.append("--cache-only")

        with st.spinner("Running pipeline. This can take a while for docking runs."):
            result = run_pipeline(cli_args)

        st.subheader("Execution Log")
        if result.stdout.strip():
            st.code(result.stdout)
        if result.stderr.strip():
            st.code(result.stderr)

        if result.returncode == 0:
            st.success("Pipeline completed.")
        else:
            st.error(f"Pipeline exited with code {result.returncode}")

    st.divider()
    render_results()


if __name__ == "__main__":
    main()
