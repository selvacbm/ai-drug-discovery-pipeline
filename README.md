# Drug Pipe

This project is a prototype AI-guided small-molecule discovery workflow for `KRAS`.
It is useful as a reproducible demo pipeline, but it is not a substitute for a fully validated medicinal chemistry campaign.

## What It Does

1. Pulls or reuses curated ChEMBL IC50 data for the target.
2. Trains a QSAR model with Morgan fingerprints plus molecular descriptors.
3. Reports both random-split and scaffold-split validation metrics.
4. Generates candidate analogs with BRICS recombination.
5. Filters generated molecules with basic medicinal chemistry sanity checks.
6. Ranks candidates by predicted activity.
7. Optionally docks the top candidates with AutoDock Vina.
8. Writes ranked outputs, plots, molecule images, and a run summary.

## What Changed In This Version

- Model features are stronger than the original 5-descriptor setup.
- Validation is more honest: scaffold-split metrics are reported and marked as the recommended metric for reporting.
- Offline behavior is better: if ChEMBL is unreachable and cached data exists, the pipeline falls back quickly.
- Docking is more transparent: the script records where the docking box came from.
- Generated molecules are deduplicated and filtered more carefully.
- A `results/run_summary.json` file captures what actually happened during a run.

## Important Scientific Limits

- BRICS recombination is not a deep generative AI model. It is structure recombination.
- Docking is only as good as the docking box and receptor preparation.
- If the docking box is inferred from KRAS site residues instead of a co-crystal ligand, treat docking results as approximate.
- A scaffold-split QSAR metric is still not enough to claim prospective success.
- This project does not include synthesis planning, MD, FEP, assay integration, or experimental validation.

## Recommended Language

Good description:

`Prototype AI-assisted hit ideation and prioritization workflow for KRAS`

Avoid claiming:

`Complete AI drug discovery pipeline`

unless you add validated pocket definition, external benchmarking, stronger prospective evaluation, and downstream experimental confirmation.

## Usage

Run with local cached data only:

```bash
/Users/selvaraj/miniconda3/envs/drug_env/bin/python pipeline.py --cache-only
```

Run with manual docking box:

```bash
/Users/selvaraj/miniconda3/envs/drug_env/bin/python pipeline.py \
  --center 1.0 -3.0 4.0 \
  --size 20 20 20 \
  --vina-exhaustiveness 8
```

## Main Outputs

- `results/data/cleaned_data.csv`
- `results/data/generated.csv`
- `results/data/generated_predictions.csv`
- `results/model/metrics.txt`
- `results/plots/distribution.png`
- `results/plots/parity.png`
- `results/top_hits.csv`
- `results/run_summary.json`

## What To Improve Next

1. Use a co-crystal ligand or curated pocket coordinates for docking.
2. Add scaffold-aware hyperparameter tuning and confidence estimation.
3. Add external test sets or time-split validation.
4. Add stronger medicinal chemistry filters and synthetic accessibility scoring.
5. Add downstream refinement such as rescoring, MD, or free-energy calculations.
