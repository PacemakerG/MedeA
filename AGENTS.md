# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-driven PTB-XL ECG research workspace. Core model code lives in `models/`: baseline trainers are in `models/baselinemodels/`, MeDeA variants are top-level files such as `models/MedeA.py`, and ablation studies are in `models/Ablation/`. Preprocessed datasets are stored in `processed_data/` and `processed_data_23subclasses/`. Generated checkpoints belong in `saved_models/` or `models/saved_models/`, and reports and charts belong in `experiment_visualizations/`, `ECG_Classification_Experiment_Report.md`, and related `.md` files at the repo root.

## Build, Test, and Development Commands
Use the existing Python entry scripts rather than introducing new wrappers.

```bash
conda activate LLM
python run_comprehensive_experiment.py --epochs 30 --batch_size 32 --lr 1e-3
python models/baselinemodels/cnn_baseline.py --data_file processed_data/ptbxl_processed_100hz_fold1.npz
python models/MedeA.py --cross_validation --data_dir processed_data --output_dir saved_models/medea_experiment
python generate_comprehensive_report.py
python datacheck.py
```

`run_comprehensive_experiment.py` launches multi-model training, individual model scripts handle focused experiments, `generate_comprehensive_report.py` rebuilds summaries, and `datacheck.py` verifies dataset integrity.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for model classes, and descriptive script names such as `inception_baseline.py` or `MedeA_23.py`. Keep CLI arguments explicit with `argparse`, preserve reproducibility controls such as `seed`, and prefer small, local changes over broad refactors. There is no repo-level formatter config here, so run your usual formatter/linter locally only if it does not rewrite unrelated files.

## Testing Guidelines
There is no dedicated `tests/` package in this snapshot. Treat runnable training and validation scripts as the regression surface: run the affected model entry point on a small fold, then rerun `python datacheck.py` and regenerate reports when outputs change. Name new checks after the target module, for example `test_medea_data_loading.py` if you add formal tests later.

## Commit & Pull Request Guidelines
Git history is not available in this exported workspace, so no repository-specific commit convention can be verified here. Use short imperative subjects, keep each commit focused, and mention the experiment or model touched, for example `feat: tune MedeA cross-validation logging`. Pull requests should summarize the model or data change, list commands run, link the related issue or experiment note, and include updated metrics or screenshots when plots or reports change.

## Data & Artifact Hygiene
Do not commit raw dataset downloads, large transient checkpoints, or ad hoc notebooks. Keep generated `.pth`, `.npz`, `.png`, and report files in the existing artifact directories, and document any new required dataset path or environment variable in `README.md`.
