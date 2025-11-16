**Iris Poisoning Experiments (MLflow)**

This repository contains code and experiment artifacts for studying label/data poisoning effects on classification of the Iris dataset using MLflow for experiment tracking.

**Overview**:
- **Project**: Investigate how injecting poisoned samples affects model performance on the Iris dataset and track experiments with MLflow.
- **Language**: Python
- **Tracking**: MLflow (local `mlruns/` directory)

**Repository Layout**:
- `main.py`: Main entrypoint for training/experiment runs.
- `src/`: Project source code
	- `src/train.py`: Training / experiment helpers
	- `src/poison.py`: Poisoning utility functions
	- `src/utils.py`: Utility helpers
	- `src/plot.py`: Plotting and visualization helpers
- `data/iris.csv`: Dataset used for experiments
- `experiments/run_experiments.sh`: Bash script to run multiple experiments
- `mlruns/`: MLflow run data and artifacts (tracked results)
- `artifacts/`: Example artifact outputs (e.g., `classification_report.json`)

**Requirements & Setup**:
- **Python**: 3.8+ recommended
- Install dependencies:

```powershell
pip install -r requirements.txt
```

**Quick Usage**:
- To run a single experiment (default):

```powershell
python main.py
```

- To run the included batch script (on systems with Bash):

```bash
bash experiments/run_experiments.sh
```

Note: On Windows, run the script via Git Bash, WSL, or adapt commands for PowerShell.

**View Results with MLflow UI**:
- Start the MLflow UI pointing to the local `mlruns` directory:

```powershell
mlflow ui --backend-store-uri ./mlruns --port 5000

# then open http://localhost:5000 in your browser
```

**Reproducing a Specific Run**:
- MLflow stores each run under `mlruns/<experiment_id>/<run_id>/` with `params`, `metrics`, and `artifacts`.
- To re-run an experiment with different hyperparameters, modify `main.py` or call the training entry with the desired flags (see code in `src/` for available arguments).

**Results & Artifacts**:
- Example classification reports are in `artifacts/` and `mlruns/*/*/artifacts/classification_report.json` for individual runs.
- Model outputs are stored in `mlruns/*/*/outputs/` and `mlruns/*/models/` when exported.

**Notes & Tips**:
- The included `experiments/run_experiments.sh` automates multiple runs and logs parameter variations (e.g., `poison_fraction`, `n_estimators`, `random_state`). Use it to reproduce the experiment grid.
- If you want to explore runs programmatically, inspect the files under `mlruns/` or use the MLflow Python API to load run data.

**Contact & License**:
- Author: repository owner
- License: see repository or ask the owner for details.

If you want, I can also:
- add example command-line flags to the README after inspecting `main.py` and `src/train.py` (recommended),
- or commit these changes to a branch and create a short changelog entry.

