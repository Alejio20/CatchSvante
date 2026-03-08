"""
Orchestrator that runs the full Catch Svante pipeline end-to-end.

Each step is executed as a subprocess so that the individual scripts
can also be run independently.  The orchestrator aborts on the first
failure and prints the captured ``stderr`` for debugging.

Steps
-----
1. **make_dataset** -- Convert raw JSON session files to interim CSVs.
2. **split_dataset** -- Stratified train / test split.
3. **train** -- Fit the classifier, tune the decision threshold, and
   evaluate on the test set.
4. **predict** -- Score the unlabelled verification set and write
   ``result.csv``.

Usage
-----
::

    python scripts/full_pipeline.py
"""

import subprocess
import sys


def _run_step(step_num: int, description: str, cmd: list[str]) -> None:
    """Execute *cmd* as a subprocess and abort on failure."""
    print(f"Step {step_num}: {description}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error in Step {step_num}: {result.stderr}")
        sys.exit(1)
    print(f"Step {step_num} complete.\n")


def main() -> None:
    """Run all four pipeline steps sequentially."""
    _run_step(1, "Converting JSON to CSV", [
        sys.executable, "scripts/make_dataset.py",
        "--train_json", "data/raw/dataset.json",
        "--verify_json", "data/raw/verify.json",
        "--out_dir", "data/interim",
    ])

    _run_step(2, "Splitting dataset into train/test", [
        sys.executable, "scripts/split_dataset.py",
        "--dataset_csv", "data/interim/dataset.csv",
        "--out_dir", "data/processed",
    ])

    _run_step(3, "Training and evaluating model", [
        sys.executable, "scripts/train.py",
        "--train_csv", "data/processed/train.csv",
        "--test_csv", "data/processed/test.csv",
        "--out_dir", "artifacts",
        "--runs_dir", "runs",
    ])

    _run_step(4, "Predicting on verification set", [
        sys.executable, "scripts/predict.py",
        "--verify_csv", "data/interim/verify.csv",
        "--model", "artifacts/svante_model.joblib",
        "--spec", "artifacts/feature_spec.json",
        "--out", "result.csv",
    ])

    print("Pipeline finished!")


if __name__ == "__main__":
    main()