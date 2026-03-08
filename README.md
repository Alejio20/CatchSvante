# Catch Svante

Binary classification system that identifies web-browsing sessions belonging to `user_id = 0` ("Svante") versus all other users, based on session metadata and visited-site patterns.

## Project Structure

```
Project/
├── data/
│   ├── raw/                        # Original JSON session files
│   │   ├── dataset.json            # Labelled sessions (~160 K rows)
│   │   └── verify.json             # Unlabelled sessions (~22.7 K rows)
│   ├── interim/                    # Converted CSVs
│   │   ├── dataset.csv
│   │   └── verify.csv
│   └── processed/                  # Train / test splits
│       ├── train.csv
│       └── test.csv
│
├── src/svantecatch/                 # Core Python package
│   ├── __init__.py
│   ├── config.py                   # FeatureConfig dataclass (all hyper-parameters)
│   ├── convert.py                  # JSON → CSV conversion
│   ├── features.py                 # Feature extraction (TF-IDF, site stats, time)
│   ├── model.py                    # Pipeline construction & model persistence
│   ├── evaluate.py                 # Threshold tuning & classification metrics
│   └── utils.py                    # Lightweight I/O helpers
│
├── scripts/                        # CLI entry-points
│   ├── full_pipeline.py            # Runs all 4 steps end-to-end
│   ├── make_dataset.py             # Step 1: JSON → CSV
│   ├── split_dataset.py            # Step 2: stratified train / test split
│   ├── train.py                    # Step 3: train + threshold tune + evaluate
│   └── predict.py                  # Step 4: score verification set
│
├── notebooks/
│   └── eda_presentation.ipynb      # Exploratory data analysis
│
├── artifacts/                      # Trained model & feature spec
│   ├── svante_model.joblib
│   └── feature_spec.json
│
├── runs/                           # Timestamped training summaries
│   └── <YYYYMMDD_HHMMSS>/
│       └── training_summary.json
│
├── result.csv                      # Submission labels (0 = Svante, 1 = not Svante)
├── result_with_probs.csv           # Labels + raw probabilities
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# Install dependencies and the svantecatch package in editable mode
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Run the full pipeline (convert, split, train, predict) in one command:

```bash
python scripts/full_pipeline.py
```

This produces `result.csv` with predictions on the verification set.

## Detailed Steps

### 1. Convert JSON to CSV

```bash
python scripts/make_dataset.py \
    --train_json data/raw/dataset.json \
    --verify_json data/raw/verify.json \
    --out_dir data/interim
```

### 2. Split into Train / Test

```bash
python scripts/split_dataset.py \
    --dataset_csv data/interim/dataset.csv \
    --out_dir data/processed \
    --test_size 0.2 \
    --seed 42
```

### 3. Train + Evaluate

```bash
python scripts/train.py \
    --train_csv data/processed/train.csv \
    --test_csv  data/processed/test.csv \
    --out_dir   artifacts \
    --runs_dir  runs
```

**Available options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model_type` | `hist_gradient_boosting` | `hist_gradient_boosting`, `random_forest`, or `logistic_regression` |
| `--n_estimators` | `300` | Boosting rounds (HGB) or number of trees (RF) |
| `--threshold_metric` | `f1` | Metric to maximise during threshold tuning (`f1`, `precision`, `recall`) |
| `--tfidf_max_features` | `10000` | Maximum TF-IDF vocabulary size |
| `--svd_components` | `0` (auto) | TruncatedSVD dimensions; auto-set to 200 for tree models |
| `--val_size` | `0.2` | Fraction of training data used for threshold tuning |
| `--seed` | `42` | Random seed for reproducibility |

When `--model_type logistic_regression` is selected, the script automatically runs a grid search over regularisation strength *C* and picks the best value on the validation fold.

### 4. Predict on Verification Set

```bash
python scripts/predict.py \
    --verify_csv data/interim/verify.csv \
    --model      artifacts/svante_model.joblib \
    --spec       artifacts/feature_spec.json \
    --out        result.csv
```

Output format: one label per line (`0` = Svante, `1` = not Svante).

## Model Architecture

The pipeline combines four feature branches via a `ColumnTransformer`:

| Branch | Input Columns | Transformation | Output |
|--------|--------------|----------------|--------|
| **cats** | `browser`, `os`, `locale`, `gender`, `location` | One-hot encoding | ~50 binary features |
| **time** | `date`, `time` | Parse → hour, weekday, month → scale | 3 numeric features |
| **sites_text** | `sites_json` | Tokenise → TF-IDF → TruncatedSVD | 200 dense features |
| **sites_num** | `sites_json` | Aggregate stats (count, sum, mean, max, min, std, median, unique domains, CV, range, entropy) → scale | 11 numeric features |

The default classifier is **HistGradientBoostingClassifier** with `class_weight="balanced"` to handle the extreme class imbalance (~0.5% positive rate).

## Evaluation Metrics

Performance on the held-out test set (32 000 sessions, 160 Svante):

| Metric | Value |
|--------|-------|
| **F1** | **0.810** |
| **Precision** | **0.927** |
| **Recall** | **0.719** |
| **PR-AUC** | **0.916** |
| **ROC-AUC** | **0.999** |

## Dependencies

- Python >= 3.10
- numpy, pandas, scikit-learn, joblib, matplotlib, seaborn

See `requirements.txt` for pinned minimum versions.
