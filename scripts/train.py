"""
Train the Catch Svante classifier and evaluate on a held-out test set.

Workflow
-------
1. Load the pre-split ``train.csv`` and ``test.csv``.
2. Create a secondary train / validation fold from ``train.csv``
   (stratified when possible) for threshold tuning.
3. **Logistic regression only:** run an automatic grid search over the
   regularisation parameter *C* to find the best setting on the
   validation fold.  Other model types train once with the CLI-supplied
   hyper-parameters.
4. Tune the probability decision threshold on the validation fold by
   maximising the chosen metric (default: F1).
5. Refit the pipeline on the **full** ``train.csv`` with the selected
   config (so no training data is wasted).
6. Evaluate on ``test.csv``, save the model artefacts and a
   JSON run summary under ``runs/<timestamp>/``.

Usage
-----
::

    python scripts/train.py \\
        --train_csv data/processed/train.csv \\
        --test_csv  data/processed/test.csv \\
        --out_dir   artifacts \\
        --runs_dir  runs

    # Use logistic regression with automatic C search:
    python scripts/train.py ... --model_type logistic_regression

    # Override TF-IDF vocabulary and tree depth:
    python scripts/train.py ... --tfidf_max_features 20000 \\
                                --n_estimators 800
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from svantecatch.config import FeatureConfig
from svantecatch.evaluate import pick_threshold, compute_threshold_metrics, add_auc_metrics
from svantecatch.model import build_pipeline, save_model, make_feature_spec
from svantecatch.utils import ensure_dir, utc_timestamp, save_json


def main() -> None:
    """Run the full train-tune-evaluate-save workflow."""
    p = argparse.ArgumentParser(
        description="Train Catch Svante model and evaluate on a held-out test set.",
    )
    p.add_argument("--train_csv", required=True,
                    help="Path to the training split (must include user_id).")
    p.add_argument("--test_csv", required=True,
                    help="Path to the test split (must include user_id).")
    p.add_argument("--out_dir", required=True,
                    help="Directory for model and feature-spec artefacts.")
    p.add_argument("--runs_dir", required=True,
                    help="Parent directory for timestamped run summaries.")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    p.add_argument("--val_size", type=float, default=0.2,
                    help="Fraction of training data held out for threshold tuning.")
    p.add_argument("--model_type", default="hist_gradient_boosting",
                    choices=["random_forest", "logistic_regression",
                             "hist_gradient_boosting"],
                    help="Classifier back-end.")
    p.add_argument("--n_estimators", type=int, default=300,
                    help="Number of boosting rounds (HGB) or trees (RF).")
    p.add_argument("--threshold_metric", default="f1",
                    choices=["f1", "precision", "recall"],
                    help="Metric to maximise during threshold tuning.")
    p.add_argument("--tfidf_max_features", type=int, default=10000,
                    help="Maximum TF-IDF vocabulary size.")
    p.add_argument("--svd_components", type=int, default=0,
                    help="TruncatedSVD dimensions.  0 = auto (200 for "
                         "tree models, disabled for logistic regression).")
    args = p.parse_args()

    # Tree-based models require dense input; auto-enable SVD to reduce
    # the sparse TF-IDF matrix to a manageable dense representation.
    svd = args.svd_components
    if svd == 0 and args.model_type in ("hist_gradient_boosting", "random_forest"):
        svd = 200

    ensure_dir(args.out_dir)
    ensure_dir(args.runs_dir)

    # ------------------------------------------------------------------
    # 1. Load data and create binary labels (1 = Svante, 0 = everyone else)
    # ------------------------------------------------------------------
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    for name, df in [("train", train_df), ("test", test_df)]:
        if "user_id" not in df.columns:
            raise ValueError(f"{name} csv must contain user_id")

    y_train_full = (train_df["user_id"] == 0).astype(int).values
    X_train_full = train_df.drop(columns=["user_id"])

    y_test = (test_df["user_id"] == 0).astype(int).values
    X_test = test_df.drop(columns=["user_id"])

    # Prune rare TF-IDF tokens in larger datasets to reduce noise
    min_df = 1 if len(train_df) < 500 else 2
    cfg = FeatureConfig(
        random_seed=args.seed,
        tfidf_min_df=min_df,
        tfidf_max_features=args.tfidf_max_features,
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        threshold_metric=args.threshold_metric,
        svd_components=svd,
    )

    # ------------------------------------------------------------------
    # 2. Internal train / validation split for threshold tuning
    # ------------------------------------------------------------------
    stratify = y_train_full if (y_train_full.sum() >= 2 and (len(y_train_full) - y_train_full.sum()) >= 2) else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=args.val_size,
        random_state=cfg.random_seed,
        stratify=stratify,
    )

    # ------------------------------------------------------------------
    # 3. Fit on train subset  +  tune threshold on validation
    #    For LR, sweep over C values to find the best regularisation.
    # ------------------------------------------------------------------
    if args.model_type == "logistic_regression":
        c_grid = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
        best_val_f1, best_c = -1.0, cfg.logreg_c
        best_threshold, best_val_stats = 0.5, {}
        print("Searching over C values...")
        for c_val in c_grid:
            cfg_c = FeatureConfig(
                random_seed=args.seed,
                tfidf_min_df=min_df,
                tfidf_max_features=args.tfidf_max_features,
                model_type="logistic_regression",
                logreg_c=c_val,
                threshold_metric=args.threshold_metric,
                svd_components=svd,
            )
            m = build_pipeline(cfg_c)
            m.fit(X_tr, y_tr)
            vp = m.predict_proba(X_val)[:, 1]
            t, vs = pick_threshold(y_val, vp, metric=cfg_c.threshold_metric)
            print(f"  C={c_val:>6.2f}  val_F1={vs['f1']:.4f}  "
                  f"P={vs['precision']:.4f}  R={vs['recall']:.4f}  thr={t:.4f}")
            if vs["f1"] > best_val_f1:
                best_val_f1 = vs["f1"]
                best_c = c_val
                best_threshold = t
                best_val_stats = vs
        print(f"\nBest C={best_c}, val F1={best_val_f1:.4f}")
        cfg = FeatureConfig(
            random_seed=args.seed,
            tfidf_min_df=min_df,
            tfidf_max_features=args.tfidf_max_features,
            model_type="logistic_regression",
            logreg_c=best_c,
            threshold_metric=args.threshold_metric,
            svd_components=svd,
        )
        threshold = best_threshold
        val_stats = best_val_stats
    else:
        model = build_pipeline(cfg)
        model.fit(X_tr, y_tr)
        val_prob = model.predict_proba(X_val)[:, 1]
        threshold, val_stats = pick_threshold(y_val, val_prob, metric=cfg.threshold_metric)

    # ------------------------------------------------------------------
    # 4. Refit on full training data (no validation hold-out) so the
    #    final model sees every available labelled example.
    # ------------------------------------------------------------------
    model = build_pipeline(cfg)
    model.fit(X_train_full, y_train_full)

    # ------------------------------------------------------------------
    # 5. Evaluate on the test set (threshold is fixed from step 3)
    # ------------------------------------------------------------------
    test_prob = model.predict_proba(X_test)[:, 1]
    test_stats = compute_threshold_metrics(y_test, test_prob, threshold)
    add_auc_metrics(test_stats, y_test, test_prob)

    # ------------------------------------------------------------------
    # 6. Persist artefacts and run summary
    # ------------------------------------------------------------------
    model_path = os.path.join(args.out_dir, "svante_model.joblib")
    spec_path = os.path.join(args.out_dir, "feature_spec.json")
    save_model(model, model_path)

    spec = make_feature_spec(cfg, threshold=threshold)
    save_json(spec_path, spec)

    run_id = utc_timestamp()
    run_dir = os.path.join(args.runs_dir, run_id)
    ensure_dir(run_dir)

    summary = {
        "run_id": run_id,
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(X_val)),
            "test": int(len(test_df)),
        },
        "positives": {
            "svante_train": int(y_train_full.sum()),
            "svante_val": int(y_val.sum()),
            "svante_test": int(y_test.sum()),
        },
        "threshold": float(threshold),
        "val_metrics": val_stats,
        "test_metrics": test_stats,
        "config": spec["config"],
        "artifacts": {"model_path": model_path, "spec_path": spec_path},
    }
    save_json(os.path.join(run_dir, "training_summary.json"), summary)

    print(f"Saved model: {model_path}")
    print(f"Saved spec:  {spec_path}")
    print(f"Saved run:   {run_dir}")
    print("Test metrics:", test_stats)


if __name__ == "__main__":
    main()
