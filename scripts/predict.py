"""
Predict Svante vs not-Svante labels for unlabelled verification sessions.

Loads a trained pipeline (``svante_model.joblib``) and the accompanying
feature specification (``feature_spec.json``), scores every session in
the verification CSV, and writes two output files:

* ``result.csv`` -- one integer label per line in submission format
  (``0`` = Svante, ``1`` = not Svante).
* ``result_with_probs.csv`` -- raw probabilities alongside the
  predicted labels for manual inspection.

The decision threshold stored in ``feature_spec.json`` is the one
tuned during training; it is **not** re-tuned here.

Usage
-----
::

    python scripts/predict.py \\
        --verify_csv data/interim/verify.csv \\
        --model      artifacts/svante_model.joblib \\
        --spec       artifacts/feature_spec.json \\
        --out        result.csv
"""

import argparse
import json
import os

import pandas as pd

from svantecatch.model import load_model
from svantecatch.utils import ensure_dir


def main() -> None:
    """Load model + spec, score verification sessions, and write results."""
    p = argparse.ArgumentParser(
        description="Predict Svante vs not Svante for verification sessions.",
    )
    p.add_argument("--verify_csv", required=True,
                    help="Path to the unlabelled verification CSV.")
    p.add_argument("--model", required=True,
                    help="Path to the trained model (joblib).")
    p.add_argument("--spec", required=True,
                    help="Path to feature_spec.json (contains threshold).")
    p.add_argument("--out", required=True,
                    help="Output path for submission labels.")
    p.add_argument("--out_probs", default="result_with_probs.csv",
                    help="Output path for labels + probabilities.")
    args = p.parse_args()

    df = pd.read_csv(args.verify_csv)
    model = load_model(args.model)

    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    threshold = float(spec.get("threshold", 0.5))

    prob = model.predict_proba(df)[:, 1]
    pred_svante = (prob >= threshold).astype(int)

    # The model internally encodes Svante as 1 (positive class), but the
    # submission format requires 0 = Svante, 1 = not Svante.
    out_labels = [0 if pj == 1 else 1 for pj in pred_svante.tolist()]

    out_dir = os.path.dirname(args.out) or "."
    ensure_dir(out_dir)

    with open(args.out, "w", encoding="utf-8") as f:
        for y in out_labels:
            f.write(f"{y}\n")

    pd.DataFrame({"p_svante": prob, "label": out_labels}).to_csv(
        args.out_probs, index=False,
    )

    print(f"Wrote: {args.out}")
    print(f"Wrote: {args.out_probs}")


if __name__ == "__main__":
    main()
