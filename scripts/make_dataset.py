"""
Convert raw JSON session files into interim CSVs.

Usage:

    python scripts/make_dataset.py \\
        --train_json data/raw/dataset.json \\
        --verify_json data/raw/verify.json \\
        --out_dir data/interim
"""

import argparse
import os

from svantecatch.convert import json_to_raw_csv
from svantecatch.utils import ensure_dir


def main() -> None:
    """Parse CLI args and convert both the labelled and verification JSONs."""
    p = argparse.ArgumentParser(description="Convert JSON datasets to raw CSV files.")
    p.add_argument("--train_json", required=True, help="Path to labeled dataset.json")
    p.add_argument("--verify_json", required=True, help="Path to unlabeled verify.json")
    p.add_argument("--out_dir", required=True, help="Output dir, e.g. data/interim")
    args = p.parse_args()

    ensure_dir(args.out_dir)

    dataset_csv = os.path.join(args.out_dir, "dataset.csv")
    verify_csv = os.path.join(args.out_dir, "verify.csv")

    json_to_raw_csv(args.train_json, dataset_csv)
    json_to_raw_csv(args.verify_json, verify_csv)

    print(f"Wrote: {dataset_csv}")
    print(f"Wrote: {verify_csv}")

if __name__ == "__main__":
    main()
