"""
Split the labelled interim dataset CSV into train / test sets.

Stratified splitting is used when both classes have at least two samples,
ensuring the Svante-to-others ratio is preserved in each split.

Usage:

    python scripts/split_dataset.py \\
        --dataset_csv data/interim/dataset.csv \\
        --out_dir data/processed
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from svantecatch.utils import ensure_dir


def main() -> None:
    """Parse CLI args, load the labelled CSV, and write train/test splits."""
    p = argparse.ArgumentParser(description="Split labeled dataset.csv into train and test sets.")
    p.add_argument("--dataset_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ensure_dir(args.out_dir)

    df = pd.read_csv(args.dataset_csv)
    if "user_id" not in df.columns:
        raise ValueError("dataset_csv must include user_id for splitting")

    # Binary label: 1 = Svante (user_id 0), 0 = everyone else
    y = (df["user_id"] == 0).astype(int)

    # Stratify only when both classes have enough samples for the split
    stratify = None
    if y.sum() >= 2 and (len(y) - y.sum()) >= 2:
        stratify = y

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )

    train_path = os.path.join(args.out_dir, "train.csv")
    test_path = os.path.join(args.out_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Wrote: {train_path} (rows={len(train_df)})")
    print(f"Wrote: {test_path} (rows={len(test_df)})")
    print(f"Svante in train: {int((train_df['user_id']==0).sum())}")
    print(f"Svante in test:  {int((test_df['user_id']==0).sum())}")

if __name__ == "__main__":
    main()
