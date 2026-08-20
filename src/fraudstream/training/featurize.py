"""Stage 2: build features from interim parquet, write processed parquet."""
from pathlib import Path

import pandas as pd

from fraudstream.features.transforms import build_features

IN_DIR = Path("data/interim")
OUT_DIR = Path("data/processed")


def main() -> None:
    train_df = pd.read_parquet(IN_DIR / "train.parquet")
    test_df = pd.read_parquet(IN_DIR / "test.parquet")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    build_features(train_df).to_parquet(OUT_DIR / "X_train.parquet", index=False)
    train_df["isFraud"].to_frame().to_parquet(OUT_DIR / "y_train.parquet", index=False)
    build_features(test_df).to_parquet(OUT_DIR / "X_test.parquet", index=False)
    test_df["isFraud"].to_frame().to_parquet(OUT_DIR / "y_test.parquet", index=False)

    print(f"Features written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
