"""Stage 1: read raw CSV, split by step quantile, write interim parquet."""
import sys
from pathlib import Path

import pandas as pd
import yaml

CSV_PATH = Path("data/raw/paysim.csv")
PARAMS_PATH = Path("params.yaml")
OUT_DIR = Path("data/interim")


def main() -> None:
    if not CSV_PATH.exists():
        print(
            f"Dataset not found at {CSV_PATH}.\n"
            "Download with:\n"
            "  kaggle datasets download -d ealaxi/paysim1 --path data/raw/ --unzip"
        )
        sys.exit(1)

    params = yaml.safe_load(PARAMS_PATH.read_text())
    ratio = params["split"]["ratio"]

    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df):,} rows")

    split_step = df["step"].quantile(ratio)
    train = df[df["step"] <= split_step].reset_index(drop=True)
    test = df[df["step"] > split_step].reset_index(drop=True)
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT_DIR / "train.parquet", index=False)
    test.to_parquet(OUT_DIR / "test.parquet", index=False)
    print(f"Written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
