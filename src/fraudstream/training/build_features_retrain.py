"""Airflow-only: chunked split+featurize, avoids loading full CSV into memory.

Mirrors prepare.py + featurize.py's split/feature logic, but reads data/raw/paysim.csv
in chunks (never the full 6.3M-row DataFrame at once) so it's safe to run as a
KubernetesPodOperator with normal pod memory limits.
"""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from fraudstream.features.transforms import build_features

RAW_PATH = Path("data/raw/paysim.csv")
PARAMS_PATH = Path("params.yaml")
OUT_DIR = Path("data/processed")
CHUNK_SIZE = 100_000


def compute_split_step(csv_path: Path, ratio: float) -> float:
    """Light pass: only the `step` column, to get the same split point prepare.py computes."""
    steps = pd.read_csv(csv_path, usecols=["step"])["step"]
    return steps.quantile(ratio)


def _write_chunk(
    writer: pq.ParquetWriter | None, df: pd.DataFrame, out_path: Path
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema)
    writer.write_table(table)
    return writer


def split_and_featurize(
    csv_path: Path, split_step: float, out_dir: Path, chunksize: int = CHUNK_SIZE
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter | None] = {
        "X_train": None,
        "y_train": None,
        "X_test": None,
        "y_test": None,
    }
    try:
        for chunk in pd.read_csv(csv_path, chunksize=chunksize):
            for name, df in (
                ("train", chunk[chunk["step"] <= split_step]),
                ("test", chunk[chunk["step"] > split_step]),
            ):
                if df.empty:
                    continue
                X = build_features(df)
                y = df[["isFraud"]]
                writers[f"X_{name}"] = _write_chunk(
                    writers[f"X_{name}"], X, out_dir / f"X_{name}.parquet"
                )
                writers[f"y_{name}"] = _write_chunk(
                    writers[f"y_{name}"], y, out_dir / f"y_{name}.parquet"
                )
    finally:
        for writer in writers.values():
            if writer is not None:
                writer.close()


def main() -> None:
    params = yaml.safe_load(PARAMS_PATH.read_text())
    ratio = params["split"]["ratio"]

    split_step = compute_split_step(RAW_PATH, ratio)
    print(f"Split step: {split_step}")

    split_and_featurize(RAW_PATH, split_step, OUT_DIR)
    print(f"Features written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
