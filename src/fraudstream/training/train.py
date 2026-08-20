import sys
from pathlib import Path

CSV_PATH = Path("data/raw/paysim.csv")


def main() -> None:
    if not CSV_PATH.exists():
        msg = (
            f"Dataset not found at {CSV_PATH}.\n"
            "Download with:\n"
            "  kaggle datasets download -d ealaxi/paysim1 --path data/raw/ --unzip"
        )
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
