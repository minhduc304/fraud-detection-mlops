import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from fraudstream.features.transforms import build_features
from fraudstream.training.evaluate import report
from fraudstream.training.register import save_model

CSV_PATH = Path("data/raw/paysim.csv")


def _load_and_split(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    split_step = df["step"].quantile(0.8)
    train = df[df["step"] <= split_step].reset_index(drop=True)
    test = df[df["step"] > split_step].reset_index(drop=True)
    return train, test


def main() -> None:
    if not CSV_PATH.exists():
        msg = (
            f"Dataset not found at {CSV_PATH}.\n"
            "Download with:\n"
            "  kaggle datasets download -d ealaxi/paysim1 --path data/raw/ --unzip"
        )
        print(msg)
        sys.exit(1)

    print("Loading data...")
    train_df, test_df = _load_and_split(CSV_PATH)
    print(f"Train: {len(train_df):,} rows  Test: {len(test_df):,} rows")

    X_train = build_features(train_df)
    y_train = train_df["isFraud"]
    X_test = build_features(test_df)
    y_test = test_df["isFraud"]

    scale = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"scale_pos_weight: {scale}")

    print("\nTraining logistic regression...")
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    lr_scores = pd.Series(lr.predict_proba(X_test)[:, 1])
    lr_report = report("logistic_regression", y_test, lr_scores)

    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,
        eval_metric="aucpr",
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_scores = pd.Series(xgb.predict_proba(X_test)[:, 1])
    xgb_report = report("xgboost", y_test, xgb_scores)

    print("\n--- Results ---")
    header = f"{'Model':<22} {'PR-AUC':>8} {'P@R80':>8}"
    print(header)
    print("-" * len(header))
    for r in [lr_report, xgb_report]:
        print(f"{r['model']:<22} {r['pr_auc']:>8.4f} {r['precision_at_recall_80']:>8.4f}")

    winner_name = max([lr_report, xgb_report], key=lambda r: r["pr_auc"])["model"]
    winner_model = xgb if winner_name == "xgboost" else lr
    print(f"\nWinner: {winner_name}")

    model_path = Path("models/model.pkl")
    save_model(winner_model, model_path)
    print(f"Saved to {model_path}")


if __name__ == "__main__":
    main()
