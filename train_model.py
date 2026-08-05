"""
train_model.py — first-attempt classifier: who subscribes (y)?

A clean Logistic Regression baseline that puts into practice what the EDA showed:
  - drops 'duration' (leakage — known only after the call ends).
  - turns pdays==999 ("never contacted before") into a clean binary flag.
  - one-hot encodes categoricals, scales numerics — all inside a Pipeline.
  - handles the 11% imbalance with class_weight="balanced".
  - reports the RIGHT metrics (ROC-AUC, PR-AUC, precision/recall), not accuracy.

Usage:
    python train_model.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
)

# --- Settings (edit here) ---
DATA_PATH = Path(
    r"C:\Users\HDTeam\PycharmProjects\bank_marketing\src\data\bank-additional-full.csv"
)
SEP = ";"
TARGET = "y"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Where evaluation plots are saved
PLOT_DIR = Path(
    r"C:\Users\HDTeam\PycharmProjects\bank_marketing\src\back_marketing"
) / "model_plots"
SHOW_PLOTS = True   # open chart windows too (set False to only save)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


def load_and_prepare() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=SEP)

    # 1) Drop the leakage feature: duration is only known after the call.
    if "duration" in df.columns:
        df = df.drop(columns="duration")

    # 2) pdays==999 means "never contacted before" — a flag, not a real number.
    if "pdays" in df.columns:
        df["was_contacted_before"] = (df["pdays"] != 999).astype(int)
        df["pdays"] = df["pdays"].replace(999, 0)  # neutralize the sentinel value

    return df


def split_xy(df: pd.DataFrame):
    # Encode the target to 0/1 and separate features
    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    X = df.drop(columns=[TARGET])
    return X, y


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Auto-detect column types
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    # Preprocess numerics (scale) and categoricals (one-hot) together
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])

    # Logistic Regression baseline; balanced weights counter the 11% imbalance
    model = LogisticRegression(max_iter=1000, class_weight="balanced")

    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(pipe: Pipeline, X_test, y_test) -> None:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]  # probability of "yes"

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX  (rows=true, cols=pred)")
    print("=" * 60)
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["true_no", "true_yes"], columns=["pred_no", "pred_yes"],
    ))

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=3))

    print("=" * 60)
    print("RANKING METRICS (the ones that matter with imbalance)")
    print("=" * 60)
    print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.3f}")
    print(f"PR-AUC  : {average_precision_score(y_test, y_proba):.3f}")
    print(f"(baseline PR-AUC ≈ positive rate = {y_test.mean():.3f})")


def show_top_features(pipe: Pipeline, top: int = 15) -> None:
    # Which features push toward "yes"? Read the logistic-regression coefficients.
    feat_names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    s = pd.Series(coefs, index=feat_names).sort_values(key=abs, ascending=False)

    print("\n" + "=" * 60)
    print(f"TOP {top} FEATURES BY |coefficient|  (+ = pushes toward yes)")
    print("=" * 60)
    print(s.head(top).round(3))


def save_plots(pipe: Pipeline, X_test, y_test) -> None:
    # Save ROC curve, Precision-Recall curve and confusion matrix as PNGs
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # 1) ROC curve — true-positive vs false-positive rate across thresholds
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1)  # random-guess reference
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "roc_curve.png", dpi=120)

    # 2) Precision-Recall curve — the key view under class imbalance
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.axhline(y_test.mean(), color="red", ls="--",
               label=f"baseline {y_test.mean():.3f}")  # positive rate
    ax.set_title("Precision-Recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pr_curve.png", dpi=120)

    # 3) Confusion matrix at the default 0.5 threshold
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pipe.predict(X_test), display_labels=["no", "yes"],
        cmap="Blues", ax=ax,
    )
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confusion_matrix.png", dpi=120)

    print(f"\nPlots saved to: {PLOT_DIR}")
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")


def main():
    df = load_and_prepare()
    X, y = split_xy(df)
    print(f"Data: {X.shape[0]:,} rows x {X.shape[1]} features | positive rate = {y.mean():.3f}")

    # Stratify keeps the ~11% positive rate identical in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    pipe = build_pipeline(X)
    pipe.fit(X_train, y_train)

    evaluate(pipe, X_test, y_test)
    show_top_features(pipe)
    save_plots(pipe, X_test, y_test)


if __name__ == "__main__":
    main()
