"""
eda_plots.py — visualize the trends in the data.

Produces four kinds of charts that tell the story of the dataset:
  1. Target balance          — how rare are subscribers (~11%).
  2. Conversion rate by category — which groups say "yes" more than baseline.
  3. Numeric features by target  — how distributions differ for yes vs no.
  4. Correlation heatmap        — how numeric features relate to each other + target.

Charts are shown on screen and saved as PNG files under ./eda_plots/.

Usage:
    python eda_plots.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Settings (edit here) ---
DATA_PATH = Path(
    r"C:\Users\HDTeam\PycharmProjects\bank_marketing\src\data\bank-additional-full.csv"
)
SEP = ";"
TARGET = "y"
POSITIVE = "yes"
OUT_DIR = Path(__file__).resolve().parent / "eda_plots"  # where PNGs are saved
SHOW = True      # open chart windows (set False to only save files)
SAVE = True      # save PNG files

sns.set_theme(style="whitegrid")


def load() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=SEP)


def _finish(fig, name: str) -> None:
    # Save and/or show a finished figure, then free memory
    fig.tight_layout()
    if SAVE:
        OUT_DIR.mkdir(exist_ok=True)
        fig.savefig(OUT_DIR / f"{name}.png", dpi=120)
        print(f"saved: {OUT_DIR / f'{name}.png'}")
    if SHOW:
        plt.show()
    plt.close(fig)


def plot_target_balance(df: pd.DataFrame) -> None:
    # Bar chart of the class counts + percentage labels
    counts = df[TARGET].value_counts()
    pct = df[TARGET].value_counts(normalize=True).mul(100).round(1)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{pct.iloc[i]}%", ha="center", va="bottom")
    ax.set_title("Target balance (y)")
    ax.set_ylabel("count")
    _finish(fig, "01_target_balance")


def plot_conversion_by_category(df: pd.DataFrame) -> None:
    # For each categorical column: bar of positive-rate per category vs baseline
    baseline = (df[TARGET] == POSITIVE).mean()
    cat_cols = (
        df.drop(columns=[TARGET])
        .select_dtypes(include=["object", "string", "category"])
        .columns
    )

    for col in cat_cols:
        rate = (
            df.assign(_pos=(df[TARGET] == POSITIVE))
            .groupby(col, observed=True)["_pos"]
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(7, 0.45 * len(rate) + 1.5))
        sns.barplot(x=rate.values, y=rate.index, ax=ax)
        ax.axvline(baseline * 100, color="red", ls="--",
                   label=f"baseline {baseline*100:.1f}%")  # overall positive rate
        ax.set_title(f"Conversion rate by '{col}'")
        ax.set_xlabel("subscribed (%)")
        ax.legend()
        _finish(fig, f"02_conv_{col}")


def plot_numeric_by_target(df: pd.DataFrame) -> None:
    # Boxplot of each numeric feature split by the target class
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
        ax.set_title(f"{col} by target")
        _finish(fig, f"03_num_{col}")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    # Correlation among numeric features + numeric-encoded target
    num = df.select_dtypes(include="number").copy()
    if not pd.api.types.is_numeric_dtype(df[TARGET]):
        num[TARGET] = (df[TARGET].astype(str).str.strip().str.lower()
                       .map({"yes": 1, "no": 0}))  # add target as 0/1 for context
    corr = num.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(1.0 * len(corr) + 2, 0.8 * len(corr) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Numeric correlation (incl. target)")
    _finish(fig, "04_correlation_heatmap")


def main():
    df = load()
    plot_target_balance(df)
    plot_conversion_by_category(df)
    plot_numeric_by_target(df)
    plot_correlation_heatmap(df)
    print("\nDone. Charts are in:", OUT_DIR)


if __name__ == "__main__":
    main()
