"""
eda.py — understand the data before modeling.

Loads the dataset directly (like the original script) and prints a full
overview: shape, dtypes, missing values, target balance, numeric/categorical
summaries and numeric-vs-target correlation.

Usage:
    python eda.py
"""

from pathlib import Path
import pandas as pd

# --- Settings (edit here) ---
DATA_PATH = Path(
    r"C:\Users\HDTeam\PycharmProjects\bank_marketing\src\data\bank-additional-full.csv"
)
SEP = ";"          # UCI bank file is semicolon-separated
TARGET = "y"       # target column: did the client subscribe

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


def load_data() -> pd.DataFrame:
    # Read the dataset from the fixed path above
    return pd.read_csv(DATA_PATH, sep=SEP)


def overview(df: pd.DataFrame) -> None:
    # High-level shape, dtypes, missing values and duplicates
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows  x  {df.shape[1]} columns")
    print("=" * 60)

    print("\n--- Dtypes ---")
    print(df.dtypes)

    print("\n--- Missing values (only columns with any) ---")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)  # keep only columns that have NaNs
    if miss.empty:
        print("No missing values 🎉")
    else:
        pct = (miss / len(df) * 100).round(2)
        print(pd.DataFrame({"missing": miss, "pct": pct}))

    print("\n--- Duplicated rows ---")
    print(f"{df.duplicated().sum():,}")


def split_columns(df: pd.DataFrame):
    # Separate features into numeric vs categorical (target excluded)
    features = [c for c in df.columns if c != TARGET]
    numeric = df[features].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def target_report(df: pd.DataFrame) -> None:
    # Class distribution + imbalance warning (key for loan/subscription problems)
    if TARGET not in df.columns:
        print(f"\n[!] target column '{TARGET}' not found — skipping target report.")
        return
    print("\n" + "=" * 60)
    print(f"TARGET: '{TARGET}'")
    print("=" * 60)
    counts = df[TARGET].value_counts(dropna=False)
    pct = df[TARGET].value_counts(normalize=True, dropna=False).mul(100).round(2)
    print(pd.DataFrame({"count": counts, "pct": pct}))
    if len(counts) == 2:  # binary target -> report minority share
        minority = pct.min()
        print(f"\nClass balance: minority class ≈ {minority:.1f}%")
        if minority < 20:
            print("⚠️  Strong imbalance — consider class_weight / SMOTE and look at "
                  "ROC-AUC / PR-AUC, not accuracy.")


def numeric_report(df: pd.DataFrame, numeric) -> None:
    # describe() for numeric features (transposed for readability)
    if not numeric:
        return
    print("\n" + "=" * 60)
    print("NUMERIC FEATURES — describe")
    print("=" * 60)
    print(df[numeric].describe().T)


def categorical_report(df: pd.DataFrame, categorical, top: int = 10) -> None:
    # Cardinality + most frequent values per categorical column
    if not categorical:
        return
    print("\n" + "=" * 60)
    print("CATEGORICAL FEATURES — cardinality & top values")
    print("=" * 60)
    for col in categorical:
        nun = df[col].nunique(dropna=False)
        print(f"\n[{col}]  unique={nun}")
        print(df[col].value_counts(dropna=False).head(top))


def target_correlation(df: pd.DataFrame, numeric) -> None:
    # Correlation of numeric features with the target (binary/numeric only)
    if TARGET not in df.columns or not numeric:
        return
    y = df[TARGET]
    if y.dtype == object:
        # Map a textual binary target (yes/no, true/false) to 0/1
        uniq = set(str(v).lower() for v in y.dropna().unique())
        mapping = None
        if uniq <= {"yes", "no"}:
            mapping = {"yes": 1, "no": 0}
        elif uniq <= {"true", "false"}:
            mapping = {"true": 1, "false": 0}
        if mapping is None:  # not a simple binary target -> skip
            return
        y = y.str.lower().map(mapping)
    print("\n" + "=" * 60)
    print("Correlation of numeric features with target")
    print("=" * 60)
    corr = df[numeric].apply(lambda c: c.corr(y)).sort_values(key=abs, ascending=False)
    print(corr.round(3))


def main():
    df = load_data()
    overview(df)

    numeric, categorical = split_columns(df)
    print(f"\nDetected {len(numeric)} numeric and {len(categorical)} categorical features.")

    target_report(df)
    numeric_report(df, numeric)
    categorical_report(df, categorical)
    target_correlation(df, numeric)


if __name__ == "__main__":
    main()
