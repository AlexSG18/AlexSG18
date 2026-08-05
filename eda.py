"""
eda.py — understand the data before modeling.

Usage:
    python eda.py --data path/to/data.csv --target loan

Column-agnostic: auto-detects numeric vs categorical and prints a full overview.
"""

import argparse
import pandas as pd


def load_data(path: str, sep: str) -> pd.DataFrame:
    # Read CSV with the given separator (UCI bank uses ';')
    df = pd.read_csv(path, sep=sep)
    return df


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


def split_columns(df: pd.DataFrame, target: str):
    # Separate features into numeric vs categorical (target excluded)
    features = [c for c in df.columns if c != target]
    numeric = df[features].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def target_report(df: pd.DataFrame, target: str) -> None:
    # Class distribution + imbalance warning (key for loan/subscription problems)
    if target not in df.columns:
        print(f"\n[!] target column '{target}' not found — skipping target report.")
        return
    print("\n" + "=" * 60)
    print(f"TARGET: '{target}'")
    print("=" * 60)
    counts = df[target].value_counts(dropna=False)
    pct = df[target].value_counts(normalize=True, dropna=False).mul(100).round(2)
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
    with pd.option_context("display.max_columns", None, "display.width", 200):
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


def target_correlation(df: pd.DataFrame, numeric, target: str) -> None:
    # Correlation of numeric features with the target (binary/numeric only)
    if target not in df.columns or not numeric:
        return
    y = df[target]
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
    # CLI args: data path, target column name, CSV separator
    parser = argparse.ArgumentParser(description="EDA for bank loan-prediction data")
    parser.add_argument("--data", required=True, help="path to CSV file")
    parser.add_argument("--target", default="loan", help="name of the target column")
    parser.add_argument("--sep", default=",", help="CSV separator (UCI bank uses ';')")
    args = parser.parse_args()

    df = load_data(args.data, args.sep)
    overview(df)

    numeric, categorical = split_columns(df, args.target)
    print(f"\nDetected {len(numeric)} numeric and {len(categorical)} categorical features.")

    target_report(df, args.target)
    numeric_report(df, numeric)
    categorical_report(df, categorical)
    target_correlation(df, numeric, args.target)


if __name__ == "__main__":
    main()
