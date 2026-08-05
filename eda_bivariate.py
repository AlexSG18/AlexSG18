"""
eda_bivariate.py — קשר בין הפיצ'רים ל-target (y).

זה השלב שאחרי ה-EDA הבסיסי: לא "מה יש בעמודה" אלא "איך העמודה מנבאת נרשמים".
מודד:
  - conversion rate לכל קטגוריה (אחוז y=yes) מול ה-baseline, כולל lift.
  - התפלגות פיצ'רים מספריים מפוצלת לפי y (ממוצע/חציון בכל מחלקה).

שימוש:
    python eda_bivariate.py
"""

from pathlib import Path
import pandas as pd

DATA_PATH = Path(
    r"C:\Users\HDTeam\PycharmProjects\bank_marketing\src\data\bank-additional-full.csv"
)
TARGET = "y"
POSITIVE = "yes"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


def load() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, sep=";")


def categorical_conversion(df: pd.DataFrame) -> None:
    baseline = (df[TARGET] == POSITIVE).mean()
    print("=" * 80)
    print(f"CONVERSION RATE BY CATEGORY   (baseline = {baseline*100:.2f}%)")
    print("=" * 80)

    cat_cols = (
        df.drop(columns=[TARGET])
        .select_dtypes(include=["object", "string", "category"])
        .columns
    )

    for col in cat_cols:
        grp = df.groupby(col, observed=True)[TARGET].agg(
            n="count",
            conv_rate=lambda s: (s == POSITIVE).mean(),
        )
        grp["conv_pct"] = (grp["conv_rate"] * 100).round(2)
        grp["lift_vs_base"] = (grp["conv_rate"] / baseline).round(2)
        grp = grp.drop(columns="conv_rate").sort_values("conv_pct", ascending=False)
        print(f"\n[{col}]")
        print(grp.to_string())


def numeric_by_target(df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("NUMERIC FEATURES BY TARGET  (mean / median per class)")
    print("=" * 80)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        print("No numeric columns.")
        return

    summary = df.groupby(TARGET)[num_cols].agg(["mean", "median"]).T
    print(summary.to_string())

    # רמז ל-leakage: פער עצום ב-duration בין המחלקות
    if "duration" in num_cols:
        print("\n⚠️  שים לב ל-'duration' — פער גדול בין yes/no הוא סימן לדליפה. "
              "להסיר לפני אימון מודל שמיועד לחיזוי אמיתי.")


def main():
    df = load()
    categorical_conversion(df)
    numeric_by_target(df)


if __name__ == "__main__":
    main()
