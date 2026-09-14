"""
data_prep.py
Loads the raw Kaggle "Customer Support Ticket Dataset" CSV, cleans it,
and produces train/val/test splits for:
  - category classification (Ticket Type)
  - urgency classification  (Ticket Priority)

Usage:
    python src/data_prep.py
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/tickets_raw.csv"
OUT_DIR = "data/processed"

# Columns we actually need from the raw file
TEXT_COLS = ["Ticket Subject", "Ticket Description"]
PRODUCT_COL = "Product Purchased"
CATEGORY_COL = "Ticket Type"
URGENCY_COL = "Ticket Priority"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows missing anything essential
    required = TEXT_COLS + [PRODUCT_COL, CATEGORY_COL, URGENCY_COL]
    df = df.dropna(subset=required).copy()

    # This dataset often has a literal "{product_purchased}" placeholder
    # in the description instead of the real product name — replace it.
    def fill_placeholder(row):
        desc = str(row["Ticket Description"])
        product = str(row[PRODUCT_COL])
        return desc.replace("{product_purchased}", product)

    df["Ticket Description"] = df.apply(fill_placeholder, axis=1)

    # Combine subject + description into a single input text
    df["ticket_text"] = (
        df["Ticket Subject"].astype(str) + ". " + df["Ticket Description"].astype(str)
    )

    # Basic cleaning: lowercase, collapse whitespace/newlines, strip
    def basic_clean(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\s+", " ", text)  # collapse newlines/multiple spaces
        text = text.strip()
        return text

    df["ticket_text"] = df["ticket_text"].apply(basic_clean)

    # Rename label columns to simple, consistent names
    df = df.rename(columns={CATEGORY_COL: "category", URGENCY_COL: "urgency"})

    # Keep only what downstream steps need
    df = df[["ticket_text", "category", "urgency"]]

    # Drop any empty text rows post-cleaning
    df = df[df["ticket_text"].str.len() > 0]

    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15, seed=42):
    # Stratify by category so class balance is preserved across splits
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["category"]
    )
    relative_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val_size,
        random_state=seed,
        stratify=train_val["category"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main():
    df = load_data(RAW_PATH)
    print(f"Loaded raw data: {df.shape[0]} rows")

    df = clean_text(df)
    print(f"After cleaning: {df.shape[0]} rows")
    print("\nCategory distribution:\n", df["category"].value_counts())
    print("\nUrgency distribution:\n", df["urgency"].value_counts())

    train, val, test = split_data(df)
    print(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    train.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val.to_csv(f"{OUT_DIR}/val.csv", index=False)
    test.to_csv(f"{OUT_DIR}/test.csv", index=False)
    print(f"\nSaved processed splits to {OUT_DIR}/")


if __name__ == "__main__":
    main()