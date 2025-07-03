#!/usr/bin/env python3
# src/data_preprocessing.py

"""
Data Preprocessing Script for RAG Complaint Analysis Project

- Loads CFPB complaints data
- Performs basic EDA summaries
- Filters data for target products
- Cleans complaint narratives
- Saves filtered dataset for downstream RAG pipeline
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# -------------------------------------------
# CONFIG
# -------------------------------------------

RAW_DATA_PATH = "data/raw/complaints.csv"
OUTPUT_PATH = "data/interim/filtered_complaints.csv"

TARGET_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later (BNPL)",
    "Savings account",
    "Money transfers",
]

# -------------------------------------------
# FUNCTIONS
# -------------------------------------------


def load_data(path):
    """Load CFPB dataset from CSV"""
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Data shape: {df.shape}")
    return df


def plot_product_distribution(df):
    """Visualize number of complaints by product"""
    plt.figure(figsize=(10, 5))
    sns.countplot(
        y="Product",
        data=df,
        order=df["Product"].value_counts().index,
        palette="Blues_d",
    )
    plt.title("Complaint Distribution Across Products")
    plt.xlabel("Number of Complaints")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.savefig("reports/product_distribution.png")
    plt.close()
    print("Saved plot: reports/product_distribution.png")


def analyze_narrative_lengths(df):
    """Calculate and plot narrative lengths"""
    df_narratives = df[df["Consumer complaint narrative"].notna()].copy()
    df_narratives["narrative_length"] = df_narratives[
        "Consumer complaint narrative"
    ].apply(lambda x: len(str(x).split()))

    # Basic stats
    desc = df_narratives["narrative_length"].describe()
    print("Narrative length stats:\n", desc)

    # Plot histogram
    plt.figure(figsize=(10, 5))
    sns.histplot(df_narratives["narrative_length"], bins=50, kde=True, color="purple")
    plt.title("Distribution of Narrative Lengths")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("reports/narrative_length_distribution.png")
    plt.close()
    print("Saved plot: reports/narrative_length_distribution.png")


def count_narrative_presence(df):
    """Count complaints with and without narratives"""
    n_with = df["Consumer complaint narrative"].notna().sum()
    n_without = df["Consumer complaint narrative"].isna().sum()

    print(f"Complaints WITH narrative: {n_with}")
    print(f"Complaints WITHOUT narrative: {n_without}")


def filter_data(df, target_products):
    """Filter dataset for target products and non-empty narratives"""
    filtered = df[
        df["Product"].isin(target_products) & df["Consumer complaint narrative"].notna()
    ].copy()

    print(f"Filtered dataset shape: {filtered.shape}")
    return filtered


def clean_text(text):
    """Perform basic cleaning on complaint narrative"""
    text = text.lower()
    # Remove common boilerplate intro patterns
    text = re.sub(r"i am writing.*?complaint", "", text)
    text = re.sub(r"i want to.*?complaint", "", text)
    # Remove non-alphanumeric characters except basic punctuation
    text = re.sub(r"[^a-z0-9\s.,]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_narratives(df):
    """Apply cleaning to the narrative text"""
    df["Cleaned Narrative"] = df["Consumer complaint narrative"].apply(clean_text)
    return df


def save_filtered_data(df, path):
    """Save cleaned dataframe to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved cleaned data → {path}")


# -------------------------------------------
# MAIN SCRIPT
# -------------------------------------------

if __name__ == "__main__":
    # Load data
    df = load_data(RAW_DATA_PATH)

    # EDA
    plot_product_distribution(df)
    analyze_narrative_lengths(df)
    count_narrative_presence(df)

    # Filtering
    df_filtered = filter_data(df, TARGET_PRODUCTS)

    # Clean narratives
    df_cleaned = clean_narratives(df_filtered)

    # Save output
    save_filtered_data(df_cleaned, OUTPUT_PATH)

    print("✅ Data preprocessing complete.")
