"""
Dataset Download, Integration, Cleaning, and Analysis Pipeline
Intelligent News Categorization System - Phase 1

This script:
1. Downloads and extracts the BBC News Full-Text dataset (Greene & Cunningham, 2006).
2. Fetches and filters the HuffPost News Category Dataset (Misra, 2022).
3. Maps source labels to the canonical 5 classes:
   - Politics
   - Business
   - Technology
   - Sports
   - Entertainment
4. Cleans text, removes duplicates and near-duplicates, handles missing data.
5. Merges into a unified, balanced dataset with 'text', 'category', 'source', 'text_type'.
6. Computes comprehensive statistics and saves analysis visualizations.
"""

import os
import re
import urllib.request
import zipfile
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure output directories exist
os.makedirs("dataset/raw", exist_ok=True)
os.makedirs("dataset/processed", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)

# Canonical 5 categories
CANONICAL_CATEGORIES = ["Politics", "Business", "Technology", "Sports", "Entertainment"]

def download_file(url, target_path):
    print(f"Downloading {url} to {target_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, "wb") as f:
        f.write(resp.read())
    print("Download complete.")

def load_bbc_news():
    """
    Downloads and parses the primary BBC News dataset from University College Dublin.
    License: Academic/Research non-commercial.
    Returns DataFrame with columns ['text', 'category', 'source', 'text_type'].
    """
    zip_path = "dataset/raw/bbc-fulltext.zip"
    extract_dir = "dataset/raw/bbc"
    
    if not os.path.exists(zip_path):
        url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
        download_file(url, zip_path)
        
    if not os.path.exists(extract_dir):
        print("Extracting BBC fulltext zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("dataset/raw")
            
    # Source mapping
    bbc_category_map = {
        "business": "Business",
        "entertainment": "Entertainment",
        "politics": "Politics",
        "sport": "Sports",
        "tech": "Technology"
    }
    
    records = []
    base_folder = os.path.join("dataset", "raw", "bbc")
    
    for cat_folder, canonical_cat in bbc_category_map.items():
        folder_path = os.path.join(base_folder, cat_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.endswith(".txt"):
                file_path = os.path.join(folder_path, fname)
                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read().strip()
                        if content:
                            records.append({
                                "text": content,
                                "category": canonical_cat,
                                "source": "BBC News",
                                "text_type": "full_article"
                            })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
    df_bbc = pd.DataFrame(records)
    print(f"Loaded BBC News: {len(df_bbc)} articles across {df_bbc['category'].nunique()} categories.")
    df_bbc.to_csv("dataset/raw/bbc_news_raw.csv", index=False)
    return df_bbc

def load_huffpost_sample(target_samples_per_class=2000, random_state=42):
    """
    Loads HuffPost News Category Dataset from HuggingFace datasets cache or direct API.
    Filters specifically for the 5 target categories:
    - POLITICS -> Politics
    - BUSINESS -> Business
    - TECH -> Technology
    - SPORTS -> Sports
    - ENTERTAINMENT -> Entertainment
    (Explicitly drops WORLD NEWS and other categories to avoid mislabeling).
    Combines headline and short_description.
    """
    from datasets import load_dataset
    print("Loading HuffPost News Category Dataset...")
    ds = load_dataset("heegyu/news-category-dataset", split="train")
    
    huffpost_category_map = {
        "POLITICS": "Politics",
        "BUSINESS": "Business",
        "TECH": "Technology",
        "SPORTS": "Sports",
        "ENTERTAINMENT": "Entertainment"
    }
    
    records = []
    for row in ds:
        cat = row.get("category", "")
        if cat in huffpost_category_map:
            headline = (row.get("headline") or "").strip()
            desc = (row.get("short_description") or "").strip()
            
            # Combine headline and short description
            if headline and desc:
                full_text = f"{headline} - {desc}"
            elif headline:
                full_text = headline
            elif desc:
                full_text = desc
            else:
                continue
                
            records.append({
                "text": full_text,
                "category": huffpost_category_map[cat],
                "source": "HuffPost",
                "text_type": "headline_summary"
            })
            
    df_huffpost_raw = pd.DataFrame(records)
    print(f"Total matching HuffPost records: {len(df_huffpost_raw)}")
    print("Raw HuffPost counts per class:")
    print(df_huffpost_raw["category"].value_counts())
    
    # Save raw filtered data
    df_huffpost_raw.to_csv("dataset/raw/huffpost_news_raw.csv", index=False)
    
    # Stratified balanced sampling per category to prevent severe imbalance
    # Tech has the minimum around 2,050 samples in HuffPost
    min_available = df_huffpost_raw["category"].value_counts().min()
    sample_size = min(target_samples_per_class, min_available)
    print(f"Sampling {sample_size} articles per category from HuffPost (Total: {sample_size * 5})...")
    
    balanced_huffpost = df_huffpost_raw.groupby("category", group_keys=False).apply(
        lambda x: x.sample(n=sample_size, random_state=random_state)
    ).reset_index(drop=True)
    
    return balanced_huffpost

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Normalize excessive whitespaces, newlines, tabs
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def integrate_and_clean_dataset():
    # 1. Load BBC
    df_bbc = load_bbc_news()
    
    # 2. Load HuffPost balanced
    df_huffpost = load_huffpost_sample(target_samples_per_class=2000, random_state=42)
    
    # 3. Concatenate
    df = pd.concat([df_bbc, df_huffpost], ignore_index=True)
    raw_total = len(df)
    print(f"\nTotal combined initial articles: {raw_total}")
    
    # 4. Cleaning
    df['text'] = df['text'].apply(clean_text)
    
    # Remove missing/empty
    df = df[df['text'].str.len() > 15].copy()
    df = df[df['category'].isin(CANONICAL_CATEGORIES)].copy()
    
    # 5. Deduplication
    # Exact text duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    exact_dupes_removed = initial_count - len(df)
    
    # Near-duplicates (case-insensitive & stripped punctuation)
    df['text_norm'] = df['text'].str.lower().apply(lambda s: re.sub(r'[^a-z0-9 ]', '', s))
    norm_count = len(df)
    df = df.drop_duplicates(subset=['text_norm']).reset_index(drop=True)
    near_dupes_removed = norm_count - len(df)
    df = df.drop(columns=['text_norm'])
    
    print(f"Exact duplicates removed: {exact_dupes_removed}")
    print(f"Near duplicates removed: {near_dupes_removed}")
    print(f"Cleaned dataset total articles: {len(df)}")
    
    # Calculate word and char counts
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text'].apply(len)
    
    # Save cleaned dataset
    # Columns required by user: text, category, and source for auditing
    save_cols = ['text', 'category', 'source', 'text_type', 'word_count', 'char_count']
    cleaned_csv_path = "dataset/processed/news_dataset_cleaned.csv"
    df[save_cols].to_csv(cleaned_csv_path, index=False)
    print(f"Saved cleaned dataset to: {cleaned_csv_path}")
    
    return df

def generate_analysis_and_charts(df):
    print("\n=== DATASET ANALYSIS & METRICS ===")
    total_articles = len(df)
    print(f"Total articles: {total_articles}")
    
    category_counts = df['category'].value_counts()
    print("\nArticles per Category:")
    for cat, count in category_counts.items():
        pct = (count / total_articles) * 100
        print(f"  - {cat:15s}: {count:5d} ({pct:.2f}%)")
        
    source_counts = df['source'].value_counts()
    print("\nArticles per Source:")
    for src, count in source_counts.items():
        pct = (count / total_articles) * 100
        print(f"  - {src:15s}: {count:5d} ({pct:.2f}%)")
        
    print("\nSource vs Category Breakdown:")
    ct = pd.crosstab(df['category'], df['source'], margins=True)
    print(ct)
    
    print("\nText Length Statistics (Words):")
    word_stats = df['word_count'].describe()
    print(word_stats)
    
    print("\nText Length Statistics (Characters):")
    char_stats = df['char_count'].describe()
    print(char_stats)
    
    print("\nText Length by Source:")
    print(df.groupby('source')[['word_count', 'char_count']].agg(['mean', 'median', 'std', 'min', 'max']))
    
    # Visualizations
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Class Distribution Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    palette = sns.color_palette("deep", len(CANONICAL_CATEGORIES))
    sns.countplot(data=df, x='category', order=CANONICAL_CATEGORIES, palette=palette, ax=axes[0])
    axes[0].set_title("Overall Class Distribution (5 Target Classes)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Category", fontsize=11)
    axes[0].set_ylabel("Article Count", fontsize=11)
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=10)
                         
    # Stacked/Grouped by Source
    sns.countplot(data=df, x='category', hue='source', order=CANONICAL_CATEGORIES, palette="Set2", ax=axes[1])
    axes[1].set_title("Class Distribution by Source (BBC News vs HuffPost)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Category", fontsize=11)
    axes[1].set_ylabel("Article Count", fontsize=11)
    axes[1].legend(title="Source")
    
    plt.tight_layout()
    chart1_path = "evaluation/class_distribution.png"
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"Saved class distribution chart to {chart1_path}")
    
    # 2. Text Length Distribution Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Word count distribution by source
    sns.histplot(data=df[df['word_count'] < 1000], x='word_count', hue='source', kde=True, bins=40, palette="tab10", ax=axes[0])
    axes[0].set_title("Word Count Distribution (Truncated at 1000 words for clarity)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Word Count", fontsize=11)
    axes[0].set_ylabel("Article Count", fontsize=11)
    
    # Boxplot of word count across categories
    sns.boxplot(data=df, x='category', y='word_count', order=CANONICAL_CATEGORIES, palette="pastel", showfliers=False, ax=axes[1])
    axes[1].set_title("Word Count Range across Categories (without outliers)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Category", fontsize=11)
    axes[1].set_ylabel("Word Count", fontsize=11)
    
    plt.tight_layout()
    chart2_path = "evaluation/text_length_distribution.png"
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved text length distribution chart to {chart2_path}")

    # Generate Samples Summary
    samples = {}
    for cat in CANONICAL_CATEGORIES:
        sample_rows = df[df['category'] == cat].sample(2, random_state=42)
        samples[cat] = [
            {"source": row['source'], "text": row['text'][:250] + ("..." if len(row['text']) > 250 else "")}
            for _, row in sample_rows.iterrows()
        ]
        
    return {
        "total_articles": total_articles,
        "category_counts": category_counts.to_dict(),
        "source_counts": source_counts.to_dict(),
        "word_stats": word_stats.to_dict(),
        "char_stats": char_stats.to_dict(),
        "samples": samples
    }

if __name__ == "__main__":
    print("Starting Phase 1 Dataset Pipeline...")
    df_cleaned = integrate_and_clean_dataset()
    stats = generate_analysis_and_charts(df_cleaned)
    print("\nPhase 1 Data Pipeline completed successfully!")
