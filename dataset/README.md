# Dataset Documentation & Data Lineage

This directory contains raw source datasets and the unified, cleaned news categorization dataset for the **Intelligent News Categorization System**.

---

## 1. Target Classes (Exactly 5 Categories)

All articles are mapped to one of the following canonical labels:
1. `Politics`
2. `Business`
3. `Technology`
4. `Sports`
5. `Entertainment`

---

## 2. Integrated Data Sources

### A. BBC News Full-Text Classification Dataset
- **Authors:** Derek Greene and Pádraig Cunningham.
- **Citation:** Greene, D., & Cunningham, P. (2006). "Practical Solutions to the Problem of Diagonal Dominance in High-Dimensional Low-Sample-Size Data". *Proceedings of the 23rd International Conference on Machine Learning (ICML 2006)*.
- **Source URL:** http://mlg.ucd.ie/datasets/bbc.html | http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip
- **License:** Non-commercial academic and research use.
- **Original Format:** 2,225 individual raw `.txt` files in 5 topic directories (`business`, `entertainment`, `politics`, `sport`, `tech`).
- **Target Category Mapping:**
  - `business` -> `Business`
  - `entertainment` -> `Entertainment`
  - `politics` -> `Politics`
  - `sport` -> `Sports`
  - `tech` -> `Technology`
- **Text Characteristics:** Full-length news articles (Mean length: 384 words; range: 89 - 4,432 words).
- **Post-cleaning retained:** 2,122 articles.

### B. HuffPost News Category Dataset
- **Author:** Rishabh Misra.
- **Citation:** Misra, Rishabh. "News Category Dataset." *arXiv preprint arXiv:2205.10512 (2022)*.
- **Source URL:** https://www.kaggle.com/datasets/rishabhms/news-category-dataset
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Original Format:** JSON lines containing 209,527 articles across 42 categories from 2012 to 2022.
- **Target Category Mapping:**
  - `POLITICS` -> `Politics`
  - `BUSINESS` -> `Business`
  - `TECH` -> `Technology`
  - `SPORTS` -> `Sports`
  - `ENTERTAINMENT` -> `Entertainment`
- **Explicit Exclusions:**
  - `WORLD NEWS`: Strictly excluded to prevent general world events from being erroneously conflated with domestic politics.
  - All lifestyle, humor, regional, and non-target classes (e.g., `COMEDY`, `CRIME`, `QUEER VOICES`, `TRAVEL`) are dropped.
- **Text Characteristics:** Short-form text comprising headlines and short descriptions (`headline + " - " + short_description`). Mean length: 27 words.
- **Sampling & Post-cleaning retained:** 9,950 articles (~1,990 per class).

---

## 3. Dataset Integration & Preprocessing Pipeline

1. **Whitespace Normalization:** Redundant whitespace characters (`\n`, `\t`, multiple spaces) collapsed to single spaces.
2. **Missing Value Filtering:** Articles with text length ≤ 15 characters or missing values discarded.
3. **Exact Deduplication:** 144 exact duplicates identified and removed.
4. **Near-Deduplication:** 3 near-duplicates identified via normalized lowercase alphanumeric matching and removed.
5. **Class Balancing:** To prevent severe class imbalance (where Politics would otherwise comprise >50% of the dataset), stratified sampling from HuffPost was matched with BBC full-length articles, yielding a balanced ~2,300 to 2,500 articles per category.

---

## 4. Final Cleaned Dataset Specification

- **File Path:** `dataset/processed/news_dataset_cleaned.csv`
- **Total Rows:** 12,072
- **Columns:**
  - `text` (string): The news article body or headline+description.
  - `category` (string): One of the 5 canonical categories.
  - `source` (string): Provenance tracking (`BBC News` or `HuffPost`).
  - `text_type` (string): `full_article` or `headline_summary`.
  - `word_count` (int): Number of whitespace-separated tokens.
  - `char_count` (int): Total character count.
