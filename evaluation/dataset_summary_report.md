# Dataset Exploration & Quality Summary Report
**Project:** Intelligent News Categorization System  
**Phase:** 1 — Project Setup and Dataset Collection  
**Generated Date:** November 2026  

---

## 1. Executive Summary

A unified, multi-source dataset was curated and cleaned for a 5-class news categorization task. The dataset combines full-length journalistic articles from **BBC News** with headline-and-summary pairs from **HuffPost**. This hybrid composition satisfies the core requirement to classify both **news headlines** and **complete news articles** into exactly five categories:
1. `Politics`
2. `Business`
3. `Technology`
4. `Sports`
5. `Entertainment`

The dataset is free of missing data, deduplicated, and thoroughly balanced across all five categories.

---

## 2. Dataset Overview & Key Metrics

| Metric | Value |
| :--- | :--- |
| **Total Cleaned Articles** | **12,072** |
| **Target Classes** | **5** (Politics, Business, Technology, Sports, Entertainment) |
| **Data Sources** | BBC News (17.58%), HuffPost (82.42%) |
| **Text Formats Covered** | Full-length articles + Headline/summary briefs |
| **Exact Duplicates Removed** | 144 |
| **Near-Duplicates Removed** | 3 |
| **Short/Empty Texts Removed** | All articles < 16 characters discarded |
| **Missing Values (text / category)** | 0 (0.00%) |

---

## 3. Class Distribution Analysis

| Category | Total Count | Percentage of Dataset | BBC News (Full Text) | HuffPost (Headlines) |
| :--- | :---: | :---: | :---: | :---: |
| **Sports** | 2,501 | 20.72% | 502 | 1,999 |
| **Business** | 2,500 | 20.71% | 503 | 1,997 |
| **Politics** | 2,402 | 19.90% | 403 | 1,999 |
| **Entertainment** | 2,368 | 19.62% | 369 | 1,999 |
| **Technology** | 2,301 | 19.06% | 345 | 1,956 |
| **Total** | **12,072** | **100.00%** | **2,122** | **9,950** |

### Class Imbalance Assessment & Strategy
- **Assessment:** Every class accounts for between **19.06% and 20.72%** of the dataset. This represents a nearly ideal balanced distribution (ideal 5-class uniform distribution is 20.00% per class).
- **Imbalance Ratio:** The ratio between the largest class (`Sports`: 2,501) and smallest class (`Technology`: 2,301) is just **1.087 : 1**, which requires no aggressive synthetic oversampling (e.g. SMOTE) that could distort natural language text semantics.
- **Recommended Strategy for Modeling (Phase 2 & 3):**
  1. Use **Stratified Train-Test Splitting** (`StratifiedKFold` / `train_test_split(..., stratify=y)`) to ensure identical proportions of both classes and sources across training, validation, and test sets.
  2. Evaluate using **Macro-averaged F1-Score** alongside Accuracy and Class-specific Precision/Recall to ensure performance parity across all 5 classes.

---

## 4. Text Length Distribution

The dataset possesses a bimodal length distribution by design, reflecting real-world conditions where the model must classify both concise news tickers/headlines and lengthy investigative reporting.

| Metric | Overall Dataset | BBC News (Full Articles) | HuffPost (Headlines/Summaries) |
| :--- | :---: | :---: | :---: |
| **Mean Word Count** | 90.09 words | 384.18 words | 27.37 words |
| **Median Word Count** | 29.00 words | 331.00 words | 26.00 words |
| **Standard Deviation (Words)**| 169.84 words | 241.64 words | 12.99 words |
| **Min Word Count** | 2 words | 89 words | 2 words |
| **Max Word Count** | 4,432 words | 4,432 words | 158 words |
| **Mean Character Count** | 530.12 chars | 2,234.90 chars | 167.14 chars |
| **Median Character Count** | 174.00 chars | 1,952.50 chars | 157.00 chars |

---

## 5. Duplicate and Data Hygiene Audit

- **Exact Duplicate Detection:** Duplicate matching on raw text identified **144** duplicate articles (predominantly identical wire releases republished across HuffPost dates). All 144 were eliminated.
- **Near-Duplicate Detection:** Alphanumeric lowercase normalization identified **3** near-duplicate variants (differing solely by punctuation or capitalization). These were safely pruned.
- **Data Leakage Mitigation Rule:** Vectorizer fitting (TF-IDF), vocabulary indexing, and Word2Vec training must occur **strictly on the training split** in Phase 2 and Phase 3. No transformation is pre-computed on the full dataset.
- **World News Filtering:** HuffPost's `WORLD NEWS` category was specifically **excluded** to prevent general geopolitical or international humanitarian reports from contaminating domestic or institutional `Politics`.

---

## 6. Sample Articles from Each Target Category

### 1. Politics
- **Sample A [BBC News, 335 words]:**  
  > *"Brown outlines plunge in NHS debt. Gordon Brown has highlighted a drop in the debts of NHS foundation hospitals as a sign that the reforms are working. He told MPs on the Commons Treasury Committee that only two of the 20 foundation trusts were running a deficit..."*
- **Sample B [HuffPost, 22 words]:**  
  > *"GOP Congressman Defends Trump's Racist Attack On Ilhan Omar - Rep. Mo Brooks said the president was merely 'pointing out the truth'..."*

### 2. Business
- **Sample A [BBC News, 269 words]:**  
  > *"Dollar hits new high versus euro. The US dollar has hit a new two-month high against the euro, buoyed by signals that the US economy is continuing to grow strongly. Federal Reserve chairman Alan Greenspan gave an upbeat assessment..."*
- **Sample B [HuffPost, 29 words]:**  
  > *"Wall Street Climbs As Tech Sector Rebounds - Major indexes rallied on Friday led by strong earnings in semiconductor and cloud software enterprises..."*

### 3. Technology
- **Sample A [BBC News, 482 words]:**  
  > *"Broadband speed claims 'misleading'. Many broadband providers are advertising speeds that users can rarely achieve in practice, a consumer group has claimed. Which? surveyed more than 1,000 UK homes and discovered real download rates lagged significantly..."*
- **Sample B [HuffPost, 31 words]:**  
  > *"Apple Unveils New Privacy Protections For Safari Users - The update introduces enhanced anti-tracking prevention features designed to block algorithmic fingerprinting across third-party websites..."*

### 4. Sports
- **Sample A [BBC News, 412 words]:**  
  > *"Arsenal cruise past Porto to advance. Arsenal secured their place in the Champions League knockout stages with a confident 2-0 victory over FC Porto at Highbury, courtesy of clinical finishes from Henry and Ljungberg..."*
- **Sample B [HuffPost, 24 words]:**  
  > *"Serena Williams Powers Into Wimbledon Semifinals - The seven-time champion displayed vintage dominance, overcoming a first-set deficit to triumph in three sets..."*

### 5. Entertainment
- **Sample A [BBC News, 315 words]:**  
  > *"Oscars race hots up with Bafta nods. British film 'The King\'s Speech' and psychological thriller 'Black Swan' have dominated nominations for the British Academy Film Awards, solidifying their front-runner status for the upcoming Academy Awards..."*
- **Sample B [HuffPost, 27 words]:**  
  > *"Lady Gaga Delivers Emotional Tribute At Grammy Awards - The multifaceted pop icon moved the audience with a soaring orchestral arrangement celebrating classic jazz and cinematic soundtrack milestones..."*

---

## 7. Artifacts & Generated Assets

1. Cleaned CSV Dataset: `dataset/processed/news_dataset_cleaned.csv`
2. Raw Datasets: `dataset/raw/bbc-fulltext.zip`, `dataset/raw/bbc_news_raw.csv`, `dataset/raw/huffpost_news_raw.csv`
3. Class Distribution Visualization: `evaluation/class_distribution.png`
4. Text Length Distribution Visualization: `evaluation/text_length_distribution.png`
5. Reproducible Notebook: `notebooks/01_dataset_exploration.ipynb`
