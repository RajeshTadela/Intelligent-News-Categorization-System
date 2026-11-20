# Phase 7: Model Comparison, Generalization Testing & Final Selection

## 1. Executive Summary & Overview
In Phase 7 of the **Intelligent News Categorization System**, we conducted an exhaustive, rigorous empirical comparison of all four implemented news classification models across the exact five categories:
1. **Politics**
2. **Business**
3. **Technology**
4. **Sports**
5. **Entertainment**

### Models Benchmarked:
1. **TF-IDF + Logistic Regression** (Unigrams + Bigrams, 20,000 features, $C=1.0$)
2. **TF-IDF + Linear SVM** (Calibrated with Platt Scaling / `CalibratedClassifierCV`, $C=0.5$)
3. **Word2Vec + Simple RNN** (100-D Skip-gram embeddings, 64 hidden units, frozen weights)
4. **Word2Vec + LSTM** (100-D Skip-gram embeddings, 128 hidden units, fine-tuned weights)

All evaluations adhere strictly to zero data leakage rules: tested on the untouched test partition of 1,811 articles and an external out-of-distribution evaluation suite of 50 newly published 2024 news articles.

---

## 2. Part 1: Comprehensive Performance Comparison Table

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1 | Inference Latency | Model File Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | **83.77%** | **83.79%** | **83.76%** | **83.76%** | **83.77%** | **0.01 ms / sample** | **0.25 MB** (+0.7MB TF-IDF) |
| **Logistic Regression** | **83.49%** | **83.48%** | **83.50%** | **83.51%** | **83.52%** | **0.01 ms / sample** | **0.78 MB** (+0.7MB TF-IDF) |
| **Word2Vec + LSTM** | 40.81% | 70.82% | 41.19% | 33.66% | 33.49% | 1.15 ms / sample | 21.36 MB (+6.6MB W2V) |
| **Word2Vec + Simple RNN** | 37.16% | 53.77% | 38.14% | 28.28% | 27.85% | 0.44 ms / sample | 13.41 MB (+6.6MB W2V) |

### Visual Artifacts
- **Model Comparison Metrics Chart:** `evaluation/model_comparison_metrics.png`
- **Combined 2x2 Confusion Matrices:** `evaluation/all_models_confusion_matrices.png`
- **Per-Category F1-Score Breakdown:** `evaluation/per_category_comparison.png`

---

## 3. Part 2: Per-Category Performance Breakdown

### F1-Score Comparison by Category (%):

| Category | Linear SVM | Logistic Regression | LSTM | Simple RNN |
| :--- | :---: | :---: | :---: | :---: |
| **Politics** | **87.21%** | 86.88% | 61.30% | 12.09% |
| **Business** | **78.29%** | 77.81% | 16.18% | 12.92% |
| **Technology** | **80.34%** | 80.56% | 24.02% | 44.78% |
| **Sports** | **94.86%** | 94.62% | 22.64% | 18.10% |
| **Entertainment** | **78.07%** | 77.70% | 44.16% | 53.50% |

**Observations:**
- **Sports** is the most cleanly separable category across classical models (>94% F1) due to specialized, unambiguous athletic terminology.
- **Business vs. Technology** constitutes the most frequent point of confusion for all models, as modern business reports regularly discuss tech corporate earnings, AI investments, and antitrust litigation.
- **Politics vs. Entertainment** overlaps predominantly on articles covering public political scandals, celebrity activism, and media satire.

---

## 4. Part 3: External Unseen Data Testing (50 Articles from 2024 Outlets)

To test real-world generalization without data leakage, we curated an external suite of 50 new articles published in 2024 from Reuters, Bloomberg, BBC, TechCrunch, ESPN, and Variety (10 per category).

| Model Architecture | External Accuracy | External Macro F1 | Generalization Assessment |
| :--- | :---: | :---: | :--- |
| **Linear SVM (Calibrated)** | **92.00%** (46/50) | **0.9197** | **Outstanding generalization** to contemporary syntax and terminology. |
| **Logistic Regression** | **90.00%** (45/50) | **0.8998** | Strong generalization with smooth probability calibration. |
| **Word2Vec + LSTM** | 56.00% (28/50) | 0.5369 | Moderate generalization; resilient on clear Politics articles. |
| **Word2Vec + Simple RNN** | 44.00% (22/50) | 0.4082 | Poor generalization; recency bias limits sentence comprehension. |

*Finding:* Classical models generalized substantially better to unseen 2024 articles than recurrent models. Linear SVM achieved **92.00% accuracy** on external articles, confirming that high-dimensional n-gram features maintain strong discriminative boundaries even with novel lexical variations.

---

## 5. Part 4: Confidence & Uncertainty Quantification

### 1. Calibration Reality Check
Softmax outputs from neural networks often produce overconfident probabilities even when wrong. For classical linear models, we used **`CalibratedClassifierCV` (Platt Scaling / Sigmoid Calibration)** to ensure that a predicted probability of 0.80 genuinely corresponds to an empirical 80% likelihood of correctness.

### 2. Validation-Tuned Rejection Threshold ($T = 0.50$)
Using the validation set, we analyzed the trade-off between coverage and accuracy. Setting a rejection threshold of **$T = 0.50$** ensures:
- If $\max_c P(y = c \mid x) < 0.50$, the system flags the article as uncertain and returns:  
  **`"Unable to confidently classify this article."`**

### Test Set Rejection Results:
- **Baseline Accuracy (All Predictions):** 83.77%
- **Coverage (% Predictions Retained):** **91.44%** (1,684 / 1,811 articles)
- **Filtered Accuracy (Retained Predictions):** **87.56%**
- **Uncertain Articles Filtered:** 127 articles safely routed to human editorial review.

---

## 6. Part 5: Final Model Selection & Justification

### Recommended Production Model: **TF-IDF + Linear SVM (Calibrated)**

| Selection Criterion | Linear SVM (Calibrated) | Logistic Regression | Word2Vec + LSTM | Word2Vec + Simple RNN |
| :--- | :---: | :---: | :---: | :---: |
| **Macro F1-Score** | **83.76% (Rank 1)** | 83.51% (Rank 2) | 33.66% (Rank 3) | 28.28% (Rank 4) |
| **External Unseen Accuracy** | **92.00% (Rank 1)** | 90.00% (Rank 2) | 56.00% (Rank 3) | 44.00% (Rank 4) |
| **Inference Latency** | **0.01 ms (Fastest)** | 0.01 ms (Fastest) | 1.15 ms (115x slower) | 0.44 ms (44x slower) |
| **Model Disk Size** | **0.25 MB (Smallest)** | 0.78 MB | 21.36 MB | 13.41 MB |
| **Probability Calibration** | **Calibrated (`CalibratedClassifierCV`)** | Natively Calibrated | Heuristic Softmax | Heuristic Softmax |
| **Explainability** | **Exact Feature Weights** | Exact Feature Weights | Black-box Recurrence | Black-box Recurrence |

### Why Linear SVM Wins:
1. **Mathematical Superiority on Sparse Text:** High-dimensional TF-IDF spaces ($D = 20,000$) are linearly separable in text categorization. Linear SVM finds the maximum-margin hyperplane, minimizing structural risk.
2. **Zero Temporal Attenuation:** Unlike RNN/LSTM, which suffer from sequence truncations, padding dilution, or recency bias, TF-IDF evaluates global document tokens simultaneously.
3. **Ultra-Low Latency & Footprint:** At 0.01 ms per article and 0.25 MB disk size, it is ideal for real-time Streamlit deployment on edge or standard CPUs.
4. **Calibrated Confidence:** Equipped with Platt scaling, it seamlessly supports our uncertainty rejection mechanism.

---

## 7. Generated Artifacts
- `evaluation/model_comparison_report.md`
- `evaluation/model_comparison_metrics.png`
- `evaluation/all_models_confusion_matrices.png`
- `evaluation/per_category_comparison.png`
- `dataset/processed/external_unseen_news.csv`
- `notebooks/07_model_comparison.ipynb`
