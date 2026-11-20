# TF-IDF & Classical Machine Learning Evaluation Report
**Project:** Intelligent News Categorization System  
**Phase:** 3 — TF-IDF Feature Extraction, Model Training & Hyperparameter Tuning  
**Generated Date:** November 2026  

---

## 1. Executive Summary

In Phase 3, we implemented, tuned, and evaluated classical machine learning classifiers using **Term Frequency - Inverse Document Frequency (TF-IDF)** feature representations for 5-class news categorization:
- **Politics**
- **Business**
- **Technology**
- **Sports**
- **Entertainment**

Two core architectures were compared:
1. **TF-IDF + Multinomial Logistic Regression** ($L_2$ regularization)
2. **TF-IDF + Linear Support Vector Machine** (LinearSVC with sigmoid probability calibration via `CalibratedClassifierCV`)

Both models achieved strong, well-calibrated performance on a completely held-out, untouched test set of **1,811 articles**:
- **Linear SVM (Calibrated):** **83.77% Accuracy**, **83.76% Macro F1-Score**
- **Logistic Regression:** **83.49% Accuracy**, **83.51% Macro F1-Score**

---

## 2. Mathematical Concepts Implemented

### 2.1 Term Frequency (TF)
Measures the frequency of term $t$ in document $d$. To prevent high frequency bias, sublinear term frequency scaling was used:
$$\text{TF}(t, d) = \begin{cases} 1 + \ln(f_{t,d}) & \text{if } f_{t,d} > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 2.2 Inverse Document Frequency (IDF)
Penalizes common corpus words and rewards terms with high discriminative utility:
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### 2.3 Normalized TF-IDF Vector & Sparse Representation
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Vectors are normalized via $L_2$ Euclidean norm:
$$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2}$$
With a vocabulary of 20,000 unigrams and bigrams, document vectors are represented as memory-efficient **Compressed Sparse Row** matrices (`scipy.sparse.csr_matrix`), storing only non-zero coordinates.

---

## 3. Data Partitioning & Leakage Prevention Protocol

The cleaned dataset (12,072 articles) was split using **Stratified Splitting**:
- **Training Set (70.0%):** 8,450 articles — Used strictly for vocabulary fitting and model parameter training.
- **Validation Set (15.0%):** 1,811 articles — Used strictly for n-gram feature search and regularization ($C$) tuning.
- **Test Set (15.0%):** 1,811 articles — Completely held out and evaluated only once at the conclusion of tuning.

```
Total: 12,072 samples
├── Training:   8,450 (70.0%) -> Fit TF-IDF & Train Models
├── Validation: 1,811 (15.0%) -> Tune Vectorizer & Hyperparameters
└── Test:       1,811 (15.0%) -> Final Unbiased Evaluation Only
```

---

## 4. Feature Engineering Experiments: TF-IDF Search

Evaluated on the Validation partition (Logistic Regression baseline, $C=1.0$):

| Configuration | N-gram Range | Max Features | Val Accuracy | Val Macro F1 | Selection |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unigrams (5k)** | (1, 1) | 5,000 | 82.50% | 0.8242 | Baseline |
| **Unigrams (10k)** | (1, 1) | 10,000 | 83.99% | 0.8393 | Moderate |
| **Unigrams + Bigrams (10k)** | (1, 2) | 10,000 | 83.77% | 0.8372 | Good |
| **Unigrams + Bigrams (15k)** | (1, 2) | 15,000 | 84.15% | 0.8410 | High |
| **Unigrams + Bigrams (20k)** | **(1, 2)** | **20,000** | **84.37%** | **0.8432** | **Optimal** |

**Conclusion:** Including bigrams (`(1, 2)`) with `max_features=20,000`, `min_df=2`, and `max_df=0.85` captured essential contextual pairings (e.g. `prime minister`, `interest rates`, `operating system`, `cloud computing`) that significantly boosted classification discriminability.

---

## 5. Hyperparameter Tuning ($C$ Regularization)

Tuned on the Validation partition across $C \in \{0.1, 0.5, 1.0, 2.0, 5.0, 10.0\}$:

### Logistic Regression
- $C=0.1$: Val Macro F1 = 0.8081
- $C=0.5$: Val Macro F1 = 0.8331
- **$C=1.0$: Val Macro F1 = 0.8432 (Selected)**
- $C=2.0$: Val Macro F1 = 0.8428
- $C=5.0$: Val Macro F1 = 0.8378
- $C=10.0$: Val Macro F1 = 0.8356

### Linear SVM
- $C=0.1$: Val Macro F1 = 0.8400
- **$C=0.5$: Val Macro F1 = 0.8403 (Selected)**
- $C=1.0$: Val Macro F1 = 0.8344
- $C=2.0$: Val Macro F1 = 0.8259
- $C=5.0$: Val Macro F1 = 0.8171
- $C=10.0$: Val Macro F1 = 0.8121

---

## 6. Final Test Set Evaluation Results

Unbiased evaluation on the 1,811 test articles:

### 6.1 Model Comparison Summary

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Weighted F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** ($C=1.0$) | 83.49% | 0.8367 | 0.8343 | 0.8351 | 0.8353 |
| **Linear SVM (Calibrated)** ($C=0.5$) | **83.77%** | **0.8389** | **0.8370** | **0.8376** | **0.8380** |

### 6.2 Per-Category Performance Breakdown

#### Logistic Regression
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Politics** | 0.7568 | 0.8213 | 0.7877 | 375 |
| **Business** | 0.8584 | 0.8535 | 0.8559 | 355 |
| **Technology** | 0.8277 | 0.8116 | 0.8196 | 361 |
| **Sports** | 0.9027 | 0.8907 | 0.8966 | 375 |
| **Entertainment** | 0.8379 | 0.7942 | 0.8155 | 345 |
| **Macro Average** | **0.8367** | **0.8343** | **0.8351** | **1,811** |

#### Linear SVM (Calibrated)
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Politics** | 0.7700 | 0.8213 | 0.7948 | 375 |
| **Business** | 0.8543 | 0.8592 | 0.8567 | 355 |
| **Technology** | 0.8139 | 0.8116 | 0.8128 | 361 |
| **Sports** | 0.9158 | 0.8987 | 0.9071 | 375 |
| **Entertainment** | 0.8405 | 0.7942 | 0.8167 | 345 |
| **Macro Average** | **0.8389** | **0.8370** | **0.8376** | **1,811** |

---

## 7. Model Explainability: Top Discriminative Features

Extracting feature coefficients from Logistic Regression reveals clear, domain-specific lexical drivers for each class:

- **Politics:** `trump` (+6.76), `republican` (+3.92), `gop` (+3.44), `democrat` (+3.31), `obama` (+3.15), `election` (+2.52), `house` (+2.44), `state` (+2.42), `congress` (+2.33), `voter` (+2.29).
- **Business:** `business` (+4.96), `company` (+3.93), `bank` (+3.06), `market` (+2.86), `economy` (+2.66), `job` (+2.46), `employee` (+2.45), `ceo` (+2.41), `customer` (+2.31), `worker` (+2.27).
- **Technology:** `apple` (+6.00), `facebook` (+5.70), `google` (+5.11), `tech` (+3.88), `technology` (+3.66), `phone` (+3.41), `internet` (+3.30), `twitter` (+3.17), `iphone` (+3.11), `online` (+3.06).
- **Sports:** `sport` (+4.44), `player` (+4.38), `nfl` (+4.18), `nba` (+3.96), `olympic` (+3.82), `coach` (+3.54), `team` (+3.48), `game` (+3.39), `football` (+3.24), `win` (+3.06).
- **Entertainment:** `film` (+4.46), `movie` (+3.54), `singer` (+3.40), `music` (+3.37), `star` (+3.27), `actor` (+2.83), `song` (+2.81), `oscar` (+2.57), `show` (+2.56), `album` (+2.44).

---

## 8. Generalization Testing on Diverse Unseen Articles

Both models were tested against completely unseen articles representing diverse syntactic structures, lengths, and writing styles:

| Article Snippet | Style | Expected | Logistic Regression Prediction | Calibrated SVM Prediction | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| *"Senate committee passes bipartisan electoral integrity bill ahead of midnight deadline."* | Short Headline | Politics | **Politics (64.7%)** | **Politics (89.0%)** | Correct |
| *"Automaker reports quarterly revenue surge of 18 percent following unprecedented expansion in commercial fleet deliveries across European hubs."* | Financial Analysis | Business | **Business (42.1%)** | **Business (70.1%)** | Correct |
| *"Cybersecurity researchers uncover zero-day vulnerability in widespread open-source cryptographic library affecting enterprise cloud clusters."* | Technical Brief | Technology | **Technology (37.5%)** | **Technology (72.6%)** | Correct |
| *"Underdog squad forces overtime penalty shootout after dramatic stoppage-time equalizer to eliminate reigning champions."* | Match Recap | Sports | **Sports (60.2%)** | **Sports (86.2%)** | Correct |
| *"Streaming platform greenlights sequel series following record-breaking viewership numbers and widespread critical acclaim for debut season."* | Cultural Media | Entertainment | **Entertainment (50.5%)** | **Entertainment (74.3%)** | Correct |

---

## 9. Artifact Storage & Google Drive Transfer Instructions

All trained model weights and preprocessing objects are saved:
- `models/tfidf_vectorizer.joblib` (779 KB)
- `models/logistic_regression_model.joblib` (801 KB)
- `models/linear_svm_model.joblib` (802 KB)
- `models/linear_svm_base.joblib` (800 KB)
- `models/tfidf_model_config.json`

### To Sync With Google Drive in Google Colab:
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy saved models from Colab session to Google Drive
!mkdir -p /content/drive/MyDrive/news_categorization_models/
!cp -r models/* /content/drive/MyDrive/news_categorization_models/

print("Artifacts successfully synced to Google Drive.")
```
To load locally, simply place the `.joblib` and `.json` files into the local `models/` directory.
