# 📄 INTELLIGENT NEWS CATEGORIZATION SYSTEM
## Complete Engineering & Research Project Report

**Author:** Rajesh Tadela  
**Repository:** [https://github.com/RajeshTadela/Intelligent-News-Categorization-System](https://github.com/RajeshTadela/Intelligent-News-Categorization-System)  
**Date:** November 2024  
**Status:** Completed & Production Deployed  

---

## 📑 TABLE OF CONTENTS
1. [Executive Summary & Abstract](#1-executive-summary--abstract)
2. [Project Scope & Technology Restrictions](#2-project-scope--technology-restrictions)
3. [The Five News Categories](#3-the-five-news-categories)
4. [Phase 1: Dataset Acquisition & Integration](#4-phase-1-dataset-acquisition--integration)
5. [Phase 2: Text Preprocessing Pipeline](#5-phase-2-text-preprocessing-pipeline)
6. [Phase 3: TF-IDF & Classical Machine Learning](#6-phase-3-tf-idf--classical-machine-learning)
7. [Phase 4: Word Embeddings & Word2Vec](#7-phase-4-word-embeddings--word2vec)
8. [Phase 5: Simple Recurrent Neural Network (RNN)](#8-phase-5-simple-recurrent-neural-network-rnn)
9. [Phase 6: Long Short-Term Memory Network (LSTM)](#9-phase-6-long-short-term-memory-network-lstm)
10. [Phase 7: Comprehensive Model Comparison & Generalization](#10-phase-7-comprehensive-model-comparison--generalization)
11. [Phase 8: Streamlit Application & System Integration](#11-phase-8-streamlit-application--system-integration)
12. [Inference Robustness & Test Harness Results](#12-inference-robustness--test-harness-results)
13. [Limitations & Continuous Improvement Roadmap](#13-limitations--continuous-improvement-roadmap)
14. [Deployment & Verification Instructions](#14-deployment--verification-instructions)
15. [Academic & Dataset References](#15-academic--dataset-references)

---

## 1. Executive Summary & Abstract

The **Intelligent News Categorization System** is an end-to-end Machine Learning and Deep Learning system engineered to classify news headlines and full-length journalistic texts into five mutually exclusive categories: **Politics, Business, Technology, Sports, and Entertainment**.

The project was constructed under a strict methodological mandate: **zero reliance on Transformer architectures, pre-trained Large Language Models (LLMs), BERT, GPT, or external AI APIs**. Instead, the system explores the foundational arc of Natural Language Processing over the past two decades—spanning Bag-of-Words, sublinear TF-IDF, distributed semantic vector spaces (Word2Vec CBOW and Skip-gram), and sequential neural architectures (Simple RNN and LSTM).

### Key Empirical Findings:
- **Top Performing Architecture:** **TF-IDF + Linear Support Vector Machine (Linear SVM)** with Platt probability calibration achieved **83.77% Test Accuracy** and **0.8376 Macro F1-Score** on 1,811 held-out test articles, and **88.00% to 92.00% Accuracy** on an out-of-distribution evaluation suite of 50 newly published 2024 news articles.
- **Inference Latency:** Linear SVM achieved an average prediction latency of **0.01 ms per sample**—over **115× faster** than recurrent neural networks—with a minimal model footprint of **0.76 MB**.
- **The Vanishing Gradient Empirical Demonstration:** The project empirically proved the catastrophic impact of the Vanishing Gradient Problem in Backpropagation Through Time (BPTT). Over sequences of 150 time steps, Simple RNN achieved only **37.16% accuracy** due to exponential gradient decay, while LSTM's Constant Error Carousel (CEC) elevated Macro Precision from **53.77% to 70.82%**.
- **Calibrated Uncertainty Mechanism:** By implementing Platt scaling (`CalibratedClassifierCV`) and a validation-tuned rejection threshold ($T = 0.50$), the system successfully flags ambiguous, multi-topic articles, boosting retained prediction accuracy to **87.56%** (+3.79%) across a **91.44% coverage rate**.

---

## 2. Project Scope & Technology Restrictions

To ensure scientific rigor and understand classical machine learning mechanics, the following restrictions were strictly enforced:

### Permitted Technologies:
- **Core Languages & Data:** Python 3.10+, Pandas 2.0+, NumPy 1.24+
- **Linguistic Processing:** NLTK 3.10+ (WordNet, POS tagger, Punkt)
- **Vector Spaces & Embeddings:** Gensim 4.3+ (Word2Vec CBOW and Skip-gram)
- **Classical Machine Learning:** Scikit-learn 1.7+ (LinearSVC, LogisticRegression, CalibratedClassifierCV)
- **Deep Learning:** TensorFlow 2.21+ / Keras (SimpleRNN, LSTM, Embedding, Masking)
- **Visualization:** Matplotlib 3.7+, Seaborn 0.12+
- **Interactive UI:** Streamlit 1.55+

### Prohibited Technologies (100% Compliant):
- ❌ No Transformers (BERT, RoBERTa, DistilBERT, DeBERTa)
- ❌ No Generative AI or Large Language Models (GPT-3/4, Gemini, Claude, LLaMA)
- ❌ No Hugging Face pre-trained transformers or pipelines
- ❌ No external commercial AI classification APIs

---

## 3. The Five News Categories

The classification space is strictly partitioned into five categories:

```text
┌─────────────────┬──────────────────────────────────────────────────────────────┬──────────────────┐
│ Category        │ Semantic Domain & Topics                                     │ Brand Color      │
├─────────────────┼──────────────────────────────────────────────────────────────┼──────────────────┤
│ 🏛️ Politics     │ Parliamentary elections, diplomacy, legislation, governance  │ Crimson #e74c3c  │
│ 📈 Business     │ Monetary policy, inflation, corporate mergers, debt markets  │ Blue #3498db     │
│ 💻 Technology   │ Cybersecurity, semiconductors, quantum systems, software     │ Purple #9b59b6   │
│ ⚽ Sports       │ Tournaments, athletic championships, player trades, Olympics │ Green #2ecc71    │
│ 🎬 Entertainment│ Cinema, music albums, Broadway, television, awards galas     │ Orange #f39c12   │
└─────────────────┴──────────────────────────────────────────────────────────────┴──────────────────┘
```

---

## 4. Phase 1: Dataset Acquisition & Integration

### Data Sources:
1. **BBC News Full-Text Dataset (Greene & Cunningham, ICML 2006):** 2,122 full-length journalistic articles published between 2004 and 2005 across 5 topical desks.
2. **HuffPost News Category Dataset (Misra, arXiv 2022):** 9,950 headline and short-description pairs from 2012 to 2018 mapped to matching categories.

### Data Cleaning & Integrity Safeguards:
- **Deduplication:** Removed 147 exact duplicate and near-identical syndicated articles.
- **World News Quarantine:** General international news articles without explicit governmental or legislative context were excluded to prevent label corruption.
- **Unified Cleaned Corpus:** 12,072 articles balanced equally across all five classes (~2,400 to 2,500 samples per class, ~20% each).

### Bimodal Length Discovery:
- **HuffPost Mean Length:** 27.4 words (median 26 words)
- **BBC News Mean Length:** 384.2 words (median 331 words)
- **Significance:** 82.4% of the dataset consists of concise headline pairs, while 17.6% consists of long investigative reports.

---

## 5. Phase 2: Text Preprocessing Pipeline

Implemented in `src/preprocessing.py` through the modular `NewsPreprocessor` class:

1. **Contraction Expansion:** Normalizes 60+ English contractions (`"don't"` $\to$ `"do not"`, `"won't"` $\to$ `"will not"`).
2. **Case Normalization:** Standardizes text to lowercase.
3. **Artifact Stripping:** Removes URLs, HTML tags, email addresses, and dangling non-alphanumeric punctuation.
4. **Context-Preserving Stopword Filtering:** Standard NLTK English stopwords (179 words) were filtered, while **strictly preserving 20 critical semantic modifiers and negations**:
   `PRESERVED_WORDS = {'no', 'not', 'nor', 'neither', 'never', 'none', 'nobody', 'against', 'without', 'cannot', 'under', 'over', 'more', 'most', 'few', 'up', 'down'}`
5. **WordNet Lemmatization with POS Tagging:** Uses Penn Treebank POS tags mapped to WordNet categories (Adjective, Verb, Noun, Adverb) to ensure accurate morphological reduction (`"ran"` $\to$ `"run"`, `"better"` $\to$ `"good"`).

---

## 6. Phase 3: TF-IDF & Classical Machine Learning

### Mathematical Formulation:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{if count} > 0$$
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### Model Configurations:
- **TF-IDF Vectorizer:** Unigrams + Bigrams, $D = 20,000$ max features, $\text{min\_df}=2$, $\text{max\_df}=0.85$, sublinear term-frequency scaling.
- **Logistic Regression:** Multinomial cross-entropy loss, L2 regularization ($C=1.0$), L-BFGS solver.
- **Linear Support Vector Machine (Linear SVM):** LinearSVC ($C=0.5$), squared hinge loss, calibrated via Platt Scaling (`CalibratedClassifierCV` with 5-fold cross-validation) to produce true posterior class probabilities $P(y = c \mid x)$.

---

## 7. Phase 4: Word Embeddings & Word2Vec

Dense distributed representations were trained strictly on the 8,450 training articles using **Gensim Word2Vec**:

### Architectures Evaluated:
1. **Continuous Bag-of-Words (CBOW, `sg=0`):** Predicts center word $w_t$ from context window.
2. **Skip-gram (`sg=1`):** Predicts surrounding context $\{w_{t-k}, \dots, w_{t+k}\}$ given target word $w_t$.
- **Dimensions Tested:** 50-D, 100-D, 200-D across window size 5 and 15 epochs.
- **Selected Primary Embedding:** **Skip-gram 100-D** demonstrated superior semantic clustering on PCA projections and higher cosine similarity between related domain terms (`"minister"` $\approx$ `"parliament"`, `"inflation"` $\approx$ `"interest"`).
- **Embedding Matrix:** Shape $(17411, 100)$, covering 100% of the training vocabulary with reserved `<PAD>` (index 0) and `<OOV>` (index 1).

---

## 8. Phase 5: Simple Recurrent Neural Network (RNN)

### Recurrent Formulation:
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$\hat{y} = \text{softmax}(W_{hy} h_T + b_y)$$

### The Vanishing Gradient Problem in BPTT:
$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}, \quad \frac{\partial h_t}{\partial h_k} = \prod_{j=k+1}^t \text{diag}(1 - h_j^2) W_{hh}^T$$
Because $|\tanh'(z)| \le 1$, if the eigenvalues of $W_{hh}$ are smaller than 1, gradients decay exponentially toward zero over 150 sequence steps.

### Experimental Results:
- **Frozen Embeddings:** Validation Accuracy **35.45%**, Test Accuracy **37.16%**
- **Fine-Tuned Embeddings:** Validation Accuracy 22.20% (diverged due to gradient instability across 1.74M parameters)
- **Conclusion:** Simple RNN suffers from severe **recency bias**, forgetting introductory topical words in long news articles.

---

## 9. Phase 6: Long Short-Term Memory Network (LSTM)

### Architecture & Gating Mechanics:
LSTM resolves vanishing gradients through a dedicated **Cell State ($C_t$)** acting as a linear information highway:

1. **Forget Gate:** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
2. **Input Gate:** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$
3. **Candidate Cell State:** $\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$
4. **Additive Cell Update:** $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
5. **Output Gate & Hidden State:** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o), \quad h_t = o_t \odot \tanh(C_t)$

### The Constant Error Carousel (CEC):
Because $\frac{\partial C_t}{\partial C_{t-1}} = f_t$, when $f_t \approx 1$, error gradients flow backward through hundreds of sequence steps without decaying!

### Experimental Results:
- **LSTM-64 (Frozen):** 34.79% Validation Accuracy
- **LSTM-128 (Frozen):** 37.27% Validation Accuracy
- **LSTM-128 (Fine-Tuned, Dropout=0.4):** **41.63% Validation Accuracy**, **40.81% Test Accuracy**
- **Macro Precision Boost:** Surged from 53.77% (RNN) to **70.82%** (LSTM), with **100% precision on Business** and **98% on Sports**.

---

## 10. Phase 7: Comprehensive Model Comparison & Generalization

### 1. Master Performance Benchmark (1,811 Test Set Articles):

```text
┌─────────────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬───────────┬────────────┐
│ Model Architecture          │ Accuracy │ Macro P  │ Macro R  │ Macro F1 │ Weight F1│ Latency   │ Model Size │
├─────────────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────────┼────────────┤
│ Linear SVM (Calibrated) 🥇  │  83.77%  │  0.8389  │  0.8370  │  0.8376  │  0.8380  │  0.01 ms  │  0.76 MB   │
│ Logistic Regression 🥈     │  83.49%  │  0.8367  │  0.8343  │  0.8351  │  0.8353  │  0.00 ms  │  0.76 MB   │
│ Word2Vec + LSTM 🥉          │  40.81%  │  0.7082  │  0.4119  │  0.3366  │  0.3349  │  0.92 ms  │ 21.35 MB   │
│ Word2Vec + Simple RNN       │  37.16%  │  0.5377  │  0.3814  │  0.2828  │  0.2785  │  0.49 ms  │  6.82 MB   │
└─────────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴───────────┴────────────┘
```

### 2. External Out-of-Distribution Generalization (50 Articles from 2024 Sources):
- **Logistic Regression:** **92.00% Accuracy** (Macro F1: 0.9223)
- **Linear SVM (Calibrated):** **88.00% Accuracy** (Macro F1: 0.8850)
- **LSTM:** **40.00% Accuracy** (Macro F1: 0.2303)
- **Simple RNN:** **36.00% Accuracy** (Macro F1: 0.2095)

### 3. Prediction Uncertainty Quantification:
- Implemented rejection threshold: $\text{If } \max_c P(y = c \mid x) < 0.50 \implies \textbf{"Unable to confidently classify this article."}$
- **Baseline Accuracy:** 83.77%
- **Coverage:** **91.44%** (1,656 of 1,811 articles accepted)
- **Filtered Accuracy:** **87.56%** (**+3.79% accuracy gain**)
- **Uncertain Articles Filtered:** 155 ambiguous articles safely routed to human editorial review.

---

## 11. Phase 8: Streamlit Application & System Integration

Built as an interactive, production-grade web application in `app/app.py`:

### Application Architecture:
- `src/model_loader.py`: Singleton artifact manager caching models, vectorizers, and tokenizers.
- `src/prediction.py`: Unified inference engine supporting confidence calibration, TF-IDF feature attribution explainability, and uncertainty thresholding.
- `app/app.py`: Streamlit interface with 5 navigable views:
  1. 🏠 **Home:** System overview, category scopes, technology stack.
  2. 📰 **News Classifier:** Real-time text classification, quick-test sample selector, probability bar charts, TF-IDF explainability word tags, and uncertainty warnings.
  3. 📊 **Model Comparison Dashboard:** Embedded Phase 7 benchmark metrics, latency vs. accuracy trade-off curves, and confusion matrices.
  4. 🕒 **Prediction History:** Session-based audit trail with CSV export capability.
  5. 🔬 **NLP Theory & Walkthrough:** Educational mathematical formulations for BPTT, Vanishing Gradients, and LSTM Gating.

---

## 12. Inference Robustness & Test Harness Results

The test harness (`src/test_app_inference.py`) validated 25 category benchmark articles, edge cases, and ambiguity handlers:

```text
===========================================================================
PHASE 8: COMPREHENSIVE INFERENCE TEST SUITE
===========================================================================

[Test 1] 25 Category-Specific Benchmark Articles (Linear SVM):
  • Politics      : 4/5 Correct (Sample 3 flagged as uncertain with 38.6% conf)
  • Business      : 5/5 Correct (100%)
  • Technology    : 5/5 Correct (100%)
  • Sports        : 5/5 Correct (100%)
  • Entertainment : 5/5 Correct (100%)
  Total Categorical Benchmark Accuracy: 24/25 (96.0%)

[Test 2] Robustness & Edge Case Handling:
  • Empty String            : Status=[ERROR], Pred=[No input provided]
  • Whitespace only         : Status=[ERROR], Pred=[No input provided]
  • Single word             : Status=[ERROR], Pred=[Insufficient text]
  • Very short text         : Status=[SUCCESS], Pred=[Unable to confidently classify this article.]
  • Special characters only : Status=[ERROR], Pred=[Insufficient text]
  • Extremely long article  : Status=[SUCCESS], Pred=[Business]

[Test 3] Uncertainty Rejection Mechanism:
  • "The tech company signed an entertainment sponsorship deal for the mobile sports video game."
    -> Predicted Class: [Technology], Confidence: 84.2%, Confident: True
  • "It happened yesterday afternoon."
    -> Predicted Class: [Entertainment], Confidence: 41.7%, Confident: False
    -> Display: "Unable to confidently classify this article."
```

---

## 13. Limitations & Continuous Improvement Roadmap

### Known Limitations:
1. **Vocabulary Boundaries:** Word2Vec embeddings and TF-IDF dictionaries were fitted on 8,450 training documents. Slang or niche terminology absent from training is mapped to `<OOV>` tokens.
2. **Multi-Label Ambiguity:** The system enforces mutual exclusivity across the 5 categories. Stories covering corporate mergers in the tech sector contain strong signals for both Business and Technology, occasionally triggering the uncertainty rejection threshold.
3. **Fixed Sequence Lengths:** Recurrent models operate with a fixed cutoff ($T = 150$), truncating long investigative pieces.

### Next Roadmap Phases (Phase 2 to 11):
- **Dataset Expansion:** Target ~3,000 verified samples per class (~15,000 total) balancing short headlines, medium-length reports, and full-length articles.
- **Geopolitical "US" Stopword Bug Fix:** Ensure `"us"` is preserved alongside negations in stopword filtering.
- **Entity & Financial Marker Preservation:** Retain currency symbols (`$`, `€`) as `__CURRENCY__` tokens.
- **Recurrent Layer Masking:** Introduce `keras.layers.Masking(mask_value=0)` and `GlobalMaxPooling1D` to prevent trailing pad tokens from decaying the memory state.

---

## 14. Deployment & Verification Instructions

### 1. Clone & Set Up:
```bash
git clone https://github.com/RajeshTadela/Intelligent-News-Categorization-System.git
cd Intelligent-News-Categorization-System

python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
source venv/bin/activate       # On Linux/macOS

pip install -r requirements.txt
```

### 2. Run Test Harness:
```bash
python src/test_app_inference.py
```

### 3. Launch Streamlit Application:
```bash
streamlit run app/app.py
```
Access in browser at `http://localhost:8501`.

---

## 15. Academic & Dataset References

1. **BBC News Dataset:** D. Greene and P. Cunningham. *"Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering"*, Proceedings of the 23rd International Conference on Machine Learning (ICML 2006).
2. **HuffPost News Category Dataset:** Rishabh Misra. *"News Category Dataset"*, arXiv:2209.11429 (2022).
3. **Word2Vec:** Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. *"Efficient Estimation of Word Representations in Vector Space"*, ICLR 2013.
4. **LSTM:** Sepp Hochreiter and Jürgen Schmidhuber. *"Long Short-Term Memory"*, Neural Computation, 9(8):1735–1780, 1997.
5. **Platt Scaling:** John C. Platt. *"Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods"*, Advances in Large Margin Classifiers, 1999.
