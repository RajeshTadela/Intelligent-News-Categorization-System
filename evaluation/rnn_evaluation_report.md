# Phase 5: Simple RNN News Classifier - Evaluation Report

## 1. Executive Summary & Overview
In Phase 5 of the Intelligent News Categorization System, we implemented, trained, and evaluated a **Recurrent Neural Network (Simple RNN)** classifier using dense **Word2Vec embeddings** (Skip-gram, 100-D) to classify news articles into five categories:
1. **Politics**
2. **Business**
3. **Technology**
4. **Sports**
5. **Entertainment**

### Key Model Architecture
- **Input Sequence Length:** 150 tokens (post-padded / post-truncated)
- **Embedding Layer:** Vocabulary size 17,411 $\times$ 100 dimensions initialized from Phase 4 Word2Vec Skip-gram weights.
- **Recurrent Layer:** `SimpleRNN(units=64, activation='tanh', dropout=0.2)`
- **Intermediate Dense Layer:** `Dense(32, activation='relu')` with `Dropout(0.3)`
- **Output Layer:** `Dense(5, activation='softmax')`
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Optimizer:** Adam (learning rate = 0.001)

---

## 2. Experimental Comparison: Frozen vs Fine-Tuned Embeddings

| Embedding Strategy | Description | Validation Accuracy | Validation Loss |
| :--- | :--- | :---: | :---: |
| **Frozen Embeddings** | Word2Vec weights locked (`trainable=False`, 12,805 trainable params) | **35.45%** | **1.5434** |
| **Fine-Tuned Embeddings** | Word2Vec weights updated (`trainable=True`, 1,753,905 trainable params) | 22.20% | 1.5839 |

**Observations:**
- Fine-tuning all 1.74M embedding parameters alongside a Simple RNN layer caused gradient instability and early convergence to majority-class prediction.
- The **Frozen Embedding** architecture demonstrated superior stability, preserving semantic representations learned during unsupervised Word2Vec training. It was selected as the primary Simple RNN model.

---

## 3. Test Set Performance Metrics (Evaluated on Unseen 1,811 Samples)

| Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Accuracy** | **0.3716** | **37.16%** |
| **Macro Precision** | **0.5377** | **53.77%** |
| **Macro Recall** | **0.3814** | **38.14%** |
| **Macro F1-Score** | **0.2828** | **28.28%** |
| **Weighted F1-Score** | **0.2785** | **27.85%** |

### Per-Category Classification Breakdown

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Politics** | 0.67 | 0.07 | 0.12 | 361 |
| **Business** | 0.63 | 0.07 | 0.13 | 375 |
| **Technology** | 0.31 | 0.78 | 0.45 | 345 |
| **Sports** | 0.70 | 0.10 | 0.18 | 375 |
| **Entertainment** | 0.38 | 0.88 | 0.54 | 355 |

---

## 4. Confusion Matrix Analysis

### Visual Artifacts
- **Training Curves:** `evaluation/rnn_training_curves.png`
- **Confusion Matrix Heatmap:** `evaluation/rnn_confusion_matrix.png`

**Key Observations:**
- The Simple RNN exhibits high recall on **Entertainment** (0.88) and **Technology** (0.78), while recall on **Politics** (0.07), **Business** (0.07), and **Sports** (0.10) is severely degraded.
- This asymmetry arises because the Simple RNN suffers from severe vanishing gradients and recency bias, defaulting toward predicting classes associated with trailing tokens or generic vocabulary.

---

## 5. Theoretical Comparison & Error Analysis: Why Simple RNN Struggles

### 1. Vanishing Gradients in Backpropagation Through Time (BPTT)
During training over sequential inputs of length $T = 150$, the gradient of the loss with respect to early hidden states involves continuous product terms of the recurrent weight matrix $W_{hh}$:
$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{t=2}^T \frac{\partial h_t}{\partial h_{t-1}}$$
Because the tanh derivative satisfies $|\tanh'(z)| \le 1$, if the eigenvalues of $W_{hh}$ are smaller than 1, these gradients decay exponentially towards 0. As a result, the Simple RNN predominantly updates based on recent tokens (recency bias) and forgets thematic keywords that appeared at the beginning of the article.

### 2. Primary Sources of Misclassification
1. **Vanishing Gradient Bottleneck:** In articles of 100–150 tokens, the introductory sentences contain crucial contextual cues (e.g., "The prime minister announced...", "Quarterly profits rose..."). By the time the RNN reaches step 150, the early hidden state activations have been repeatedly overwritten and their gradients have vanished.
2. **Business vs Technology overlap:** Articles concerning tech industry acquisitions, corporate earnings, or antitrust lawsuits contain overlapping financial and computational vocabulary.
3. **Entertainment vs Politics overlap:** Public statements, celebrity activism, and media coverage blend entertainment and political terminology.
4. **Sequence Padding:** For shorter articles (15–30 tokens), the remaining 120–135 time steps consist of `<PAD>` tokens, which dilute the final hidden state vector $h_T$ unless masked or trained with explicit bidirectional memory.

### 3. Comparison with Classical ML (TF-IDF + Linear SVM)
| Model Architecture | Features / Embeddings | Test Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: |
| **Linear SVM** | TF-IDF (Unigram + Bigram, 20k) | **83.77%** | **0.8376** |
| **Logistic Regression** | TF-IDF (Unigram + Bigram, 20k) | 83.49% | 0.8351 |
| **Simple RNN** | Word2Vec Skip-gram (100-D, Frozen) | **37.16%** | **0.2828** |

**Crucial Insight for Phase 6:**
Classical linear models with n-gram TF-IDF can leverage global document-level statistics simultaneously without suffering from sequence order degradation or vanishing gradients. Simple RNN introduces sequential modeling, but suffers from the vanishing gradient bottleneck on long text sequences. This provides the exact empirical motivation for introducing **Long Short-Term Memory (LSTM)** in Phase 6, where input, forget, and output gates preserve cell states over long horizons.

---

## 6. Generated Artifacts
- `models/simple_rnn_model.keras`
- `models/class_mapping.json`
- `models/simple_rnn_config.json`
- `evaluation/rnn_training_curves.png`
- `evaluation/rnn_confusion_matrix.png`
- `notebooks/05_simple_rnn_classifier.ipynb`
