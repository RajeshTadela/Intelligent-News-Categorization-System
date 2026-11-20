# Phase 6: Word2Vec + LSTM News Classifier - Evaluation Report

## 1. Executive Summary & Overview
In Phase 6 of the Intelligent News Categorization System, we implemented, optimized, and evaluated a **Long Short-Term Memory (LSTM)** neural network classifier using dense **Word2Vec embeddings** (Skip-gram, 100-D) across the five target news categories:
1. **Politics**
2. **Business**
3. **Technology**
4. **Sports**
5. **Entertainment**

### Key Model Architecture
- **Input Sequence Length:** 150 tokens (post-padded / post-truncated)
- **Embedding Layer:** Vocabulary size 17,411 $\times$ 100 dimensions initialized from Phase 4 Word2Vec Skip-gram weights.
- **Recurrent Layer:** `LSTM` with internal memory cell and three gating mechanisms (Forget, Input, Output).
- **Dense Intermediate Layer:** `Dense(32, activation='relu')` with Dropout.
- **Output Layer:** `Dense(5, activation='softmax')`
- **Loss Function:** Sparse Categorical Cross-Entropy | **Optimizer:** Adam ($lr = 0.001$)

---

## 2. Experimental Validation & Hyperparameter Comparison

We compared three architectural configurations on the **Validation Set** (1,811 samples, never exposed to training):

| Configuration | Architecture | Trainable Parameters | Validation Accuracy | Validation Loss |
| :--- | :--- | :---: | :---: | :---: |
| **Experiment 1** | LSTM-64 (Dropout=0.3, Frozen Embeddings) | 44,517 | 34.79% | 1.3972 |
| **Experiment 2** | LSTM-128 (Dropout=0.3, Frozen Embeddings) | 120,037 | 37.27% | 1.3435 |
| **Experiment 3** | LSTM-FineTuned (Dropout=0.4, Trainable Embeddings) | ~1.86M | 41.63% | 1.2846 |

**Selected Primary Model:** `LSTM-128 (Fine-Tuned)`

---

## 3. Test Set Performance Metrics (Unseen 1,811 Samples)

| Evaluation Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Test Accuracy** | **0.4081** | **40.81%** |
| **Macro Precision** | **0.7082** | **70.82%** |
| **Macro Recall** | **0.4119** | **41.19%** |
| **Macro F1-Score** | **0.3366** | **33.66%** |
| **Weighted F1-Score** | **0.3349** | **33.49%** |

### Per-Category Classification Breakdown

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Politics** | 0.4885 | 0.8227 | 0.6130 | 361 |
| **Business** | 1.0000 | 0.0880 | 0.1618 | 375 |
| **Technology** | 0.7778 | 0.1420 | 0.2402 | 345 |
| **Sports** | 0.9796 | 0.1280 | 0.2264 | 375 |
| **Entertainment** | 0.2949 | 0.8789 | 0.4416 | 355 |

---

## 4. Confusion Matrix Analysis

The confusion matrix on the test set is recorded below:

```
Predicted ->   Politics   Business  Technology    Sports   Entertainment
Politics          297        0          1          0          63        
Business          177        33         8          1          156       
Technology        36         0          49         0          260       
Sports            60         0          0          48         267       
Entertainment     38         0          5          0          312       
```

### Visual Artifacts
- **Training Curves:** `evaluation/lstm_training_curves.png`
- **Confusion Matrix Heatmap:** `evaluation/lstm_confusion_matrix.png`

---

## 5. Architectural Deep-Dive: Why LSTM Solves Ordinary RNN's Limitations

### 1. Ordinary RNN Bottleneck
In a Simple RNN, the hidden state update is strictly multiplicative through time:
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
During Backpropagation Through Time (BPTT), computing $\frac{\partial L}{\partial h_1}$ involves continuous product matrices $\prod_{j=2}^T W_{hh}^T \text{diag}(1 - h_j^2)$, resulting in exponential gradient decay.

### 2. The LSTM Gating Mechanism & Constant Error Carousel
LSTM introduces an isolated **Cell State** ($C_t$) that acts as an additive information highway running directly down the entire sequence chain. Information is modulated via three specialized non-linear gates:

1. **Forget Gate ($f_t$):** Determines what proportion of past memory $C_{t-1}$ to discard:
   $$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$
2. **Input Gate ($i_t$) & Candidate Cell State ($\tilde{C}_t$):** Determines what new information to write into the cell:
   $$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
   $$\tilde{C}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
3. **Cell State Update:**
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
   *Crucial Mathematical Property:* The gradient $\frac{\partial C_t}{\partial C_{t-1}} = f_t$. If the forget gate is near $1$, error gradients flow backward through hundreds of steps without decaying, completely resolving the vanishing gradient problem!
4. **Output Gate ($o_t$) & Hidden State ($h_t$):** Modulates what part of the cell state is emitted to the next layer:
   $$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
   $$h_t = o_t \odot \tanh(C_t)$$

---

## 6. Comprehensive 4-Model Comparison Across All Project Phases

| Phase | Model | Representation | Test Accuracy | Macro F1-Score | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Phase 3** | **Linear SVM (Calibrated)** | TF-IDF (Unigram + Bigram, 20k) | **83.77%** | **0.8376** | Classical ML Benchmark |
| **Phase 3** | **Logistic Regression** | TF-IDF (Unigram + Bigram, 20k) | **83.49%** | **0.8351** | Convex Linear Baseline |
| **Phase 5** | **Simple RNN** | Word2Vec Skip-gram (100-D, Frozen) | **37.16%** | **0.2828** | Degraded by Vanishing Gradients |
| **Phase 6** | **LSTM** | Word2Vec Skip-gram (100-D) | **40.81%** | **0.3366** | Deep Recurrent Architecture |

---

## 7. Generated Artifacts
- `models/lstm_model.keras`
- `models/lstm_config.json`
- `evaluation/lstm_training_curves.png`
- `evaluation/lstm_confusion_matrix.png`
- `evaluation/lstm_evaluation_report.md`
- `notebooks/06_lstm_classifier.ipynb`
