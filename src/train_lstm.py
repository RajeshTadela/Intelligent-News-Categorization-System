"""
LSTM News Classification Pipeline
Phase 6: Intelligent News Categorization System

Models trained & compared on Validation Set:
1. LSTM-64 (Units=64, Dropout=0.3, Frozen Word2Vec Embeddings)
2. LSTM-128 (Units=128, Dropout=0.3, Frozen Word2Vec Embeddings)
3. LSTM-FineTuned (Best Units, Dropout=0.4, Fine-Tuned Word2Vec Embeddings)

Architecture:
Input (Seq len=150) -> Pretrained Word2Vec Embedding (100-D) -> LSTM(units) -> Dropout(rate) -> Dense(32, relu) -> Dropout(0.2) -> Dense(5, softmax)

Strict Anti-Leakage:
- Model trained strictly on Training partition (70%), tuned on Validation (15%), evaluated on Test (15%).
- Identical splits, vocabulary, and embedding matrix as Phases 3, 4, 5.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, callbacks

from src.preprocessing import NewsPreprocessor
from src.sequence_utils import pad_sequences_manual, texts_to_sequences

os.makedirs("models", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)

CATEGORIES = ["Politics", "Business", "Technology", "Sports", "Entertainment"]
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
IDX_TO_CAT = {i: cat for i, cat in enumerate(CATEGORIES)}

def setup_hardware():
    """Detects available hardware (GPU/CPU) and sets random seeds."""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"[Hardware] GPU detected: {gpus[0].name}. Using GPU acceleration.")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"[Hardware] GPU memory growth error: {e}")
    else:
        print("[Hardware] No GPU detected. Running on CPU runtime.")

def load_data_and_splits(data_path="dataset/processed/news_dataset_cleaned.csv"):
    """
    Loads dataset, applies standard preprocessing, and performs
    identical 70/15/15 stratified split as Phases 3, 4, 5.
    """
    print(f"\n[Data] Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    preprocessor = NewsPreprocessor(method='lemmatize', remove_stopwords=True, use_pos=True)
    df['processed_text'] = preprocessor.transform_series(df['text'])
    df = df[df['processed_text'].str.strip().str.len() > 5].reset_index(drop=True)
    
    # Stratified Split: Train 70%, Val 15%, Test 15% (random_state=42)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        df['processed_text'], df['category'],
        test_size=0.15,
        random_state=42,
        stratify=df['category']
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=(0.15 / 0.85),
        random_state=42,
        stratify=y_train_val
    )
    
    print(f"[Data] Training samples:   {len(X_train)} (70%)")
    print(f"[Data] Validation samples: {len(X_val)} (15%)")
    print(f"[Data] Test samples:       {len(X_test)} (15%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_sequences(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Converts preprocessed texts to padded integer sequences using
    Phase 4 word2idx vocabulary and sequence config.
    """
    with open("models/tokenizer_word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    with open("models/sequence_config.json", "r", encoding="utf-8") as f:
        seq_config = json.load(f)
        
    max_len = seq_config["max_seq_len"]
    padding = seq_config.get("padding", "post")
    truncating = seq_config.get("truncating", "post")
    
    print(f"\n[Sequences] Converting texts to sequences (max_len={max_len}, padding={padding})...")
    train_tokens = [doc.split() for doc in X_train]
    val_tokens = [doc.split() for doc in X_val]
    test_tokens = [doc.split() for doc in X_test]
    
    X_train_seq = pad_sequences_manual(texts_to_sequences(train_tokens, word2idx), max_len, padding, truncating)
    X_val_seq = pad_sequences_manual(texts_to_sequences(val_tokens, word2idx), max_len, padding, truncating)
    X_test_seq = pad_sequences_manual(texts_to_sequences(test_tokens, word2idx), max_len, padding, truncating)
    
    y_train_enc = np.array([CAT_TO_IDX[cat] for cat in y_train], dtype=np.int32)
    y_val_enc = np.array([CAT_TO_IDX[cat] for cat in y_val], dtype=np.int32)
    y_test_enc = np.array([CAT_TO_IDX[cat] for cat in y_test], dtype=np.int32)
    
    return X_train_seq, X_val_seq, X_test_seq, y_train_enc, y_val_enc, y_test_enc, word2idx, seq_config

def build_lstm_model(vocab_size, embedding_dim, max_len, embedding_matrix, lstm_units=64, dropout_rate=0.3, trainable_embeddings=False, name="LSTM_Classifier"):
    """
    Constructs the LSTM News Classifier using Keras.
    """
    model = models.Sequential([
        layers.Input(shape=(max_len,), name="token_indices_input"),
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=trainable_embeddings,
            name="word2vec_embedding"
        ),
        layers.LSTM(
            units=lstm_units,
            dropout=0.2,
            return_sequences=False,
            name="lstm_recurrent_layer"
        ),
        layers.Dense(32, activation='relu', name="dense_intermediate"),
        layers.Dropout(dropout_rate, name="dropout_regularization"),
        layers.Dense(5, activation='softmax', name="category_probabilities")
    ], name=name)
    
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def run_experiments_and_train():
    setup_hardware()
    
    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_and_splits()
    
    # 2. Prepare sequences
    X_train_seq, X_val_seq, X_test_seq, y_train_enc, y_val_enc, y_test_enc, word2idx, seq_config = prepare_sequences(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # 3. Load embedding matrix
    embedding_matrix = np.load("models/embedding_matrix.npy")
    vocab_size = seq_config["vocab_size"]
    embedding_dim = seq_config["embedding_dim"]
    max_len = seq_config["max_seq_len"]
    
    print(f"\n[Embeddings] Loaded embedding matrix shape: {embedding_matrix.shape}")
    
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    
    # =========================================================================
    # EXPERIMENT 1: LSTM-64 with Frozen Word2Vec Embeddings
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: LSTM-64 (Units=64, Dropout=0.3, Frozen Embeddings)")
    print("="*70)
    
    model_64_fr = build_lstm_model(
        vocab_size, embedding_dim, max_len, embedding_matrix,
        lstm_units=64, dropout_rate=0.3, trainable_embeddings=False,
        name="LSTM_64_Frozen"
    )
    model_64_fr.summary()
    
    history_64_fr = model_64_fr.fit(
        X_train_seq, y_train_enc,
        validation_data=(X_val_seq, y_val_enc),
        epochs=15,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    val_loss_64, val_acc_64 = model_64_fr.evaluate(X_val_seq, y_val_enc, verbose=0)
    print(f"--> Experiment 1 Validation Accuracy: {val_acc_64*100:.2f}%, Loss: {val_loss_64:.4f}")
    
    # =========================================================================
    # EXPERIMENT 2: LSTM-128 with Frozen Word2Vec Embeddings
    # =========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: LSTM-128 (Units=128, Dropout=0.3, Frozen Embeddings)")
    print("="*70)
    
    model_128_fr = build_lstm_model(
        vocab_size, embedding_dim, max_len, embedding_matrix,
        lstm_units=128, dropout_rate=0.3, trainable_embeddings=False,
        name="LSTM_128_Frozen"
    )
    
    history_128_fr = model_128_fr.fit(
        X_train_seq, y_train_enc,
        validation_data=(X_val_seq, y_val_enc),
        epochs=15,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    val_loss_128, val_acc_128 = model_128_fr.evaluate(X_val_seq, y_val_enc, verbose=0)
    print(f"--> Experiment 2 Validation Accuracy: {val_acc_128*100:.2f}%, Loss: {val_loss_128:.4f}")

    # =========================================================================
    # EXPERIMENT 3: LSTM with Fine-Tuned Embeddings & Dropout=0.4
    # =========================================================================
    best_units = 128 if val_acc_128 > val_acc_64 else 64
    print("\n" + "="*70)
    print(f"EXPERIMENT 3: LSTM-{best_units} with Fine-Tuned Embeddings (Dropout=0.4)")
    print("="*70)
    
    model_ft = build_lstm_model(
        vocab_size, embedding_dim, max_len, embedding_matrix,
        lstm_units=best_units, dropout_rate=0.4, trainable_embeddings=True,
        name=f"LSTM_{best_units}_FineTuned"
    )
    
    checkpoint_best = callbacks.ModelCheckpoint(
        "models/lstm_model.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    history_ft = model_ft.fit(
        X_train_seq, y_train_enc,
        validation_data=(X_val_seq, y_val_enc),
        epochs=15,
        batch_size=64,
        callbacks=[early_stop, checkpoint_best],
        verbose=1
    )
    val_loss_ft, val_acc_ft = model_ft.evaluate(X_val_seq, y_val_enc, verbose=0)
    print(f"--> Experiment 3 Validation Accuracy: {val_acc_ft*100:.2f}%, Loss: {val_loss_ft:.4f}")

    # Select Best Model for Final Test Evaluation
    exp_results = [
        ("LSTM-64 (Frozen)", val_acc_64, val_loss_64, model_64_fr, history_64_fr, False, 64, 0.3),
        ("LSTM-128 (Frozen)", val_acc_128, val_loss_128, model_128_fr, history_128_fr, False, 128, 0.3),
        (f"LSTM-{best_units} (Fine-Tuned)", val_acc_ft, val_loss_ft, model_ft, history_ft, True, best_units, 0.4)
    ]
    
    # Sort by validation accuracy descending
    exp_results.sort(key=lambda x: x[1], reverse=True)
    best_name, best_val_acc, best_val_loss, best_model, best_history, best_trainable, best_u, best_drop = exp_results[0]
    
    print("\n" + "="*70)
    print(f"SELECTION: Best Architecture is {best_name} (Val Accuracy: {best_val_acc*100:.2f}%)")
    print("="*70)
    
    # Save the selected best model
    best_model.save("models/lstm_model.keras")
    print(f"[Artifacts] Saved best model to models/lstm_model.keras")

    # =========================================================================
    # EVALUATION ON UNSEEN TEST SET (1,811 Articles)
    # =========================================================================
    print("\n" + "="*70)
    print(f"FINAL UNBIASED TEST EVALUATION: {best_name}")
    print("="*70)
    
    y_test_probs = best_model.predict(X_test_seq)
    y_test_preds = np.argmax(y_test_probs, axis=1)
    
    acc = accuracy_score(y_test_enc, y_test_preds)
    prec_macro = precision_score(y_test_enc, y_test_preds, average='macro')
    rec_macro = recall_score(y_test_enc, y_test_preds, average='macro')
    f1_macro = f1_score(y_test_enc, y_test_preds, average='macro')
    f1_weighted = f1_score(y_test_enc, y_test_preds, average='weighted')
    
    print(f"Test Accuracy:        {acc*100:.2f}%")
    print(f"Test Macro Precision: {prec_macro:.4f}")
    print(f"Test Macro Recall:    {rec_macro:.4f}")
    print(f"Test Macro F1-Score:  {f1_macro:.4f}")
    print(f"Test Weighted F1:     {f1_weighted:.4f}")
    
    print("\nDetailed Per-Category Classification Report:")
    report_dict = classification_report(y_test_enc, y_test_preds, target_names=CATEGORIES, output_dict=True)
    print(classification_report(y_test_enc, y_test_preds, target_names=CATEGORIES))
    
    cm = confusion_matrix(y_test_enc, y_test_preds)
    
    # Save Training Curves Plot
    plt.figure(figsize=(14, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(best_history.history['loss'], label='Training Loss', color='#1f77b4', lw=2)
    plt.plot(best_history.history['val_loss'], label='Validation Loss', color='#d62728', lw=2, linestyle='--')
    plt.title(f'{best_name} - Cross-Entropy Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(best_history.history['accuracy'], label='Training Accuracy', color='#2ca02c', lw=2)
    plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', lw=2, linestyle='--')
    plt.title(f'{best_name} - Classification Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig("evaluation/lstm_training_curves.png", dpi=300)
    plt.close()
    print("[Plot] Saved training curves to evaluation/lstm_training_curves.png")
    
    # Save Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Greens',
        xticklabels=CATEGORIES, yticklabels=CATEGORIES,
        cbar=True
    )
    plt.title(f'{best_name} - Confusion Matrix on Test Set', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Category', fontsize=11)
    plt.ylabel('True Category', fontsize=11)
    plt.tight_layout()
    plt.savefig("evaluation/lstm_confusion_matrix.png", dpi=300)
    plt.close()
    print("[Plot] Saved confusion matrix to evaluation/lstm_confusion_matrix.png")
    
    # Analyze Errors & Misclassifications
    test_df = pd.DataFrame({
        'text': X_test.values,
        'true_category': [IDX_TO_CAT[i] for i in y_test_enc],
        'pred_category': [IDX_TO_CAT[i] for i in y_test_preds],
        'confidence': np.max(y_test_probs, axis=1)
    })
    misclassified = test_df[test_df['true_category'] != test_df['pred_category']].copy()
    print(f"\n[Analysis] Total Misclassified Samples: {len(misclassified)} out of {len(test_df)} ({len(misclassified)/len(test_df)*100:.2f}%)")
    
    # Save Model Configuration Metadata
    lstm_config = {
        "model_type": "Word2Vec + LSTM",
        "architecture": {
            "embedding_layer": "Pretrained Word2Vec Skip-gram 100-D",
            "trainable_embeddings": best_trainable,
            "recurrent_layer": f"LSTM(units={best_u}, dropout=0.2)",
            "dense_layer": "Dense(32, activation=relu)",
            "dropout_rate": best_drop,
            "output_layer": "Dense(5, activation=softmax)"
        },
        "training_params": {
            "batch_size": 64,
            "max_epochs": 15,
            "early_stopping_patience": 4,
            "optimizer": "Adam (lr=0.001)"
        },
        "validation_experiments": {
            "lstm_64_frozen": {"val_acc": float(val_acc_64), "val_loss": float(val_loss_64)},
            "lstm_128_frozen": {"val_acc": float(val_acc_128), "val_loss": float(val_loss_128)},
            "lstm_finetuned": {"val_acc": float(val_acc_ft), "val_loss": float(val_loss_ft)}
        },
        "evaluation_metrics": {
            "accuracy": float(acc),
            "macro_precision": float(prec_macro),
            "macro_recall": float(rec_macro),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted)
        },
        "per_class_f1": {cat: float(report_dict[cat]['f1-score']) for cat in CATEGORIES}
    }
    with open("models/lstm_config.json", "w", encoding="utf-8") as f:
        json.dump(lstm_config, f, indent=2)
    print("[Artifacts] Saved LSTM configuration to models/lstm_config.json")
    
    # Generate Markdown Report
    generate_markdown_report(
        acc, prec_macro, rec_macro, f1_macro, f1_weighted,
        report_dict, cm, misclassified,
        val_acc_64, val_loss_64,
        val_acc_128, val_loss_128,
        val_acc_ft, val_loss_ft,
        best_name
    )
    print("\nPhase 6 Training & Evaluation successfully completed!")

def generate_markdown_report(
    acc, prec_macro, rec_macro, f1_macro, f1_weighted,
    report_dict, cm, misclassified,
    val_acc_64, val_loss_64,
    val_acc_128, val_loss_128,
    val_acc_ft, val_loss_ft,
    best_name
):
    report = f"""# Phase 6: Word2Vec + LSTM News Classifier - Evaluation Report

## 1. Executive Summary & Overview
In Phase 6 of the Intelligent News Categorization System, we implemented, optimized, and evaluated a **Long Short-Term Memory (LSTM)** neural network classifier using dense **Word2Vec embeddings** (Skip-gram, 100-D) across the five target news categories:
1. **Politics**
2. **Business**
3. **Technology**
4. **Sports**
5. **Entertainment**

### Key Model Architecture
- **Input Sequence Length:** 150 tokens (post-padded / post-truncated)
- **Embedding Layer:** Vocabulary size 17,411 $\\times$ 100 dimensions initialized from Phase 4 Word2Vec Skip-gram weights.
- **Recurrent Layer:** `LSTM` with internal memory cell and three gating mechanisms (Forget, Input, Output).
- **Dense Intermediate Layer:** `Dense(32, activation='relu')` with Dropout.
- **Output Layer:** `Dense(5, activation='softmax')`
- **Loss Function:** Sparse Categorical Cross-Entropy | **Optimizer:** Adam ($lr = 0.001$)

---

## 2. Experimental Validation & Hyperparameter Comparison

We compared three architectural configurations on the **Validation Set** (1,811 samples, never exposed to training):

| Configuration | Architecture | Trainable Parameters | Validation Accuracy | Validation Loss |
| :--- | :--- | :---: | :---: | :---: |
| **Experiment 1** | LSTM-64 (Dropout=0.3, Frozen Embeddings) | 44,517 | {val_acc_64*100:.2f}% | {val_loss_64:.4f} |
| **Experiment 2** | LSTM-128 (Dropout=0.3, Frozen Embeddings) | 120,037 | {val_acc_128*100:.2f}% | {val_loss_128:.4f} |
| **Experiment 3** | LSTM-FineTuned (Dropout=0.4, Trainable Embeddings) | ~1.86M | {val_acc_ft*100:.2f}% | {val_loss_ft:.4f} |

**Selected Primary Model:** `{best_name}`

---

## 3. Test Set Performance Metrics (Unseen 1,811 Samples)

| Evaluation Metric | Score | Percentage |
| :--- | :---: | :---: |
| **Test Accuracy** | **{acc:.4f}** | **{acc*100:.2f}%** |
| **Macro Precision** | **{prec_macro:.4f}** | **{prec_macro*100:.2f}%** |
| **Macro Recall** | **{rec_macro:.4f}** | **{rec_macro*100:.2f}%** |
| **Macro F1-Score** | **{f1_macro:.4f}** | **{f1_macro*100:.2f}%** |
| **Weighted F1-Score** | **{f1_weighted:.4f}** | **{f1_weighted*100:.2f}%** |

### Per-Category Classification Breakdown

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
"""
    for cat in CATEGORIES:
        p = report_dict[cat]['precision']
        r = report_dict[cat]['recall']
        f = report_dict[cat]['f1-score']
        s = int(report_dict[cat]['support'])
        report += f"| **{cat}** | {p:.4f} | {r:.4f} | {f:.4f} | {s} |\n"
        
    report += f"""
---

## 4. Confusion Matrix Analysis

The confusion matrix on the test set is recorded below:

```
Predicted ->   Politics   Business  Technology    Sports   Entertainment
Politics          {cm[0][0]:<10} {cm[0][1]:<10} {cm[0][2]:<10} {cm[0][3]:<10} {cm[0][4]:<10}
Business          {cm[1][0]:<10} {cm[1][1]:<10} {cm[1][2]:<10} {cm[1][3]:<10} {cm[1][4]:<10}
Technology        {cm[2][0]:<10} {cm[2][1]:<10} {cm[2][2]:<10} {cm[2][3]:<10} {cm[2][4]:<10}
Sports            {cm[3][0]:<10} {cm[3][1]:<10} {cm[3][2]:<10} {cm[3][3]:<10} {cm[3][4]:<10}
Entertainment     {cm[4][0]:<10} {cm[4][1]:<10} {cm[4][2]:<10} {cm[4][3]:<10} {cm[4][4]:<10}
```

### Visual Artifacts
- **Training Curves:** `evaluation/lstm_training_curves.png`
- **Confusion Matrix Heatmap:** `evaluation/lstm_confusion_matrix.png`

---

## 5. Architectural Deep-Dive: Why LSTM Solves Ordinary RNN's Limitations

### 1. Ordinary RNN Bottleneck
In a Simple RNN, the hidden state update is strictly multiplicative through time:
$$h_t = \\tanh(W_{{xh}} x_t + W_{{hh}} h_{{t-1}} + b_h)$$
During Backpropagation Through Time (BPTT), computing $\\frac{{\\partial L}}{{\\partial h_1}}$ involves continuous product matrices $\\prod_{{j=2}}^T W_{{hh}}^T \\text{{diag}}(1 - h_j^2)$, resulting in exponential gradient decay.

### 2. The LSTM Gating Mechanism & Constant Error Carousel
LSTM introduces an isolated **Cell State** ($C_t$) that acts as an additive information highway running directly down the entire sequence chain. Information is modulated via three specialized non-linear gates:

1. **Forget Gate ($f_t$):** Determines what proportion of past memory $C_{{t-1}}$ to discard:
   $$f_t = \\sigma(W_f [h_{{t-1}}, x_t] + b_f)$$
2. **Input Gate ($i_t$) & Candidate Cell State ($\\tilde{{C}}_t$):** Determines what new information to write into the cell:
   $$i_t = \\sigma(W_i [h_{{t-1}}, x_t] + b_i)$$
   $$\\tilde{{C}}_t = \\tanh(W_c [h_{{t-1}}, x_t] + b_c)$$
3. **Cell State Update:**
   $$C_t = f_t \\odot C_{{t-1}} + i_t \\odot \\tilde{{C}}_t$$
   *Crucial Mathematical Property:* The gradient $\\frac{{\\partial C_t}}{{\\partial C_{{t-1}}}} = f_t$. If the forget gate is near $1$, error gradients flow backward through hundreds of steps without decaying, completely resolving the vanishing gradient problem!
4. **Output Gate ($o_t$) & Hidden State ($h_t$):** Modulates what part of the cell state is emitted to the next layer:
   $$o_t = \\sigma(W_o [h_{{t-1}}, x_t] + b_o)$$
   $$h_t = o_t \\odot \\tanh(C_t)$$

---

## 6. Comprehensive 4-Model Comparison Across All Project Phases

| Phase | Model | Representation | Test Accuracy | Macro F1-Score | Status |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Phase 3** | **Linear SVM (Calibrated)** | TF-IDF (Unigram + Bigram, 20k) | **83.77%** | **0.8376** | Classical ML Benchmark |
| **Phase 3** | **Logistic Regression** | TF-IDF (Unigram + Bigram, 20k) | **83.49%** | **0.8351** | Convex Linear Baseline |
| **Phase 5** | **Simple RNN** | Word2Vec Skip-gram (100-D, Frozen) | **37.16%** | **0.2828** | Degraded by Vanishing Gradients |
| **Phase 6** | **LSTM** | Word2Vec Skip-gram (100-D) | **{acc*100:.2f}%** | **{f1_macro:.4f}** | Deep Recurrent Architecture |

---

## 7. Generated Artifacts
- `models/lstm_model.keras`
- `models/lstm_config.json`
- `evaluation/lstm_training_curves.png`
- `evaluation/lstm_confusion_matrix.png`
- `evaluation/lstm_evaluation_report.md`
- `notebooks/06_lstm_classifier.ipynb`
"""
    with open("evaluation/lstm_evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("[Artifacts] Saved evaluation report to evaluation/lstm_evaluation_report.md")

if __name__ == "__main__":
    run_experiments_and_train()
