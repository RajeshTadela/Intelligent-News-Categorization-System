"""
Generates the Google Colab Notebook: notebooks/06_lstm_classifier.ipynb
Phase 6: Word2Vec + LSTM News Classifier
"""
import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Intelligent News Categorization System\n",
                "## Phase 6: Word2Vec + LSTM News Classifier\n",
                "\n",
                "### Objective\n",
                "Implement, train, evaluate, and interpret a **Long Short-Term Memory (LSTM)** neural network classifier using dense **Word2Vec embeddings** (Skip-gram, 100-D) to classify news headlines and full articles into five target categories:\n",
                "1. **Politics**\n",
                "2. **Business**\n",
                "3. **Technology**\n",
                "4. **Sports**\n",
                "5. **Entertainment**\n",
                "\n",
                "---\n",
                "\n",
                "### Conceptual Foundations: Long Short-Term Memory (LSTM)\n",
                "\n",
                "#### 1. Why Ordinary RNN Struggles with Long-Term Dependencies\n",
                "In Phase 5, we observed that a standard Simple RNN achieved only **37.16% test accuracy**, severely lagging behind classical TF-IDF models (~83.77%).\n",
                "- In a standard RNN, the hidden state update is strictly multiplicative:\n",
                "  $$h_t = \\tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$\n",
                "- Backpropagating through 150 time steps requires repeatedly multiplying by $W_{hh}^T$. Because $|\tanh'(z)| \\le 1$, if the spectral radius of $W_{hh}$ is below 1, gradients decay exponentially to $0$.\n",
                "- **Consequence:** The network suffers from catastrophic recency bias, completely forgetting foundational topic keywords located in the opening sentences of an article.\n",
                "\n",
                "#### 2. The LSTM Architecture Overview\n",
                "Introduced by Hochreiter & Schmidhuber (1997), LSTM overcomes vanishing gradients by splitting memory into two distinct paths:\n",
                "1. **Hidden State ($h_t$):** The short-term working output passed to subsequent layers.\n",
                "2. **Cell State ($C_t$):** An internal, long-term memory conveyor belt that allows gradients to flow linearly without exponential decay.\n",
                "\n",
                "```\n",
                "                   [ Cell State Highway: C_{t-1} ---> (x) -------------> (+) ---------------> C_t ]\n",
                "                                                       ^                  ^\n",
                "                                                       | [f_t]            | [i_t * C_tilde]\n",
                "                                                   [Forget Gate]      [Input Gate]\n",
                "                                                       |                  |\n",
                "  [ Hidden State:  h_{t-1} ] ---+----------------------+------------------+-----------+------> h_t\n",
                "                                |                                                     |\n",
                "  [ Word Vector:   x_t     ] ---+                                                [Output Gate]\n",
                "                                                                                      |\n",
                "                                                                                    [o_t]\n",
                "```\n",
                "\n",
                "#### 3. Step-by-Step Gating Formulations with Numerical Intuition\n",
                "\n",
                "##### A. The Forget Gate ($f_t$)\n",
                "Decides what fraction of information from previous cell state $C_{t-1}$ to throw away.\n",
                "$$f_t = \\sigma(W_f \\cdot [h_{t-1}, x_t] + b_f)$$\n",
                "- Range: $[0, 1]$ via the sigmoid activation $\\sigma$.\n",
                "- **Numerical Example:** If an article transitions from discussing corporate earnings to introducing an executive's political campaign, $f_t$ for finance-specific memory drops to $0.05$ (erasing past context), while general entity memory stays at $0.95$.\n",
                "\n",
                "##### B. The Input Gate ($i_t$) & Candidate State ($\\tilde{C}_t$)\n",
                "Decides which new information from the current token $x_t$ should be stored in the cell state.\n",
                "$$i_t = \\sigma(W_i \\cdot [h_{t-1}, x_t] + b_i)$$\n",
                "$$\\tilde{C}_t = \\tanh(W_c \\cdot [h_{t-1}, x_t] + b_c)$$\n",
                "- $i_t \\in [0, 1]$ acts as an attention filter.\n",
                "- $\\tilde{C}_t \\in [-1, 1]$ represents the candidate values to add.\n",
                "\n",
                "##### C. Updating the Cell State ($C_t$)\n",
                "The cell state update is **additive**, which is the fundamental mathematical breakthrough of LSTM:\n",
                "$$C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$$\n",
                "\n",
                "##### D. The Output Gate ($o_t$) & Hidden State ($h_t$)\n",
                "Decides what part of the cell state to output as the hidden state $h_t$:\n",
                "$$o_t = \\sigma(W_o \\cdot [h_{t-1}, x_t] + b_o)$$\n",
                "$$h_t = o_t \\odot \\tanh(C_t)$$\n",
                "\n",
                "#### 4. How LSTM Solves the Vanishing Gradient Problem\n",
                "In standard RNN, $\\frac{\\partial h_t}{\\partial h_{t-1}} = \\tanh'(...) W_{hh}^T$, leading to exponential decay.\n",
                "In LSTM, the gradient across cell states is:\n",
                "$$\\frac{\\partial C_t}{\\partial C_{t-1}} = f_t$$\n",
                "- If the forget gate $f_t \\approx 1$, the gradient flows back through hundreds of sequence steps without shrinking!\n",
                "- This creates a **Constant Error Carousel (CEC)**, allowing the network to retain critical words from the very beginning of a long news article."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 1: Environment Setup & Hardware Verification"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import json\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import tensorflow as tf\n",
                "from tensorflow import keras\n",
                "from keras import layers, models, callbacks\n",
                "\n",
                "# Set random seeds for strict reproducibility\n",
                "np.random.seed(42)\n",
                "tf.random.set_seed(42)\n",
                "\n",
                "print(f\"TensorFlow Version: {tf.__version__}\")\n",
                "print(f\"Keras Version:      {keras.__version__}\")\n",
                "\n",
                "gpus = tf.config.list_physical_devices('GPU')\n",
                "if gpus:\n",
                "    print(f\"[GPU Detected]: {gpus[0].name} - Hardware acceleration enabled!\")\n",
                "    for gpu in gpus:\n",
                "        tf.config.experimental.set_memory_growth(gpu, True)\n",
                "else:\n",
                "    print(\"[Hardware Notice]: No GPU detected. Running on CPU runtime.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Optional: Google Drive Mount (for Google Colab Users)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Mount Google Drive if running in Colab\n",
                "try:\n",
                "    from google.colab import drive\n",
                "    drive.mount('/content/drive')\n",
                "    COLAB_ENV = True\n",
                "    print(\"Mounted Google Drive successfully.\")\n",
                "except Exception:\n",
                "    COLAB_ENV = False\n",
                "    print(\"Running locally / non-Colab environment.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 2: Load Dataset, Splits, and Pretrained Word2Vec Embeddings\n",
                "To ensure strict scientific consistency and prevent data leakage, we reuse:\n",
                "1. The identical 70% Train, 15% Validation, and 15% Test split (`random_state=42`).\n",
                "2. The exact Phase 4 vocabulary mapping (`models/tokenizer_word2idx.json`).\n",
                "3. The pretrained 100-D Word2Vec Skip-gram embedding matrix (`models/embedding_matrix.npy`).\n",
                "4. Sequence configuration (`max_seq_len = 150`, post-padding)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.model_selection import train_test_split\n",
                "from src.preprocessing import NewsPreprocessor\n",
                "from src.sequence_utils import pad_sequences_manual, texts_to_sequences\n",
                "\n",
                "# Load cleaned dataset\n",
                "data_path = \"dataset/processed/news_dataset_cleaned.csv\"\n",
                "df = pd.read_csv(data_path)\n",
                "print(f\"Loaded dataset: {len(df)} articles.\")\n",
                "\n",
                "# Preprocess\n",
                "preprocessor = NewsPreprocessor(method='lemmatize', remove_stopwords=True, use_pos=True)\n",
                "df['processed_text'] = preprocessor.transform_series(df['text'])\n",
                "df = df[df['processed_text'].str.strip().str.len() > 5].reset_index(drop=True)\n",
                "\n",
                "# Strict Stratified Split: Train 70%, Val 15%, Test 15% (random_state=42)\n",
                "X_train_val, X_test, y_train_val, y_test = train_test_split(\n",
                "    df['processed_text'], df['category'],\n",
                "    test_size=0.15,\n",
                "    random_state=42,\n",
                "    stratify=df['category']\n",
                ")\n",
                "X_train, X_val, y_train, y_val = train_test_split(\n",
                "    X_train_val, y_train_val,\n",
                "    test_size=(0.15 / 0.85),\n",
                "    random_state=42,\n",
                "    stratify=y_train_val\n",
                ")\n",
                "\n",
                "print(f\"Training set:   {len(X_train)} samples (70%)\")\n",
                "print(f\"Validation set: {len(X_val)} samples (15%)\")\n",
                "print(f\"Test set:       {len(X_test)} samples (15%)\")\n",
                "\n",
                "# Load Saved Sequence Artifacts\n",
                "with open(\"models/tokenizer_word2idx.json\", \"r\", encoding=\"utf-8\") as f:\n",
                "    word2idx = json.load(f)\n",
                "with open(\"models/sequence_config.json\", \"r\", encoding=\"utf-8\") as f:\n",
                "    seq_config = json.load(f)\n",
                "embedding_matrix = np.load(\"models/embedding_matrix.npy\")\n",
                "\n",
                "print(f\"Vocabulary size:     {len(word2idx)}\")\n",
                "print(f\"Embedding matrix:    {embedding_matrix.shape}\")\n",
                "print(f\"Max sequence length: {seq_config['max_seq_len']}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 3: Sequence Conversion & Label Encoding"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "CATEGORIES = [\"Politics\", \"Business\", \"Technology\", \"Sports\", \"Entertainment\"]\n",
                "CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}\n",
                "IDX_TO_CAT = {i: cat for i, cat in enumerate(CATEGORIES)}\n",
                "\n",
                "max_len = seq_config[\"max_seq_len\"]\n",
                "padding = seq_config.get(\"padding\", \"post\")\n",
                "truncating = seq_config.get(\"truncating\", \"post\")\n",
                "\n",
                "# Convert text to integer sequences with padding\n",
                "train_tokens = [doc.split() for doc in X_train]\n",
                "val_tokens = [doc.split() for doc in X_val]\n",
                "test_tokens = [doc.split() for doc in X_test]\n",
                "\n",
                "X_train_seq = pad_sequences_manual(texts_to_sequences(train_tokens, word2idx), max_len, padding, truncating)\n",
                "X_val_seq = pad_sequences_manual(texts_to_sequences(val_tokens, word2idx), max_len, padding, truncating)\n",
                "X_test_seq = pad_sequences_manual(texts_to_sequences(test_tokens, word2idx), max_len, padding, truncating)\n",
                "\n",
                "# Integer encoded labels\n",
                "y_train_enc = np.array([CAT_TO_IDX[cat] for cat in y_train], dtype=np.int32)\n",
                "y_val_enc = np.array([CAT_TO_IDX[cat] for cat in y_val], dtype=np.int32)\n",
                "y_test_enc = np.array([CAT_TO_IDX[cat] for cat in y_test], dtype=np.int32)\n",
                "\n",
                "print(f\"X_train shape: {X_train_seq.shape}, y_train shape: {y_train_enc.shape}\")\n",
                "print(f\"X_val shape:   {X_val_seq.shape}, y_val shape:   {y_val_enc.shape}\")\n",
                "print(f\"X_test shape:  {X_test_seq.shape}, y_test shape:  {y_test_enc.shape}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 4: LSTM Model Architecture Definition\n",
                "\n",
                "The network architecture is composed of:\n",
                "1. **Input Layer:** Integer sequences of length $T = 150$.\n",
                "2. **Embedding Layer:** Pretrained 100-D Word2Vec Skip-gram weights.\n",
                "3. **LSTM Layer:** Gated recurrent memory units with internal forget, input, and output gates.\n",
                "4. **Dense Layer:** 32 units with ReLU activation.\n",
                "5. **Dropout Layer:** Regularization to avoid overfitting on high-dimensional text.\n",
                "6. **Classification Output Layer:** 5 units with Softmax activation."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def build_lstm_classifier(vocab_size, embedding_dim, max_len, embedding_matrix, lstm_units=64, dropout_rate=0.3, trainable_embeddings=False, name=\"LSTM_Classifier\"):\n",
                "    \"\"\"\n",
                "    Constructs a compiled Keras LSTM news classification model.\n",
                "    \"\"\"\n",
                "    model = models.Sequential([\n",
                "        layers.Input(shape=(max_len,), name=\"sequence_input\"),\n",
                "        layers.Embedding(\n",
                "            input_dim=vocab_size,\n",
                "            output_dim=embedding_dim,\n",
                "            weights=[embedding_matrix],\n",
                "            trainable=trainable_embeddings,\n",
                "            name=\"word2vec_embedding\"\n",
                "        ),\n",
                "        layers.LSTM(\n",
                "            units=lstm_units,\n",
                "            dropout=0.2,\n",
                "            return_sequences=False,\n",
                "            name=\"lstm_layer\"\n",
                "        ),\n",
                "        layers.Dense(32, activation='relu', name=\"dense_layer\"),\n",
                "        layers.Dropout(dropout_rate, name=\"dropout_layer\"),\n",
                "        layers.Dense(5, activation='softmax', name=\"softmax_output\")\n",
                "    ], name=name)\n",
                "    \n",
                "    optimizer = keras.optimizers.Adam(learning_rate=1e-3)\n",
                "    model.compile(\n",
                "        optimizer=optimizer,\n",
                "        loss='sparse_categorical_crossentropy',\n",
                "        metrics=['accuracy']\n",
                "    )\n",
                "    return model\n",
                "\n",
                "preview_model = build_lstm_classifier(len(word2idx), 100, max_len, embedding_matrix, lstm_units=64)\n",
                "preview_model.summary()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 5: Hyperparameter Tuning on Validation Data Only\n",
                "We compare:\n",
                "1. **LSTM-64 (Frozen):** 64 hidden units, dropout 0.3, frozen Word2Vec.\n",
                "2. **LSTM-128 (Frozen):** 128 hidden units, dropout 0.3, frozen Word2Vec.\n",
                "3. **LSTM (Fine-Tuned):** Best unit size with trainable Word2Vec embeddings."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "early_stop = callbacks.EarlyStopping(\n",
                "    monitor='val_loss',\n",
                "    patience=4,\n",
                "    restore_best_weights=True,\n",
                "    verbose=1\n",
                ")\n",
                "\n",
                "# Experiment 1: LSTM-64 (Frozen)\n",
                "print(\"=\"*65)\n",
                "print(\"EXPERIMENT 1: LSTM-64 with Frozen Word2Vec Embeddings\")\n",
                "print(\"=\"*65)\n",
                "model_64 = build_lstm_classifier(len(word2idx), 100, max_len, embedding_matrix, lstm_units=64, dropout_rate=0.3, trainable_embeddings=False, name=\"LSTM_64_Frozen\")\n",
                "history_64 = model_64.fit(\n",
                "    X_train_seq, y_train_enc,\n",
                "    validation_data=(X_val_seq, y_val_enc),\n",
                "    epochs=15,\n",
                "    batch_size=64,\n",
                "    callbacks=[early_stop],\n",
                "    verbose=1\n",
                ")\n",
                "loss_64, acc_64 = model_64.evaluate(X_val_seq, y_val_enc, verbose=0)\n",
                "print(f\"Experiment 1 - Val Accuracy: {acc_64*100:.2f}%, Val Loss: {loss_64:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Experiment 2: LSTM-128 (Frozen)\n",
                "print(\"=\"*65)\n",
                "print(\"EXPERIMENT 2: LSTM-128 with Frozen Word2Vec Embeddings\")\n",
                "print(\"=\"*65)\n",
                "model_128 = build_lstm_classifier(len(word2idx), 100, max_len, embedding_matrix, lstm_units=128, dropout_rate=0.3, trainable_embeddings=False, name=\"LSTM_128_Frozen\")\n",
                "history_128 = model_128.fit(\n",
                "    X_train_seq, y_train_enc,\n",
                "    validation_data=(X_val_seq, y_val_enc),\n",
                "    epochs=15,\n",
                "    batch_size=64,\n",
                "    callbacks=[early_stop],\n",
                "    verbose=1\n",
                ")\n",
                "loss_128, acc_128 = model_128.evaluate(X_val_seq, y_val_enc, verbose=0)\n",
                "print(f\"Experiment 2 - Val Accuracy: {acc_128*100:.2f}%, Val Loss: {loss_128:.4f}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Experiment 3: LSTM with Fine-Tuned Word2Vec Embeddings\n",
                "best_u = 128 if acc_128 > acc_64 else 64\n",
                "print(\"=\"*65)\n",
                "print(f\"EXPERIMENT 3: LSTM-{best_u} with Fine-Tuned Word2Vec Embeddings (Dropout=0.4)\")\n",
                "print(\"=\"*65)\n",
                "model_ft = build_lstm_classifier(len(word2idx), 100, max_len, embedding_matrix, lstm_units=best_u, dropout_rate=0.4, trainable_embeddings=True, name=f\"LSTM_{best_u}_FineTuned\")\n",
                "\n",
                "checkpoint = callbacks.ModelCheckpoint(\n",
                "    \"models/lstm_model.keras\",\n",
                "    monitor='val_loss',\n",
                "    save_best_only=True,\n",
                "    verbose=0\n",
                ")\n",
                "history_ft = model_ft.fit(\n",
                "    X_train_seq, y_train_enc,\n",
                "    validation_data=(X_val_seq, y_val_enc),\n",
                "    epochs=15,\n",
                "    batch_size=64,\n",
                "    callbacks=[early_stop, checkpoint],\n",
                "    verbose=1\n",
                ")\n",
                "loss_ft, acc_ft = model_ft.evaluate(X_val_seq, y_val_enc, verbose=0)\n",
                "print(f\"Experiment 3 - Val Accuracy: {acc_ft*100:.2f}%, Val Loss: {loss_ft:.4f}\")\n",
                "\n",
                "# Determine overall best model\n",
                "candidates = [\n",
                "    (\"LSTM-64 (Frozen)\", acc_64, loss_64, model_64, history_64),\n",
                "    (\"LSTM-128 (Frozen)\", acc_128, loss_128, model_128, history_128),\n",
                "    (f\"LSTM-{best_u} (Fine-Tuned)\", acc_ft, loss_ft, model_ft, history_ft)\n",
                "]\n",
                "candidates.sort(key=lambda x: x[1], reverse=True)\n",
                "best_name, best_acc, best_loss, best_model, best_history = candidates[0]\n",
                "print(f\"\\n--> Selected Best Model: {best_name} with Validation Accuracy: {best_acc*100:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 6: Learning Dynamics & Training Visualizations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "plt.figure(figsize=(14, 5))\n",
                "\n",
                "# Cross-entropy loss\n",
                "plt.subplot(1, 2, 1)\n",
                "plt.plot(best_history.history['loss'], label='Training Loss', color='#1f77b4', lw=2)\n",
                "plt.plot(best_history.history['val_loss'], label='Validation Loss', color='#d62728', lw=2, linestyle='--')\n",
                "plt.title(f\"{best_name} - Cross-Entropy Loss\", fontsize=12, fontweight='bold')\n",
                "plt.xlabel(\"Epoch\", fontsize=10)\n",
                "plt.ylabel(\"Loss\", fontsize=10)\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.legend(frameon=True)\n",
                "\n",
                "# Classification accuracy\n",
                "plt.subplot(1, 2, 2)\n",
                "plt.plot(best_history.history['accuracy'], label='Training Accuracy', color='#2ca02c', lw=2)\n",
                "plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f0e', lw=2, linestyle='--')\n",
                "plt.title(f\"{best_name} - Classification Accuracy\", fontsize=12, fontweight='bold')\n",
                "plt.xlabel(\"Epoch\", fontsize=10)\n",
                "plt.ylabel(\"Accuracy\", fontsize=10)\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.legend(frameon=True)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 7: Final Test Set Evaluation & Confusion Matrix\n",
                "We evaluate the selected best model on the unseen test set (1,811 articles)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix\n",
                "\n",
                "y_test_probs = best_model.predict(X_test_seq)\n",
                "y_test_preds = np.argmax(y_test_probs, axis=1)\n",
                "\n",
                "acc = accuracy_score(y_test_enc, y_test_preds)\n",
                "prec_macro = precision_score(y_test_enc, y_test_preds, average='macro')\n",
                "rec_macro = recall_score(y_test_enc, y_test_preds, average='macro')\n",
                "f1_macro = f1_score(y_test_enc, y_test_preds, average='macro')\n",
                "f1_weighted = f1_score(y_test_enc, y_test_preds, average='weighted')\n",
                "\n",
                "print(\"=\"*55)\n",
                "print(f\"TEST EVALUATION METRICS: {best_name}\")\n",
                "print(\"=\"*55)\n",
                "print(f\"Accuracy:           {acc:.4f} ({acc*100:.2f}%)\")\n",
                "print(f\"Macro Precision:    {prec_macro:.4f}\")\n",
                "print(f\"Macro Recall:       {rec_macro:.4f}\")\n",
                "print(f\"Macro F1-Score:     {f1_macro:.4f}\")\n",
                "print(f\"Weighted F1-Score:  {f1_weighted:.4f}\")\n",
                "\n",
                "print(\"\\nDetailed Classification Report:\")\n",
                "print(classification_report(y_test_enc, y_test_preds, target_names=CATEGORIES))\n",
                "\n",
                "# Confusion Matrix\n",
                "cm = confusion_matrix(y_test_enc, y_test_preds)\n",
                "plt.figure(figsize=(8, 6))\n",
                "sns.heatmap(\n",
                "    cm, annot=True, fmt='d', cmap='Greens',\n",
                "    xticklabels=CATEGORIES, yticklabels=CATEGORIES,\n",
                "    cbar=True\n",
                ")\n",
                "plt.title(f\"{best_name} - Confusion Matrix (Test Set)\", fontsize=13, fontweight='bold', pad=12)\n",
                "plt.xlabel(\"Predicted Category\", fontsize=11)\n",
                "plt.ylabel(\"True Category\", fontsize=11)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 8: Error Analysis & Category Distinguishability\n",
                "We inspect which categories are most frequently confused by the LSTM."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "analysis_df = pd.DataFrame({\n",
                "    'text': X_test.values,\n",
                "    'true_category': [IDX_TO_CAT[i] for i in y_test_enc],\n",
                "    'pred_category': [IDX_TO_CAT[i] for i in y_test_preds],\n",
                "    'confidence': np.max(y_test_probs, axis=1)\n",
                "})\n",
                "\n",
                "misclassified = analysis_df[analysis_df['true_category'] != analysis_df['pred_category']]\n",
                "print(f\"Total misclassified test articles: {len(misclassified)} out of {len(analysis_df)} ({len(misclassified)/len(analysis_df)*100:.2f}%)\\n\")\n",
                "\n",
                "print(\"Confusion Pairs Breakdown:\")\n",
                "pair_counts = misclassified.groupby(['true_category', 'pred_category']).size().reset_index(name='count').sort_values('count', ascending=False)\n",
                "print(pair_counts.head(8).to_string(index=False))\n",
                "\n",
                "print(\"\\nSample Misclassified Articles:\")\n",
                "for i, row in misclassified.head(4).iterrows():\n",
                "    print(f\"True: [{row['true_category']}] -> Pred: [{row['pred_category']}] (Confidence: {row['confidence']:.2f})\")\n",
                "    print(f\"Snippet: {row['text'][:140]}...\\n\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Step 9: Save Artifacts & Google Drive Export"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "os.makedirs(\"models\", exist_ok=True)\n",
                "best_model.save(\"models/lstm_model.keras\")\n",
                "print(\"Saved primary model to models/lstm_model.keras\")\n",
                "\n",
                "if COLAB_ENV:\n",
                "    import shutil\n",
                "    drive_dest = \"/content/drive/MyDrive/Intelligent_News_Categorization/models/\"\n",
                "    os.makedirs(drive_dest, exist_ok=True)\n",
                "    shutil.copy(\"models/lstm_model.keras\", drive_dest)\n",
                "    print(f\"Successfully saved LSTM model to Google Drive at {drive_dest}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 4,
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join("notebooks", "06_lstm_classifier.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated {out_path} with {len(notebook['cells'])} cells.")
