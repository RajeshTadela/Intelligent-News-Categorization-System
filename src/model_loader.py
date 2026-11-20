"""
Model Loader and Artifact Manager
Phase 8: Intelligent News Categorization System

Loads and caches all trained models, vectorizers, tokenizers, and configuration files:
1. TF-IDF Vectorizer & Models (Logistic Regression, Linear SVM)
2. Word2Vec Embeddings & Sequential Tokenizers
3. Deep Learning Models (Simple RNN, LSTM)
"""

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

CATEGORIES = ["Politics", "Business", "Technology", "Sports", "Entertainment"]
CAT_TO_IDX = {cat: i for i, cat in enumerate(CATEGORIES)}
IDX_TO_CAT = {i: cat for i, cat in enumerate(CATEGORIES)}

CATEGORY_COLORS = {
    "Politics": "#e74c3c",       # Red
    "Business": "#3498db",       # Blue
    "Technology": "#9b59b6",     # Purple
    "Sports": "#2ecc71",         # Green
    "Entertainment": "#f39c12"   # Orange
}

class ModelManager:
    """Singleton manager that loads and caches model artifacts."""
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.tfidf_vectorizer = None
        self.word2idx = None
        self.seq_config = None
        self.class_mapping = None
        self._load_shared_components()
        self._load_available_models()

    def _load_shared_components(self):
        # 1. TF-IDF Vectorizer
        tfidf_path = os.path.join(self.models_dir, "tfidf_vectorizer.joblib")
        if os.path.exists(tfidf_path):
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            
        # 2. Tokenizer & Sequence Config
        tok_path = os.path.join(self.models_dir, "tokenizer_word2idx.json")
        if os.path.exists(tok_path):
            with open(tok_path, "r", encoding="utf-8") as f:
                self.word2idx = json.load(f)
                
        seq_path = os.path.join(self.models_dir, "sequence_config.json")
        if os.path.exists(seq_path):
            with open(seq_path, "r", encoding="utf-8") as f:
                self.seq_config = json.load(f)
                
        # 3. Class mapping
        map_path = os.path.join(self.models_dir, "class_mapping.json")
        if os.path.exists(map_path):
            with open(map_path, "r", encoding="utf-8") as f:
                self.class_mapping = json.load(f)

    def _load_available_models(self):
        # Model 1: Linear SVM (Calibrated) - Primary / Best Model
        svm_path = os.path.join(self.models_dir, "linear_svm_model.joblib")
        if os.path.exists(svm_path) and self.tfidf_vectorizer is not None:
            try:
                self.loaded_models["TF-IDF + Linear SVM (Calibrated)"] = {
                    "model": joblib.load(svm_path),
                    "type": "tfidf",
                    "description": "Linear Support Vector Classifier with Platt probability calibration (Recommended - Best Model).",
                    "accuracy": "83.77%",
                    "macro_f1": "0.8376"
                }
            except Exception as e:
                print(f"[Warning] Failed loading Linear SVM: {e}")

        # Model 2: Logistic Regression
        lr_path = os.path.join(self.models_dir, "logistic_regression_model.joblib")
        if os.path.exists(lr_path) and self.tfidf_vectorizer is not None:
            try:
                self.loaded_models["TF-IDF + Logistic Regression"] = {
                    "model": joblib.load(lr_path),
                    "type": "tfidf",
                    "description": "Multinomial Logistic Regression with L2 regularization and calibrated probabilities.",
                    "accuracy": "83.49%",
                    "macro_f1": "0.8351"
                }
            except Exception as e:
                print(f"[Warning] Failed loading Logistic Regression: {e}")

        # Model 3: Word2Vec + LSTM
        lstm_path = os.path.join(self.models_dir, "lstm_model.keras")
        if os.path.exists(lstm_path) and self.word2idx is not None:
            try:
                self.loaded_models["Word2Vec + LSTM"] = {
                    "model": keras.models.load_model(lstm_path),
                    "type": "sequence",
                    "description": "Long Short-Term Memory network with 128 units and 100-D Word2Vec Skip-gram embeddings.",
                    "accuracy": "40.81%",
                    "macro_f1": "0.3366"
                }
            except Exception as e:
                print(f"[Warning] Failed loading LSTM: {e}")

        # Model 4: Word2Vec + Simple RNN
        rnn_path = os.path.join(self.models_dir, "simple_rnn_model.keras")
        if os.path.exists(rnn_path) and self.word2idx is not None:
            try:
                self.loaded_models["Word2Vec + Simple RNN"] = {
                    "model": keras.models.load_model(rnn_path),
                    "type": "sequence",
                    "description": "Recurrent Neural Network with 64 units and frozen Word2Vec embeddings.",
                    "accuracy": "37.16%",
                    "macro_f1": "0.2828"
                }
            except Exception as e:
                print(f"[Warning] Failed loading Simple RNN: {e}")

    def get_available_model_names(self):
        return list(self.loaded_models.keys())

    def get_model_info(self, name):
        return self.loaded_models.get(name)

# Global singleton instance
_GLOBAL_MANAGER = None

def get_model_manager(models_dir="models"):
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = ModelManager(models_dir=models_dir)
    return _GLOBAL_MANAGER
