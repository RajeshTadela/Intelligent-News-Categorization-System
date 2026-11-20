"""
Unified Inference and Prediction Engine
Phase 8: Intelligent News Categorization System

Provides robust prediction, calibrated confidence quantification,
uncertainty rejection thresholding, and model explainability across:
- TF-IDF + Linear SVM (Calibrated)
- TF-IDF + Logistic Regression
- Word2Vec + LSTM
- Word2Vec + Simple RNN
"""

import time
import numpy as np
from src.model_loader import get_model_manager, CATEGORIES, CAT_TO_IDX, IDX_TO_CAT
from src.preprocessing import NewsPreprocessor
from src.sequence_utils import pad_sequences_manual, texts_to_sequences

class NewsPredictor:
    def __init__(self, models_dir="models"):
        self.manager = get_model_manager(models_dir=models_dir)
        self.preprocessor = NewsPreprocessor(method='lemmatize', remove_stopwords=True, use_pos=True)

    def predict(self, raw_text: str, model_name: str = "TF-IDF + Linear SVM (Calibrated)", rejection_threshold: float = 0.50):
        """
        Executes end-to-end classification with uncertainty rejection and explainability.
        """
        start_time = time.time()
        
        # 1. Edge Case: Empty or Whitespace Text
        if not raw_text or not raw_text.strip():
            return {
                "status": "ERROR",
                "message": "Input text is empty. Please enter a news headline or full article text.",
                "display_prediction": "No input provided",
                "is_confident": False
            }
            
        words_count = len(raw_text.strip().split())
        if words_count < 2:
            return {
                "status": "ERROR",
                "message": "Input is too short (less than 2 words). Please provide a complete sentence or headline.",
                "display_prediction": "Insufficient text",
                "is_confident": False
            }

        # 2. Preprocess text identical to training pipeline
        cleaned_text = self.preprocessor.transform(raw_text)
        if not cleaned_text.strip():
            return {
                "status": "ERROR",
                "message": "After removing stopwords and punctuation, no valid vocabulary words remained.",
                "display_prediction": "No meaningful words",
                "is_confident": False
            }

        model_entry = self.manager.get_model_info(model_name)
        if not model_entry:
            return {
                "status": "ERROR",
                "message": f"Model '{model_name}' is not currently available or failed to load.",
                "display_prediction": "Model unavailable",
                "is_confident": False
            }

        model = model_entry["model"]
        model_type = model_entry["type"]
        
        explanation = {}
        all_probs = {}

        # 3. Execution Branch: TF-IDF vs Sequential Neural Net
        if model_type == "tfidf":
            tfidf_vec = self.manager.tfidf_vectorizer
            X_vec = tfidf_vec.transform([cleaned_text])
            
            # Predict probabilities (Linear SVM is calibrated via CalibratedClassifierCV)
            probs = model.predict_proba(X_vec)[0]
            classes = list(model.classes_)
            top_idx = int(np.argmax(probs))
            predicted_class = classes[top_idx]
            confidence = float(probs[top_idx])
            
            for i, cat in enumerate(classes):
                all_probs[cat] = float(probs[i])
                
            # Explainability: Extract top influential words
            try:
                feature_names = np.array(tfidf_vec.get_feature_names_out())
                row = X_vec.tocoo()
                word_indices = row.col
                word_values = row.data
                
                # Check if base estimator has coef_
                estimator = model
                if hasattr(model, "calibrated_classifiers_"):
                    # CalibratedClassifierCV
                    base = model.calibrated_classifiers_[0].estimator
                    if hasattr(base, "coef_"):
                        estimator = base
                elif hasattr(model, "estimator"):
                    estimator = model.estimator
                    
                if hasattr(estimator, "coef_"):
                    coef = estimator.coef_[top_idx]
                    word_scores = []
                    for idx, val in zip(word_indices, word_values):
                        word = feature_names[idx]
                        score = coef[idx] * val
                        if score > 0:
                            word_scores.append((word, score))
                    word_scores.sort(key=lambda x: x[1], reverse=True)
                    explanation["top_words"] = [w for w, s in word_scores[:6]]
                else:
                    active_words = [feature_names[i] for i in word_indices[:6]]
                    explanation["top_words"] = active_words
            except Exception:
                explanation["top_words"] = cleaned_text.split()[:5]

        else:
            # Sequential Recurrent Model (RNN / LSTM)
            word2idx = self.manager.word2idx
            seq_config = self.manager.seq_config
            max_len = seq_config.get("max_seq_len", 150)
            padding = seq_config.get("padding", "post")
            truncating = seq_config.get("truncating", "post")
            
            tokens = cleaned_text.split()
            seq = [word2idx.get(t, 1) for t in tokens]  # 1 is <OOV>
            
            # Count in-vocab vs OOV
            oov_tokens = [t for t in tokens if t not in word2idx]
            in_vocab_tokens = [t for t in tokens if t in word2idx]
            
            padded = pad_sequences_manual([seq], max_len=max_len, padding=padding, truncating=truncating)
            probs = model.predict(padded, verbose=0)[0]
            keras_classes = ["Politics", "Business", "Technology", "Sports", "Entertainment"]
            top_idx = int(np.argmax(probs))
            predicted_class = keras_classes[top_idx]
            confidence = float(probs[top_idx])
            
            for i, cat in enumerate(keras_classes):
                all_probs[cat] = float(probs[i])
                
            explanation["in_vocab_words"] = in_vocab_tokens[:6]
            explanation["oov_words"] = oov_tokens[:4]
            explanation["token_count"] = len(tokens)

        elapsed_ms = (time.time() - start_time) * 1000

        # 4. Uncertainty Check & Rejection Thresholding
        if confidence < rejection_threshold:
            is_confident = False
            display_pred = "Unable to confidently classify this article."
            uncertainty_reason = (
                f"Maximum model probability ({confidence*100:.1f}%) is below the confidence "
                f"threshold ({rejection_threshold*100:.1f}%). The article may contain mixed topics, "
                f"insufficient category-specific vocabulary, or ambiguous wording."
            )
        else:
            is_confident = True
            display_pred = predicted_class
            uncertainty_reason = None

        return {
            "status": "SUCCESS",
            "model_name": model_name,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "predicted_category": predicted_class,
            "display_prediction": display_pred,
            "confidence": confidence,
            "is_confident": is_confident,
            "uncertainty_reason": uncertainty_reason,
            "all_probabilities": all_probs,
            "explanation": explanation,
            "latency_ms": elapsed_ms,
            "words_count": words_count
        }

# Global predictor instance
_GLOBAL_PREDICTOR = None

def get_news_predictor(models_dir="models"):
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = NewsPredictor(models_dir=models_dir)
    return _GLOBAL_PREDICTOR
