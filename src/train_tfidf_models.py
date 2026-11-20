"""
TF-IDF Feature Extraction and Classical ML Training Pipeline
Phase 3: Intelligent News Categorization System

Models:
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear Support Vector Machine (LinearSVC with CalibratedClassifierCV)

Strict anti-leakage:
- Stratified 70% Train, 15% Validation, 15% Test split
- TF-IDF fitted STRICTLY on training split
- Validation and Test splits transformed using training-fitted vectorizer
- Hyperparameter tuning performed on Validation split
- Final unbiased evaluation conducted on Test split
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

from src.preprocessing import NewsPreprocessor

# Output directories
os.makedirs("models", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)

CATEGORIES = ["Politics", "Business", "Technology", "Sports", "Entertainment"]

def load_and_preprocess_data(data_path="dataset/processed/news_dataset_cleaned.csv"):
    print(f"Loading cleaned dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows.")
    
    # Preprocess texts using our Phase 2 lemmatizing preprocessor
    print("Applying text preprocessing (lemmatization + custom stopword filtering)...")
    preprocessor = NewsPreprocessor(method='lemmatize', remove_stopwords=True, use_pos=True)
    df['processed_text'] = preprocessor.transform_series(df['text'])
    
    # Drop any edge-case empty strings after preprocessing
    df = df[df['processed_text'].str.strip().str.len() > 5].reset_index(drop=True)
    print(f"Dataset ready for modeling: {len(df)} samples across 5 categories.")
    return df

def split_dataset(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Stratified split into Train (70%), Validation (15%), Test (15%).
    Prevents any data leakage across splits.
    """
    print("\nExecuting stratified train-validation-test split...")
    # First split: Train+Val (85%) and Test (15%)
    X_train_val, X_test, y_train_val, y_test, idx_train_val, idx_test = train_test_split(
        df['processed_text'], df['category'], df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df['category']
    )
    
    # Second split: Train (70% of total) and Val (15% of total)
    # val_size / (1 - test_size) = 0.15 / 0.85 approx 0.17647
    adjusted_val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train_val, y_train_val, idx_train_val,
        test_size=adjusted_val_ratio,
        random_state=random_state,
        stratify=y_train_val
    )
    
    print(f"Training set:   {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"Test set:       {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    
    # Check class distributions
    print("\nClass distributions in Train set:")
    print(y_train.value_counts(normalize=True).round(3))
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def experiment_tfidf_vectorizers(X_train, X_val, y_train, y_val):
    """
    Evaluates different TF-IDF configurations on Validation set
    to determine optimal max_features, ngram_range, min_df, max_df.
    """
    print("\n=== EXPERIMENTING WITH TF-IDF CONFIGURATIONS ===")
    configs = [
        {"name": "Unigrams (max=5k)", "ngram_range": (1, 1), "max_features": 5000, "min_df": 2, "max_df": 0.85},
        {"name": "Unigrams (max=10k)", "ngram_range": (1, 1), "max_features": 10000, "min_df": 2, "max_df": 0.85},
        {"name": "Unigrams+Bigrams (max=10k)", "ngram_range": (1, 2), "max_features": 10000, "min_df": 2, "max_df": 0.85},
        {"name": "Unigrams+Bigrams (max=15k)", "ngram_range": (1, 2), "max_features": 15000, "min_df": 2, "max_df": 0.85},
        {"name": "Unigrams+Bigrams (max=20k)", "ngram_range": (1, 2), "max_features": 20000, "min_df": 2, "max_df": 0.85},
    ]
    
    results = []
    best_config = None
    best_val_f1 = 0
    best_vectorizer = None
    
    for cfg in configs:
        vec = TfidfVectorizer(
            ngram_range=cfg["ngram_range"],
            max_features=cfg["max_features"],
            min_df=cfg["min_df"],
            max_df=cfg["max_df"],
            sublinear_tf=True
        )
        # FIT ONLY ON TRAINING DATA
        X_tr_vec = vec.fit_transform(X_train)
        X_va_vec = vec.transform(X_val)
        
        # Fast baseline evaluation with Logistic Regression
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_tr_vec, y_train)
        preds = clf.predict(X_va_vec)
        
        acc = accuracy_score(y_val, preds)
        macro_f1 = f1_score(y_val, preds, average='macro')
        
        results.append({
            "Config": cfg["name"],
            "Vocab Size": len(vec.vocabulary_),
            "Val Accuracy": round(acc, 4),
            "Val Macro F1": round(macro_f1, 4)
        })
        print(f"[{cfg['name']}] Vocab: {len(vec.vocabulary_)} | Val Acc: {acc:.4f} | Val Macro F1: {macro_f1:.4f}")
        
        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_config = cfg
            best_vectorizer = vec
            
    df_results = pd.DataFrame(results)
    print("\nSummary of TF-IDF Configuration Search:")
    print(df_results.to_string(index=False))
    print(f"\nOptimal configuration: {best_config['name']} with Val Macro F1: {best_val_f1:.4f}")
    return best_vectorizer, best_config

def tune_and_train_models(X_train_vec, X_val_vec, y_train, y_val):
    """
    Tunes regularization parameter C for Logistic Regression and Linear SVM
    using the Validation set.
    """
    print("\n=== HYPERPARAMETER TUNING ON VALIDATION SET ===")
    c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # 1. Tuning Logistic Regression
    print("\n--- Tuning Logistic Regression ---")
    best_lr_c = 1.0
    best_lr_f1 = 0
    lr_tuning_results = []
    
    for c in c_values:
        lr = LogisticRegression(C=c, max_iter=1000, random_state=42, solver='lbfgs')
        lr.fit(X_train_vec, y_train)
        val_pred = lr.predict(X_val_vec)
        f1 = f1_score(y_val, val_pred, average='macro')
        acc = accuracy_score(y_val, val_pred)
        lr_tuning_results.append({"C": c, "Accuracy": acc, "Macro F1": f1})
        print(f"  C={c:4.1f} -> Val Acc: {acc:.4f}, Val Macro F1: {f1:.4f}")
        if f1 > best_lr_f1:
            best_lr_f1 = f1
            best_lr_c = c
            
    print(f"Optimal Logistic Regression C: {best_lr_c} (Macro F1: {best_lr_f1:.4f})")
    
    # 2. Tuning Linear SVM
    print("\n--- Tuning Linear SVM ---")
    best_svm_c = 1.0
    best_svm_f1 = 0
    svm_tuning_results = []
    
    for c in c_values:
        svm = LinearSVC(C=c, max_iter=2000, random_state=42)
        svm.fit(X_train_vec, y_train)
        val_pred = svm.predict(X_val_vec)
        f1 = f1_score(y_val, val_pred, average='macro')
        acc = accuracy_score(y_val, val_pred)
        svm_tuning_results.append({"C": c, "Accuracy": acc, "Macro F1": f1})
        print(f"  C={c:4.1f} -> Val Acc: {acc:.4f}, Val Macro F1: {f1:.4f}")
        if f1 > best_svm_f1:
            best_svm_f1 = f1
            best_svm_c = c
            
    print(f"Optimal Linear SVM C: {best_svm_c} (Macro F1: {best_svm_f1:.4f})")
    
    # Train Best Models
    print("\nTraining final models with optimal hyperparameters on Train set...")
    best_lr = LogisticRegression(C=best_lr_c, max_iter=1000, random_state=42)
    best_lr.fit(X_train_vec, y_train)
    
    # Train calibrated Linear SVM for probability calibration and confidence
    base_svm = LinearSVC(C=best_svm_c, max_iter=2000, random_state=42)
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, cv='prefit')
    base_svm.fit(X_train_vec, y_train)
    calibrated_svm.fit(X_val_vec, y_val)  # calibrate probabilities on validation partition
    
    return best_lr, base_svm, calibrated_svm, best_lr_c, best_svm_c

def evaluate_models_on_test(best_lr, calibrated_svm, base_svm, X_test_vec, y_test):
    """
    Unbiased final evaluation on completely unseen test partition.
    """
    print("\n=== FINAL UNBIASED EVALUATION ON TEST SET ===")
    
    # Predictions
    lr_preds = best_lr.predict(X_test_vec)
    svm_preds = calibrated_svm.predict(X_test_vec)
    
    # Metrics
    models = {
        "Logistic Regression": (lr_preds, best_lr),
        "Linear SVM (Calibrated)": (svm_preds, calibrated_svm)
    }
    
    metrics_summary = []
    
    for name, (preds, model) in models.items():
        acc = accuracy_score(y_test, preds)
        prec_macro = precision_score(y_test, preds, average='macro')
        rec_macro = recall_score(y_test, preds, average='macro')
        f1_macro = f1_score(y_test, preds, average='macro')
        f1_weighted = f1_score(y_test, preds, average='weighted')
        
        metrics_summary.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Macro Precision": round(prec_macro, 4),
            "Macro Recall": round(rec_macro, 4),
            "Macro F1": round(f1_macro, 4),
            "Weighted F1": round(f1_weighted, 4)
        })
        
        print(f"\n{'='*30} {name} Classification Report {'='*30}")
        print(classification_report(y_test, preds, target_names=CATEGORIES, digits=4))
        
    df_metrics = pd.DataFrame(metrics_summary)
    print("\nFinal Model Comparison Table:")
    print(df_metrics.to_string(index=False))
    
    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, (name, (preds, _)) in enumerate(models.items()):
        cm = confusion_matrix(y_test, preds, labels=CATEGORIES)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=axes[i])
        axes[i].set_title(f"Confusion Matrix: {name}", fontsize=12, fontweight='bold')
        axes[i].set_xlabel("Predicted Label", fontsize=11)
        axes[i].set_ylabel("True Label", fontsize=11)
        
    plt.tight_layout()
    cm_path = "evaluation/tfidf_models_confusion_matrices.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\nSaved confusion matrices to {cm_path}")
    
    return df_metrics

def extract_model_explainability(best_lr, vectorizer, top_n=10):
    """
    Extracts top positive feature weights / coefficients for each category.
    """
    print("\n=== MODEL EXPLAINABILITY: TOP TF-IDF FEATURES PER CATEGORY ===")
    feature_names = np.array(vectorizer.get_feature_names_out())
    classes = best_lr.classes_
    
    explainability_data = {}
    
    for i, cat in enumerate(classes):
        coefs = best_lr.coef_[i]
        top_positive_indices = np.argsort(coefs)[-top_n:][::-1]
        top_words = feature_names[top_positive_indices]
        top_weights = coefs[top_positive_indices]
        
        explainability_data[cat] = [
            {"feature": w, "weight": round(float(wt), 4)}
            for w, wt in zip(top_words, top_weights)
        ]
        
        print(f"\nCategory: [{cat.upper()}] Top Influential Features:")
        for w, wt in zip(top_words, top_weights):
            print(f"   {w:20s} (coeff: +{wt:.4f})")
            
    # Visualize top feature coefficients
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, 16))
    for i, cat in enumerate(classes):
        feats = [x["feature"] for x in explainability_data[cat]]
        weights = [x["weight"] for x in explainability_data[cat]]
        y_pos = np.arange(len(feats))
        axes[i].barh(y_pos, weights, align='center', color='steelblue')
        axes[i].set_yticks(y_pos)
        axes[i].set_yticklabels(feats, fontsize=10)
        axes[i].invert_yaxis()
        axes[i].set_title(f"Top 10 Indicative Features for {cat}", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Logistic Regression Coefficient")
        
    plt.tight_layout()
    feat_chart = "evaluation/tfidf_feature_importance.png"
    plt.savefig(feat_chart, dpi=300)
    plt.close()
    print(f"Saved feature importance chart to {feat_chart}")
    
    return explainability_data

def test_generalization_unseen(best_lr, calibrated_svm, vectorizer):
    """
    Tests trained models on synthetic/novel unseen articles across diverse lengths and styles.
    """
    print("\n=== GENERALIZATION TEST: UNSEEN DIVERSE ARTICLES ===")
    preprocessor = NewsPreprocessor(method='lemmatize', remove_stopwords=True, use_pos=True)
    
    test_cases = [
        {
            "category": "Politics",
            "style": "Short Headline",
            "text": "Senate committee passes bipartisan electoral integrity bill ahead of midnight deadline."
        },
        {
            "category": "Business",
            "style": "Corporate Analysis",
            "text": "Automaker reports quarterly revenue surge of 18 percent following unprecedented expansion in commercial fleet deliveries across European hubs."
        },
        {
            "category": "Technology",
            "style": "Technical Brief",
            "text": "Cybersecurity researchers uncover zero-day vulnerability in widespread open-source cryptographic library affecting enterprise cloud clusters."
        },
        {
            "category": "Sports",
            "style": "Match Recap",
            "text": "Underdog squad forces overtime penalty shootout after dramatic stoppage-time equalizer to eliminate reigning champions."
        },
        {
            "category": "Entertainment",
            "style": "Culture & Media",
            "text": "Streaming platform greenlights sequel series following record-breaking viewership numbers and widespread critical acclaim for debut season."
        }
    ]
    
    for case in test_cases:
        clean = preprocessor.transform(case["text"])
        vec = vectorizer.transform([clean])
        
        lr_pred = best_lr.predict(vec)[0]
        lr_probs = best_lr.predict_proba(vec)[0]
        lr_conf = np.max(lr_probs)
        
        svm_pred = calibrated_svm.predict(vec)[0]
        svm_probs = calibrated_svm.predict_proba(vec)[0]
        svm_conf = np.max(svm_probs)
        
        print("-" * 75)
        print(f"Expected: [{case['category']}] | Style: {case['style']}")
        print(f"Input:    \"{case['text']}\"")
        print(f"LR Pred:  {lr_pred} (Confidence: {lr_conf*100:.1f}%)")
        print(f"SVM Pred: {svm_pred} (Confidence: {svm_conf*100:.1f}%)")

def save_artifacts(vectorizer, best_lr, calibrated_svm, base_svm, best_config, best_lr_c, best_svm_c):
    print("\n=== SAVING MODEL ARTIFACTS ===")
    
    vec_path = "models/tfidf_vectorizer.joblib"
    lr_path = "models/logistic_regression_model.joblib"
    svm_path = "models/linear_svm_model.joblib"
    svm_base_path = "models/linear_svm_base.joblib"
    config_path = "models/tfidf_model_config.json"
    
    joblib.dump(vectorizer, vec_path)
    joblib.dump(best_lr, lr_path)
    joblib.dump(calibrated_svm, svm_path)
    joblib.dump(base_svm, svm_base_path)
    
    config = {
        "model_type": "TF-IDF + Classical ML",
        "categories": CATEGORIES,
        "tfidf_parameters": {
            "ngram_range": list(best_config["ngram_range"]),
            "max_features": best_config["max_features"],
            "min_df": best_config["min_df"],
            "max_df": best_config["max_df"],
            "sublinear_tf": True
        },
        "logistic_regression_parameters": {
            "C": best_lr_c,
            "max_iter": 1000,
            "solver": "lbfgs"
        },
        "linear_svm_parameters": {
            "C": best_svm_c,
            "max_iter": 2000,
            "calibration": "sigmoid (CalibratedClassifierCV)"
        }
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        
    print(f"Saved: {vec_path}")
    print(f"Saved: {lr_path}")
    print(f"Saved: {svm_path}")
    print(f"Saved: {config_path}")

if __name__ == "__main__":
    df = load_and_preprocess_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df)
    
    best_vectorizer, best_config = experiment_tfidf_vectorizers(X_train, X_val, y_train, y_val)
    
    # Transform using optimal vectorizer
    X_train_vec = best_vectorizer.transform(X_train)
    X_val_vec = best_vectorizer.transform(X_val)
    X_test_vec = best_vectorizer.transform(X_test)
    
    best_lr, base_svm, calibrated_svm, best_lr_c, best_svm_c = tune_and_train_models(
        X_train_vec, X_val_vec, y_train, y_val
    )
    
    df_metrics = evaluate_models_on_test(best_lr, calibrated_svm, base_svm, X_test_vec, y_test)
    explainability = extract_model_explainability(best_lr, best_vectorizer)
    test_generalization_unseen(best_lr, calibrated_svm, best_vectorizer)
    save_artifacts(best_vectorizer, best_lr, calibrated_svm, base_svm, best_config, best_lr_c, best_svm_c)
    print("\nPhase 3 Training Pipeline Completed Successfully!")
