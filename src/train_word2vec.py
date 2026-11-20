"""
Word2Vec Training and Embedding Matrix Generation Pipeline
Phase 4: Intelligent News Categorization System

Models trained:
- Word2Vec CBOW (Continuous Bag of Words, sg=0)
- Word2Vec Skip-gram (sg=1)
- Embedding dimension experiments: 50, 100, 200

Strict Anti-Leakage:
- Word2Vec is trained EXCLUSIVELY on the 70% Training partition.
- Validation and Test splits are NEVER exposed to Word2Vec vocabulary learning.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

from src.preprocessing import NewsPreprocessor
from src.sequence_utils import build_vocabulary, build_embedding_matrix, pad_sequences_manual, texts_to_sequences

os.makedirs("models", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)

CATEGORIES = ["Politics", "Business", "Technology", "Sports", "Entertainment"]

def load_and_split_data(data_path="dataset/processed/news_dataset_cleaned.csv"):
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess
    print("Preprocessing text...")
    prep = NewsPreprocessor(method='lemmatize', remove_stopwords=True, use_pos=True)
    df['processed_text'] = prep.transform_series(df['text'])
    df = df[df['processed_text'].str.strip().str.len() > 5].reset_index(drop=True)
    
    # Split: Train 70%, Val 15%, Test 15% (identical to Phase 3)
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
    
    print(f"Training samples:   {len(X_train)} (70%)")
    print(f"Validation samples: {len(X_val)} (15%)")
    print(f"Test samples:       {len(X_test)} (15%)")
    
    # Tokenize training documents into word lists
    train_tokens = [doc.split() for doc in X_train]
    return train_tokens, X_train, X_val, X_test, y_train, y_val, y_test

def train_word2vec_experiments(train_tokens):
    """
    Trains CBOW and Skip-gram architectures across dimensions 50, 100, 200.
    """
    print("\n=== TRAINING WORD2VEC MODELS ON TRAINING SET ONLY ===")
    dimensions = [50, 100, 200]
    trained_models = {}
    
    for dim in dimensions:
        # 1. CBOW (sg=0)
        print(f"\n--- Training CBOW (dim={dim}, window=5, min_count=2) ---")
        cbow = Word2Vec(
            sentences=train_tokens,
            vector_size=dim,
            window=5,
            min_count=2,
            sg=0,           # CBOW
            workers=4,
            epochs=15,
            seed=42
        )
        trained_models[f"cbow_{dim}"] = cbow
        print(f"CBOW-{dim} trained. Vocabulary size: {len(cbow.wv)}")
        
        # 2. Skip-gram (sg=1)
        print(f"--- Training Skip-gram (dim={dim}, window=5, min_count=2) ---")
        skipgram = Word2Vec(
            sentences=train_tokens,
            vector_size=dim,
            window=5,
            min_count=2,
            sg=1,           # Skip-gram
            workers=4,
            epochs=15,
            seed=42
        )
        trained_models[f"skipgram_{dim}"] = skipgram
        print(f"Skip-gram-{dim} trained. Vocabulary size: {len(skipgram.wv)}")
        
    return trained_models

def evaluate_word_similarities(models):
    """
    Compares semantic similarity across CBOW and Skip-gram for category keywords.
    """
    print("\n=== WORD SIMILARITY COMPARISONS ===")
    test_words = {
        "Politics": "government",
        "Business": "market",
        "Technology": "software",
        "Sports": "player",
        "Entertainment": "film"
    }
    
    similarity_records = {}
    
    for cat, word in test_words.items():
        print(f"\nTarget Word: [{word.upper()}] (Category: {cat})")
        similarity_records[word] = {}
        
        for model_name in ["cbow_100", "skipgram_100"]:
            model = models[model_name]
            if word in model.wv:
                sims = model.wv.most_similar(word, topn=4)
                sim_str = ", ".join([f"{w} ({s:.2f})" for w, s in sims])
                similarity_records[word][model_name] = [w for w, s in sims]
                print(f"  {model_name:14s}: {sim_str}")
            else:
                print(f"  {model_name:14s}: Word not in vocabulary")
                
    return similarity_records

def visualize_embeddings_pca(model_wv, output_path="evaluation/word2vec_pca_visualization.png"):
    """
    Projects selected topical word embeddings from 100-D to 2-D using PCA
    to visualize semantic clustering across the 5 categories.
    """
    print("\n=== VISUALIZING WORD EMBEDDINGS USING PCA ===")
    
    category_words = {
        "Politics": ["government", "minister", "parliament", "election", "senate", "voter", "law", "president"],
        "Business": ["company", "market", "bank", "profit", "stock", "economy", "growth", "revenue"],
        "Technology": ["software", "computer", "internet", "device", "mobile", "digital", "network", "online"],
        "Sports": ["player", "team", "coach", "match", "cup", "game", "league", "champion"],
        "Entertainment": ["film", "movie", "actor", "music", "song", "album", "award", "star"]
    }
    
    all_words = []
    labels = []
    vectors = []
    
    for cat, words in category_words.items():
        for w in words:
            if w in model_wv:
                all_words.append(w)
                labels.append(cat)
                vectors.append(model_wv[w])
                
    vectors = np.array(vectors)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)
    
    df_plot = pd.DataFrame({
        "word": all_words,
        "category": labels,
        "x": coords[:, 0],
        "y": coords[:, 1]
    })
    
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("Set1", len(category_words))
    
    sns.scatterplot(
        data=df_plot, x="x", y="y", hue="category",
        palette=palette, s=160, style="category", alpha=0.9
    )
    
    for _, row in df_plot.iterrows():
        plt.annotate(
            row["word"],
            (row["x"], row["y"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=10,
            fontweight='medium'
        )
        
    var_exp = pca.explained_variance_ratio_
    plt.title(f"Word2Vec 2D PCA Semantic Projection (Variance Explained: {var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}%)",
              fontsize=13, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}%)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved PCA embedding visualization to: {output_path}")

def prepare_and_save_sequence_artifacts(train_tokens, best_model, embedding_dim=100, max_seq_len=150):
    """
    Constructs vocabulary index mapping, sequence padding specifications,
    and embedding matrix for downstream RNN and LSTM training in Phase 5.
    """
    print("\n=== PREPARING SEQUENCE ARTIFACTS FOR RNN / LSTM ===")
    
    # 1. Build word2idx strictly from training tokens
    word2idx = build_vocabulary(train_tokens, min_freq=2)
    vocab_size = len(word2idx)
    print(f"Vocabulary Size (min_freq=2): {vocab_size} tokens (including <PAD>=0, <OOV>=1)")
    
    # 2. Build embedding matrix (vocab_size, embedding_dim)
    embedding_matrix = build_embedding_matrix(word2idx, best_model, embedding_dim=embedding_dim)
    
    # 3. Analyze sequence length distribution
    lengths = [len(seq) for seq in train_tokens]
    p50 = np.percentile(lengths, 50)
    p90 = np.percentile(lengths, 90)
    p95 = np.percentile(lengths, 95)
    print(f"\nSequence Length Stats (Tokens): Median={p50:.0f}, 90th percentile={p90:.0f}, 95th percentile={p95:.0f}")
    print(f"Chosen Maximum Sequence Length (max_seq_len): {max_seq_len}")
    
    # 4. Save artifacts
    vocab_path = "models/tokenizer_word2idx.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(word2idx, f)
    print(f"Saved vocabulary to: {vocab_path}")
    
    emb_matrix_path = "models/embedding_matrix.npy"
    np.save(emb_matrix_path, embedding_matrix)
    print(f"Saved embedding matrix to: {emb_matrix_path}")
    
    # Save best models
    best_model.save("models/word2vec_skipgram_100.model")
    print("Saved Word2Vec Skip-gram model to: models/word2vec_skipgram_100.model")
    
    # Save sequence metadata
    meta = {
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "max_seq_len": max_seq_len,
        "pad_token": "<PAD>",
        "pad_index": 0,
        "oov_token": "<OOV>",
        "oov_index": 1,
        "padding": "post",
        "truncating": "post",
        "word2vec_architecture": "Skip-gram (sg=1, window=5, epochs=15)"
    }
    with open("models/sequence_config.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved sequence metadata to: models/sequence_config.json")
    
    return word2idx, embedding_matrix

if __name__ == "__main__":
    train_tokens, X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    models = train_word2vec_experiments(train_tokens)
    evaluate_word_similarities(models)
    
    # Use Skip-gram 100-D (often richer representations for smaller corpus sizes)
    best_model = models["skipgram_100"]
    models["cbow_100"].save("models/word2vec_cbow_100.model")
    print("Saved CBOW 100-D model to: models/word2vec_cbow_100.model")
    
    visualize_embeddings_pca(best_model.wv)
    prepare_and_save_sequence_artifacts(train_tokens, best_model, embedding_dim=100, max_seq_len=150)
    print("\nPhase 4 Word2Vec Pipeline Completed Successfully!")
