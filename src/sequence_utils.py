"""
Sequence Utilities for Deep Learning Preparation
Phase 4: Intelligent News Categorization System

Provides:
- Token-to-index mapping (word2idx) with reserved <PAD> (0) and <OOV> (1) tokens
- Sequence conversion and truncation / padding
- Embedding matrix construction from Gensim Word2Vec for Keras Embedding layers
- Reusable pipeline functions for training and inference
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional

PAD_TOKEN = "<PAD>"  # Index 0
OOV_TOKEN = "<OOV>"  # Index 1

def build_vocabulary(tokenized_texts: List[List[str]], min_freq: int = 2) -> Dict[str, int]:
    """
    Builds a vocabulary dictionary from tokenized training documents.
    Reserved:
      <PAD>: 0
      <OOV>: 1
    """
    freq_dict = {}
    for doc in tokenized_texts:
        for token in doc:
            freq_dict[token] = freq_dict.get(token, 0) + 1
            
    word2idx = {PAD_TOKEN: 0, OOV_TOKEN: 1}
    idx = 2
    for token, count in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True):
        if count >= min_freq:
            word2idx[token] = idx
            idx += 1
            
    return word2idx

def texts_to_sequences(tokenized_texts: List[List[str]], word2idx: Dict[str, int]) -> List[List[int]]:
    """Converts token lists into lists of integer indices with OOV handling."""
    oov_idx = word2idx.get(OOV_TOKEN, 1)
    sequences = []
    for doc in tokenized_texts:
        seq = [word2idx.get(token, oov_idx) for token in doc]
        sequences.append(seq)
    return sequences

def pad_sequences_manual(
    sequences: List[List[int]],
    max_len: int,
    padding: str = 'post',
    truncating: str = 'post'
) -> np.ndarray:
    """
    Pads or truncates sequences to max_len.
    padding: 'pre' or 'post'
    truncating: 'pre' or 'post'
    Returns integer 2D numpy array of shape (len(sequences), max_len).
    """
    padded = np.zeros((len(sequences), max_len), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        # Truncate
        if len(seq) > max_len:
            if truncating == 'pre':
                seq_cut = seq[-max_len:]
            else:
                seq_cut = seq[:max_len]
        else:
            seq_cut = seq
            
        # Pad
        if padding == 'pre':
            padded[i, -len(seq_cut):] = seq_cut
        else:
            padded[i, :len(seq_cut)] = seq_cut
            
    return padded

def build_embedding_matrix(
    word2idx: Dict[str, int],
    w2v_model,
    embedding_dim: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Constructs the embedding matrix (vocab_size, embedding_dim) for Keras.
    - Index 0 (<PAD>) is strictly all zeros.
    - Index 1 (<OOV>) is initialized randomly with unit variance.
    - Remaining indices map to Word2Vec vectors if present, or random vectors.
    """
    np.random.seed(random_state)
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    
    # OOV token vector (unit norm)
    oov_vec = np.random.uniform(-0.05, 0.05, embedding_dim).astype(np.float32)
    embedding_matrix[1] = oov_vec
    
    in_vocab_count = 0
    oov_count = 0
    
    for word, idx in word2idx.items():
        if idx in (0, 1):
            continue
        if word in w2v_model.wv:
            embedding_matrix[idx] = w2v_model.wv[word]
            in_vocab_count += 1
        else:
            # Random initialization for tokens in word2idx but not in Word2Vec (e.g. min_count differences)
            embedding_matrix[idx] = np.random.uniform(-0.05, 0.05, embedding_dim).astype(np.float32)
            oov_count += 1
            
    print(f"Embedding matrix constructed: shape {embedding_matrix.shape}")
    print(f"Words initialized from Word2Vec: {in_vocab_count} ({(in_vocab_count/(vocab_size-2))*100:.2f}%)")
    print(f"Out-of-vocabulary words: {oov_count}")
    return embedding_matrix
