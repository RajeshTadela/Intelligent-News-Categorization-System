# Word Embeddings & Word2Vec Technical Report
**Project:** Intelligent News Categorization System  
**Phase:** 4 — Word Embeddings, Word2Vec Training & Sequence Preparation  
**Generated Date:** November 2026  

---

## 1. Executive Summary

Phase 4 bridges classical machine learning and deep learning by developing continuous dense word representations. Using **Gensim Word2Vec**, we trained both **CBOW (Continuous Bag of Words)** and **Skip-gram** architectures strictly on the **8,450 articles** of the training partition. 

Key results and deliverables:
- **Embedding Matrix:** Generated a pre-trained embedding matrix of shape `(17411, 100)` covering 100% of the training vocabulary (17,409 words + `<PAD>` + `<OOV>`).
- **Semantic Clustering:** 2D PCA projections demonstrated clear topical grouping across the 5 categories (Politics, Business, Technology, Sports, Entertainment).
- **Sequence Preparation:** Built a unified token-to-index mapping (`word2idx`), post-padding sequence utility, and maximum sequence length constraint ($L = 150$), ready for direct ingestion by **Simple RNN** and **LSTM** in Phase 5.

---

## 2. Core Concepts Explained

### 2.1 One-Hot Encoding
In one-hot encoding, each word $w_i$ is represented by a binary vector of dimension $|V|$:
$$\mathbf{e}_{\text{politics}} = [1, 0, 0, 0, \dots, 0]^T, \quad \mathbf{e}_{\text{minister}} = [0, 1, 0, 0, \dots, 0]^T$$
- **Limitation:** For all $i \neq j$, $\mathbf{e}_i \cdot \mathbf{e}_j = 0$ and $\|\mathbf{e}_i - \mathbf{e}_j\|_2 = \sqrt{2}$. All words are equidistant and mutually orthogonal; semantic similarity is completely unrepresented.

### 2.2 Sparse vs. Dense Representations
- **Sparse (One-Hot, TF-IDF):** Very high dimensional ($|V| \ge 20,000$). Over 99% of vector elements are zero. Lacks relational semantic geometry.
- **Dense (Word2Vec Embeddings):** Low dimensional ($d \in [50, 300]$). Every element is a real-valued number representing a latent semantic feature.

### 2.3 What Are Word Embeddings?
Word embeddings map discrete symbolic tokens $w \in V$ to continuous geometric vectors $\mathbf{v}_w \in \mathbb{R}^d$. This mapping operationalizes Firth's Distributional Hypothesis: words appearing in similar linguistic contexts share similar vector coordinates.

### 2.4 Semantic Similarity (Cosine Similarity)
The semantic relatedness of words is quantified by the cosine of the angle between their embedding vectors:
$$\text{sim}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

### 2.5 Word2Vec Architectures: CBOW vs. Skip-gram
- **CBOW (Continuous Bag of Words, `sg=0`):**  
  Maximizes the log probability of predicting a center word $w_t$ given context words within window $k$:
  $$\mathcal{L}_{\text{CBOW}} = \sum_{t=1}^T \log P(w_t \mid w_{t-k}, \dots, w_{t+k})$$
  *Characteristics:* Trains rapidly, averages context representations, highly effective for frequent words.
- **Skip-gram (`sg=1`):**  
  Maximizes the log probability of predicting surrounding context words given the center word $w_t$:
  $$\mathcal{L}_{\text{SG}} = \sum_{t=1}^T \sum_{-k \le j \le k, j \neq 0} \log P(w_{t+j} \mid w_t)$$
  *Characteristics:* Employs negative sampling to contrast true contexts against noise. Excels at learning distinct embeddings for rare and domain-specific terminology.

### 2.6 Embedding Dimensions ($d$)
- **50-D:** Fast, lightweight, suitable for small corpora (<5,000 articles).
- **100-D:** **Optimal balance** between representational capacity, generalization, and computational speed for ~10,000 news articles.
- **200-D:** Higher capacity, but risks overfitting on small-to-medium corpora.

### 2.7 Out-Of-Vocabulary (OOV) Handling
Any token not observed during training (or appearing with frequency $< \text{min\_count}$) cannot be mapped directly. We allocate:
- Index `0`: Reserved exclusively for `<PAD>`.
- Index `1`: Reserved for `<OOV>`, initialized with a small random uniform vector ($\mathcal{U}(-0.05, 0.05)$).

### 2.8 Padding and Sequence Length
Neural networks operate on batched matrices requiring rectangular dimensions `(batch_size, max_seq_len)`.
- Sequence length analysis revealed a median of 18 words and a 90th percentile of 184 words.
- Setting `max_seq_len = 150` covers full headline summaries and the primary diagnostic sections of full news articles while avoiding excessive zero-padding overhead.

---

## 3. Word2Vec Experimental Results

### 3.1 Model Comparison & Vocab Size
Models were trained for 15 epochs with a window size of 5 and `min_count=2` on the 8,450 training documents. The resulting vocabulary contains **17,409 words**.

### 3.2 Semantic Similarity Comparisons
Top similar words retrieved by CBOW (100-D) and Skip-gram (100-D):

| Target Word | Category | CBOW (100-D) Top Neighbors | Skip-gram (100-D) Top Neighbors |
| :--- | :--- | :--- | :--- |
| **`government`** | Politics | `eu` (0.77), `immigration` (0.75), `referendum` (0.74), `reform` (0.73) | `quango` (0.62), `re-elected` (0.58), `colonial` (0.57), `refuge` (0.57) |
| **`market`** | Business | `manufacturer` (0.74), `slowdown` (0.71), `europe` (0.70), `housing` (0.69) | `risen` (0.60), `afield` (0.58), `fujitsu` (0.58), `stock` (0.57) |
| **`software`** | Technology | `program` (0.93), `malicious` (0.89), `hardware` (0.89), `processor` (0.89) | `programmer` (0.75), `anti-virus` (0.74), `open-source` (0.73), `program` (0.73) |
| **`player`** | Sports | `team` (0.70), `squad` (0.63), `league` (0.62), `train` (0.61) | `furious` (0.52), `zen` (0.51), `emergence` (0.51), `enters` (0.50) |
| **`film`** | Entertainment| `category` (0.78), `award` (0.77), `christ` (0.75), `movie` (0.75) | `movie` (0.64), `festival` (0.62), `low-budget` (0.59), `christ` (0.58) |

---

## 4. 2D PCA Semantic Projection Analysis

A 2-component Principal Component Analysis (PCA) was fitted on the 100-dimensional embeddings of 40 selected keywords across the 5 classes:
- **Politics Cluster:** Words such as `government`, `minister`, `parliament`, `election`, `senate`, `voter`, `president` clustered closely.
- **Business Cluster:** Words such as `market`, `bank`, `company`, `profit`, `stock`, `economy`, `revenue` formed a distinct financial cluster.
- **Technology Cluster:** `software`, `computer`, `internet`, `device`, `mobile`, `digital`, `network` grouped together.
- **Sports Cluster:** `player`, `team`, `coach`, `match`, `cup`, `game`, `champion` isolated cleanly.
- **Entertainment Cluster:** `film`, `movie`, `actor`, `music`, `song`, `album`, `award` formed a coherent cultural cluster.

Visual output is saved at: [`evaluation/word2vec_pca_visualization.png`](file:///c:/Intelligent%20News%20Categorization%20System/evaluation/word2vec_pca_visualization.png).

---

## 5. Sequence Preparation Artifacts for Deep Learning (Phase 5)

| Artifact | File Path | Dimensions / Shape | Description |
| :--- | :--- | :--- | :--- |
| **Vocabulary Mapping** | `models/tokenizer_word2idx.json` | 17,411 entries | Token-to-integer dictionary (`<PAD>`: 0, `<OOV>`: 1) |
| **Embedding Matrix** | `models/embedding_matrix.npy` | `(17411, 100)` float32 | Pretrained weights for Keras `Embedding` layer |
| **Skip-gram Model** | `models/word2vec_skipgram_100.model` | 100-D vectors | Full Gensim Word2Vec model |
| **CBOW Model** | `models/word2vec_cbow_100.model` | 100-D vectors | Full Gensim Word2Vec model |
| **Sequence Metadata** | `models/sequence_config.json` | JSON metadata | Stores `max_seq_len=150`, padding/truncating settings |
