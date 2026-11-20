"""
Text Preprocessing Module for Intelligent News Categorization System
Phase 2: Modular, reusable NLP text preprocessing pipeline.

Supports:
- Text normalization & contraction expansion
- URL, email, and HTML tag removal
- Special character & number handling
- NLTK Tokenization
- Context-preserving Stopword removal (preserves negations)
- Porter Stemming
- WordNet Lemmatization with POS tag mapping
- End-to-end inference-ready Preprocessor class
"""

import os
import re
import string
import nltk

# Ensure workspace nltk_data is prioritized
workspace_nltk = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "nltk_data"))
if os.path.exists(workspace_nltk) and workspace_nltk not in nltk.data.path:
    nltk.data.path.insert(0, workspace_nltk)

from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Ensure required NLTK corpora are available
def ensure_nltk_corpora():
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    ]
    for rel_path, pkg_name in resources:
        try:
            nltk.data.find(rel_path)
        except LookupError:
            try:
                nltk.download(pkg_name, quiet=True)
            except Exception:
                pass

ensure_nltk_corpora()

# Common English Contractions Mapping
CONTRACTIONS_DICT = {
    "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
    "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
    "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'll": "he will",
    "he's": "he is", "how'd": "how did", "how'll": "how will", "how's": "how is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will", "it's": "it is",
    "let's": "let us", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that's": "that is",
    "there's": "there is", "they'd": "they would", "they'll": "they will",
    "they're": "they are", "they've": "they have", "wasn't": "was not",
    "we'd": "we would", "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is",
    "who'll": "who will", "who's": "who is", "won't": "will not",
    "would've": "would have", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}

# Negations and critical semantic modifiers to PRESERVE from stopwords
PRESERVED_WORDS = {
    'no', 'not', 'nor', 'neither', 'never', 'none', 'nobody', 'nowhere',
    'nothing', 'against', 'without', 'cannot', 'under', 'over', 'more',
    'most', 'few', 'up', 'down', 'above', 'below'
}

def get_custom_stopwords():
    """Builds a curated stopword set preserving critical semantic words."""
    try:
        base_stopwords = set(stopwords.words('english'))
    except Exception:
        base_stopwords = set()
    return base_stopwords - PRESERVED_WORDS

def expand_contractions(text: str) -> str:
    """Expands standard English contractions."""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in CONTRACTIONS_DICT.keys()) + r')\b', re.IGNORECASE)
    def replace(match):
        contracted = match.group(0).lower()
        expanded = CONTRACTIONS_DICT.get(contracted, match.group(0))
        if match.group(0)[0].isupper():
            return expanded.capitalize()
        return expanded
    return pattern.sub(replace, text)

def clean_basic(text: str, remove_digits: bool = False) -> str:
    """
    Performs core text normalization:
    - Normalizes case to lowercase
    - Expands contractions
    - Strips URLs, HTML markup, and email addresses
    - Normalizes whitespace and removes unneeded special characters
    """
    if not isinstance(text, str):
        return ""
    
    # Expand contractions first
    text = expand_contractions(text)
    
    # Lowercase
    text = text.lower()
    
    # Strip HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Strip URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    
    # Strip Email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Handle numbers: optionally remove or strip standalone numbers
    if remove_digits:
        text = re.sub(r'\b\d+\b', ' ', text)
    else:
        # Keep digits attached to text or as standard tokens, but remove noisy symbols
        pass
        
    # Replace non-alphanumeric punctuation with spaces (retaining hyphens inside compound words)
    text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
    text = re.sub(r'(?<!\w)-|-(?!\w)', ' ', text)  # remove dangling hyphens
    
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text: str):
    """Tokenizes normalized text into words."""
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    # Filter out empty or single-character punctuation
    return [t for t in tokens if t not in string.punctuation and len(t) > 1]

def filter_stopwords(tokens, custom_stopset=None):
    """Filters stopwords while preserving negations and semantic modifiers."""
    if custom_stopset is None:
        custom_stopset = get_custom_stopwords()
    return [t for t in tokens if t not in custom_stopset]

def stem_tokens(tokens):
    """Applies Porter Stemmer to tokens."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def _get_wordnet_pos(treebank_tag: str):
    """Maps Penn Treebank POS tags to WordNet POS tags for accurate lemmatization."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def lemmatize_tokens(tokens, use_pos: bool = True):
    """
    Applies WordNet Lemmatization to tokens.
    Uses POS tagging when use_pos=True for superior morphological reduction.
    """
    lemmatizer = WordNetLemmatizer()
    if not tokens:
        return []
    
    if use_pos:
        try:
            tagged = pos_tag(tokens)
            return [lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in tagged]
        except Exception:
            return [lemmatizer.lemmatize(word) for word in tokens]
    else:
        return [lemmatizer.lemmatize(word) for word in tokens]

# Four explicit preprocessing comparison pipelines
def preprocess_clean_only(text: str) -> str:
    """Strategy 1: Basic cleaning only."""
    return clean_basic(text)

def preprocess_stopwords_removed(text: str) -> str:
    """Strategy 2: Basic cleaning + Stopword removal (preserving negations)."""
    cleaned = clean_basic(text)
    tokens = tokenize_text(cleaned)
    filtered = filter_stopwords(tokens)
    return " ".join(filtered)

def preprocess_stemmed(text: str) -> str:
    """Strategy 3: Basic cleaning + Stopword removal + Porter Stemming."""
    cleaned = clean_basic(text)
    tokens = tokenize_text(cleaned)
    filtered = filter_stopwords(tokens)
    stemmed = stem_tokens(filtered)
    return " ".join(stemmed)

def preprocess_lemmatized(text: str, use_pos: bool = True) -> str:
    """Strategy 4: Basic cleaning + Stopword removal + WordNet Lemmatization."""
    cleaned = clean_basic(text)
    tokens = tokenize_text(cleaned)
    filtered = filter_stopwords(tokens)
    lemmatized = lemmatize_tokens(filtered, use_pos=use_pos)
    return " ".join(lemmatized)

class NewsPreprocessor:
    """
    Production-ready, reusable text preprocessor.
    Designed for seamless integration into training pipelines (Phase 3)
    and runtime inference in Streamlit (Phase 5).
    """
    def __init__(self, method: str = 'lemmatize', remove_stopwords: bool = True, use_pos: bool = True):
        assert method in ('none', 'clean', 'stopwords', 'stem', 'lemmatize'), \
            f"Unsupported method: {method}"
        self.method = method
        self.remove_stopwords = remove_stopwords
        self.use_pos = use_pos
        self.custom_stopwords = get_custom_stopwords() if remove_stopwords else set()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def transform(self, text: str) -> str:
        """Transforms a single text string."""
        if not text or not isinstance(text, str):
            return ""
            
        cleaned = clean_basic(text)
        if self.method == 'clean':
            return cleaned
            
        tokens = tokenize_text(cleaned)
        if self.remove_stopwords:
            tokens = filter_stopwords(tokens, self.custom_stopwords)
            
        if self.method == 'stem':
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif self.method == 'lemmatize':
            if self.use_pos:
                try:
                    tagged = pos_tag(tokens)
                    tokens = [self.lemmatizer.lemmatize(w, _get_wordnet_pos(tag)) for w, tag in tagged]
                except Exception:
                    tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
            else:
                tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
                
        return " ".join(tokens)
        
    def transform_series(self, text_series):
        """Transforms a pandas Series of texts."""
        return text_series.apply(self.transform)
