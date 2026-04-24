"""
preprocessing.py — Text cleaning & feature engineering utilities
"""

import re
import string
import numpy as np
import pandas as pd


# Common clickbait / fake news signal words
SENSATIONAL_WORDS = {
    "shocking", "exposed", "leaked", "secret", "banned", "coverup",
    "urgent", "breaking", "miracle", "must share", "they dont want",
    "wake up", "mainstream media", "proven", "bombshell", "hoax",
    "conspiracy", "illuminati", "chemtrails", "deep state", "fake",
}

CREDIBILITY_WORDS = {
    "study", "research", "university", "published", "peer-reviewed",
    "according", "official", "confirmed", "data", "evidence",
    "scientists", "experts", "report", "analysis", "statistics",
}


def preprocess_text(text: str) -> str:
    """Clean and normalize raw news text for TF-IDF."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(text: str) -> dict:
    """Extract hand-crafted linguistic features for EDA/analysis."""
    words = text.lower().split()
    word_set = set(words)

    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamation_count = text.count("!")
    question_count = text.count("?")
    sensational_hits = len(word_set & SENSATIONAL_WORDS)
    credibility_hits = len(word_set & CREDIBILITY_WORDS)

    return {
        "word_count": len(words),
        "char_count": len(text),
        "caps_ratio": round(caps_ratio, 4),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "sensational_word_count": sensational_hits,
        "credibility_word_count": credibility_hits,
        "avg_word_length": round(np.mean([len(w) for w in words]) if words else 0, 2),
    }


def load_csv_dataset(filepath: str, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Load a real-world CSV dataset (e.g., Kaggle Fake News dataset).

    Expected columns:
      text_col  — the article text or headline
      label_col — 0 for fake, 1 for real (or string 'FAKE'/'REAL')
    """
    df = pd.read_csv(filepath)
    df = df[[text_col, label_col]].dropna().rename(columns={text_col: "text", label_col: "label"})

    if df["label"].dtype == object:
        df["label"] = df["label"].str.upper().map({"REAL": 1, "FAKE": 0, "TRUE": 1, "FALSE": 0})

    df["label"] = df["label"].astype(int)
    df["clean_text"] = df["text"].apply(preprocess_text)
    return df
