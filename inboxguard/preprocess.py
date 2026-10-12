"""Text normalisation and hand-made message statistics."""
import re

import numpy as np
import pandas as pd

URL = re.compile(r"(https?://\S+|www\.\S+|\b\S+\.(com|net|org|co\.uk|info)\b\S*)")
MAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
MONEY = re.compile(r"[£$€]\s?\d[\d,.]*|\b\d[\d,.]*\s?(gbp|usd|eur|rs|inr|pounds?|dollars?)\b")
PHONE = re.compile(r"\b\+?\d[\d\s-]{6,}\d\b")
NUMBER = re.compile(r"\b\d+\b")
NON_ALNUM = re.compile(r"[^a-z\s]")
SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, replace special things by tokens, keep only letters."""
    t = str(text).lower()
    t = URL.sub(" xurlx ", t)
    t = MAIL.sub(" xmailx ", t)
    t = MONEY.sub(" xmoneyx ", t)
    t = PHONE.sub(" xphonex ", t)
    t = NUMBER.sub(" xnumx ", t)
    t = NON_ALNUM.sub(" ", t)
    return SPACES.sub(" ", t).strip()


STAT_NAMES = [
    "n_chars", "n_words", "avg_word_len", "upper_ratio", "digit_ratio",
    "exclamations", "questions", "has_url", "has_money", "has_phone", "special_ratio",
]


def message_stats(texts) -> np.ndarray:
    """Numeric 'style' signals computed on the RAW message."""
    rows = []
    for raw in pd.Series(texts).astype(str):
        n = max(len(raw), 1)
        words = raw.split()
        letters = sum(c.isalpha() for c in raw)
        low = raw.lower()
        rows.append([
            len(raw),
            len(words),
            np.mean([len(w) for w in words]) if words else 0.0,
            sum(c.isupper() for c in raw) / max(letters, 1),
            sum(c.isdigit() for c in raw) / n,
            raw.count("!"),
            raw.count("?"),
            float(bool(URL.search(low))),
            float(bool(MONEY.search(low))),
            float(bool(PHONE.search(low))),
            sum(not (c.isalnum() or c.isspace()) for c in raw) / n,
        ])
    return np.array(rows, dtype=float)


def stat_feature_names(transformer, input_features):
    """Lets scikit-learn name the style columns (used for explanations)."""
    return np.array(STAT_NAMES)
