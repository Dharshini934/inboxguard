"""Explain one prediction with the Logistic Regression weights."""
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .preprocess import STAT_NAMES, message_stats, normalize


FRIENDLY = {"xnumx": "[number]", "xphonex": "[phone number]", "xurlx": "[web link]",
            "xmoneyx": "[amount]", "xmailx": "[email address]"}


def _pretty(term: str) -> str:
    return " ".join(FRIENDLY.get(w, w) for w in term.split())


def _frame(text: str) -> pd.DataFrame:
    return pd.DataFrame({"raw": [text], "norm": [normalize(text)]})


def predict_one(model, text: str, threshold: float):
    p = float(model.predict_proba(_frame(text))[0, 1])
    return p, ("spam" if p >= threshold else "ham")


def predict_many(model, texts, threshold: float):
    frame = pd.DataFrame({"raw": list(texts), "norm": [normalize(t) for t in texts]})
    probs = model.predict_proba(frame)[:, 1]
    return probs, ["spam" if p >= threshold else "ham" for p in probs]


def top_signals(explainer, text: str, top: int = 8):
    feats = explainer.named_steps["features"]
    clf = explainer.named_steps["clf"]
    X = feats.transform(_frame(text)).tocsr()
    names = feats.get_feature_names_out()
    coef = clf.coef_[0]
    row = X[0]
    out = []
    for idx, val in zip(row.indices, row.data):
        name = names[idx]
        if name.startswith("word__") and not all(w in ENGLISH_STOP_WORDS for w in name[6:].split()):
            out.append({"term": _pretty(name[6:]), "weight": round(float(val * coef[idx]), 3)})
    out.sort(key=lambda d: abs(d["weight"]), reverse=True)
    return out[:top]


def style_summary(text: str):
    vals = message_stats([text])[0]
    return {n: round(float(v), 3) for n, v in zip(STAT_NAMES, vals)}
