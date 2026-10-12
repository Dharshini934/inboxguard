import pandas as pd

from .config import DATA_PATH
from .preprocess import normalize


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Reads a 2-column CSV (label, text). Works with the Kaggle SMS file
    (v1, v2 + empty columns) and with any other file whose first two columns
    are label and message."""
    df = pd.read_csv(path, encoding="latin-1").iloc[:, :2]
    df.columns = ["label", "raw"]
    df = df.dropna().copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["spam", "ham"])]
    before = len(df)
    df = df.drop_duplicates(subset="raw").reset_index(drop=True)
    df.attrs["duplicates_removed"] = before - len(df)
    df["y"] = (df["label"] == "spam").astype(int)
    df["norm"] = df["raw"].map(normalize)
    return df
