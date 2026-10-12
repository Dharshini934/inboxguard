"""One feature matrix = word TF-IDF + character TF-IDF + style statistics."""
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from .preprocess import message_stats, stat_feature_names


def build_features() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "norm"),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     max_features=15000, sublinear_tf=True), "norm"),
            ("style", Pipeline([("calc", FunctionTransformer(message_stats, feature_names_out=stat_feature_names)),
                                ("scale", MinMaxScaler(clip=True))]), "raw"),
        ],
        transformer_weights={"word": 1.0, "char": 0.6, "style": 0.5},
    )
