from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_features


def candidates():
    clfs = {
        "Naive Bayes": MultinomialNB(alpha=0.05),
        "Logistic Regression": LogisticRegression(C=20, max_iter=3000, class_weight="balanced"),
        "SVM": CalibratedClassifierCV(LinearSVC(C=0.8, class_weight="balanced"), cv=5),
    }
    return {name: Pipeline([("features", build_features()), ("clf", c)]) for name, c in clfs.items()}
