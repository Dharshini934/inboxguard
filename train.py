"""Train, tune the decision threshold, evaluate and save everything.

    python train.py
"""
import json
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from inboxguard.config import (EXPLAINER_PATH, MIN_PRECISION, MODEL_PATH,
                               REPORTS, SEED, TEST_SIZE)
from inboxguard.data import load_dataset
from inboxguard.explain import _pretty
from inboxguard.models import candidates
from inboxguard.preprocess import STAT_NAMES, message_stats

warnings.filterwarnings("ignore")
HAM, SPAM, INK = "#1F7A8C", "#C8553D", "#22313F"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "font.size": 10})


def pick_threshold(y, probs):
    """Lowest threshold whose precision on cross-validated predictions >= MIN_PRECISION
    (that keeps the most spam caught). Falls back to the best-F1 threshold."""
    prec, rec, thr = precision_recall_curve(y, probs)
    ok = [i for i in range(len(thr)) if prec[i] >= MIN_PRECISION]
    if ok:
        return float(thr[ok[0]])
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[int(np.argmax(f1))])


def scores(y, probs, thr):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "false_positives": int(fp), "false_negatives": int(fn),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    REPORTS.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    df = load_dataset()
    print(f"{len(df)} unique messages ({df.attrs['duplicates_removed']} duplicates removed), "
          f"spam share {df.y.mean():.1%}")

    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df.y, random_state=SEED)
    Xtr, ytr = train[["raw", "norm"]], train.y.values
    Xte, yte = test[["raw", "norm"]], test.y.values
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    results, fitted, test_probs = [], {}, {}
    for name, pipe in candidates().items():
        print(f"\n>> {name}")
        cv_probs = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        thr = pick_threshold(ytr, cv_probs)
        cv_f1 = scores(ytr, cv_probs, thr)["f1"]
        cv_ap = average_precision_score(ytr, cv_probs)   # threshold-free ranking quality
        pipe.fit(Xtr, ytr)
        probs = pipe.predict_proba(Xte)[:, 1]
        tuned, default = scores(yte, probs, thr), scores(yte, probs, 0.5)
        row = {"model": name, "threshold": round(thr, 4), "cv_f1": cv_f1, "cv_ap": cv_ap,
               "roc_auc": roc_auc_score(yte, probs), "f1_at_0.5": default["f1"], **tuned}
        results.append(row)
        fitted[name], test_probs[name] = pipe, probs
        print(f"   threshold={thr:.3f}  P={tuned['precision']:.3f} R={tuned['recall']:.3f} "
              f"F1={tuned['f1']:.3f}  FP={tuned['fp']} FN={tuned['fn']}")

    res = pd.DataFrame(results)
    best = res.loc[res.cv_ap.idxmax()]               # chosen with CV only, not the test set
    print(f"\nBest model (by CV average precision): {best.model}")
    joblib.dump({"model": fitted[best.model], "threshold": float(best.threshold),
                 "name": best.model}, MODEL_PATH)
    joblib.dump(fitted["Logistic Regression"], EXPLAINER_PATH)

    summary = {
        "dataset": {"messages": int(len(df)), "spam": int(df.y.sum()), "ham": int((1 - df.y).sum()),
                    "duplicates_removed": int(df.attrs["duplicates_removed"]),
                    "train": int(len(train)), "test": int(len(test))},
        "min_precision_target": MIN_PRECISION,
        "best_model": best.model,
        "models": json.loads(res.to_json(orient="records")),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    res.round(4).to_csv(REPORTS / "metrics.csv", index=False)

    thr = float(best.threshold)
    wrong = test.assign(spam_probability=test_probs[best.model].round(4),
                        predicted=np.where(test_probs[best.model] >= thr, "spam", "ham"))
    wrong[wrong.label != wrong.predicted][["label", "predicted", "spam_probability", "raw"]] \
        .to_csv(REPORTS / "errors.csv", index=False)

    make_figures(df, test, yte, fitted, test_probs, res)
    print("\nSaved model, metrics and figures.")


# ------------------------------------------------------------------ figures
def make_figures(df, test, yte, fitted, test_probs, res):
    # 1 overview
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    counts = df.label.value_counts()
    ax[0].pie([counts["ham"], counts["spam"]], labels=["ham", "spam"], colors=[HAM, SPAM],
              autopct="%1.1f%%", startangle=90, wedgeprops={"width": 0.4, "edgecolor": "white"})
    ax[0].set_title("Share of messages")
    for lab, col in [("ham", HAM), ("spam", SPAM)]:
        ax[1].hist(df.loc[df.label == lab, "raw"].str.len(), bins=35, alpha=.65, color=col, label=lab)
    ax[1].set_xlabel("characters per message"); ax[1].set_ylabel("messages"); ax[1].legend(frameon=False)
    ax[1].set_title("Message length")
    plt.tight_layout(); plt.savefig(REPORTS / "fig1_overview.png", dpi=150); plt.close()

    # 2 style signals
    stats = pd.DataFrame(message_stats(df.raw), columns=STAT_NAMES)
    stats["label"] = df.label.values
    show = ["upper_ratio", "digit_ratio", "special_ratio", "has_url", "has_money", "has_phone"]
    means = stats.groupby("label")[show].mean()
    x = np.arange(len(show)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w / 2, means.loc["ham"], w, color=HAM, label="ham")
    ax.bar(x + w / 2, means.loc["spam"], w, color=SPAM, label="spam")
    ax.set_xticks(x); ax.set_xticklabels(show, rotation=15); ax.legend(frameon=False)
    ax.set_title("Average style signals: spam vs ham"); ax.set_ylabel("mean value")
    plt.tight_layout(); plt.savefig(REPORTS / "fig2_style_signals.png", dpi=150); plt.close()

    # 3 precision-recall
    from sklearn.metrics import precision_recall_curve as prc
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for (name, p), c in zip(test_probs.items(), [HAM, SPAM, "#8A6FDF"]):
        pr, rc, _ = prc(yte, p)
        ax.plot(rc, pr, color=c, label=name, lw=2)
        r = res[res.model == name].iloc[0]
        ax.scatter([r.recall], [r.precision], color=c, s=60, zorder=3, edgecolor="white")
    ax.axhline(MIN_PRECISION, ls="--", color="grey", lw=1)
    ax.set_xlabel("recall (spam caught)"); ax.set_ylabel("precision (flags that were really spam)")
    ax.set_ylim(0.6, 1.01); ax.set_title("Precision-recall (dots = chosen threshold)"); ax.legend(frameon=False)
    plt.tight_layout(); plt.savefig(REPORTS / "fig3_precision_recall.png", dpi=150); plt.close()

    # 4 confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for a, (_, r) in zip(axes, res.iterrows()):
        m = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        a.imshow(m, cmap="GnBu")
        for i in range(2):
            for j in range(2):
                a.text(j, i, int(m[i, j]), ha="center", va="center",
                       color="white" if m[i, j] > m.max() / 2 else INK, fontsize=13)
        a.set_xticks([0, 1]); a.set_xticklabels(["ham", "spam"]); a.set_yticks([0, 1]); a.set_yticklabels(["ham", "spam"])
        a.set_xlabel("predicted"); a.set_ylabel("actual"); a.set_title(r.model)
    plt.tight_layout(); plt.savefig(REPORTS / "fig4_confusion.png", dpi=150); plt.close()

    # 5 metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    cols = ["precision", "recall", "f1"]
    x = np.arange(len(res)); w = 0.26
    for k, (c, colr) in enumerate(zip(cols, [HAM, SPAM, "#8A6FDF"])):
        ax.bar(x + (k - 1) * w, res[c], w, label=c, color=colr)
    ax.set_xticks(x); ax.set_xticklabels(res.model); ax.set_ylim(0.8, 1.0); ax.legend(frameon=False, ncol=3)
    ax.set_title("Test-set scores at the tuned threshold")
    plt.tight_layout(); plt.savefig(REPORTS / "fig5_metrics.png", dpi=150); plt.close()

    # 6 top terms (Logistic Regression weights on word features)
    lr = fitted["Logistic Regression"]
    names = lr.named_steps["features"].get_feature_names_out()
    coef = lr.named_steps["clf"].coef_[0]
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
    idx = [i for i, n in enumerate(names) if n.startswith("word__") and not all(w in SW for w in n[6:].split())]
    order = sorted(idx, key=lambda i: coef[i])
    ham_i, spam_i = order[:15], order[-15:]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].barh([_pretty(names[i][6:]) for i in spam_i], coef[spam_i], color=SPAM); ax[0].set_title("Words that signal spam")
    ax[1].barh([_pretty(names[i][6:]) for i in ham_i], -coef[ham_i], color=HAM); ax[1].set_title("Words that signal a normal message")
    plt.tight_layout(); plt.savefig(REPORTS / "fig6_top_terms.png", dpi=150); plt.close()
    pd.DataFrame({"spam_terms": [_pretty(names[i][6:]) for i in spam_i][::-1],
                  "ham_terms": [_pretty(names[i][6:]) for i in ham_i]}).to_csv(REPORTS / "top_terms.csv", index=False)


if __name__ == "__main__":
    main()
