
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from config import DATA_DIR, SEED


def load_corpus():
    """Corpus (Step 4) with the train/dev/test split of every wallet (Step 5)."""
    corpus = pd.read_csv(DATA_DIR / "corpus.tsv", sep="\t")
    splits = pd.read_csv(DATA_DIR / "splits.tsv", sep="\t")
    return corpus.merge(splits, on="wallet")


def mann_whitney(y, score):
    """One-sided test that liquidated wallets (y=1) score HIGHER than safe ones.
    Effect size: rank-biserial correlation r = 2 * AUC - 1 (0 = none, 1 = total)."""
    y, score = np.asarray(y), np.asarray(score)
    u, p = mannwhitneyu(score[y == 1], score[y == 0], alternative="greater")
    auc = u / ((y == 1).sum() * (y == 0).sum())
    return {"U": float(u), "p_value": float(p), "rank_biserial": float(2 * auc - 1)}


def bootstrap_auc(y, scores, pairs=(), n_boot=1000, seed=SEED):
    """ROC-AUC of each score with a 95% percentile interval. Wallets are
    resampled with replacement inside each class. For every (a, b) in `pairs`,
    also the PAIRED difference AUC(a) - AUC(b), from the same resamples.
    Returns (aucs, differences)."""
    y = np.asarray(y)
    s = {k: np.asarray(v, dtype=float) for k, v in scores.items()}
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boot = {k: np.empty(n_boot) for k in s}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, v in s.items():
            boot[k][b] = roc_auc_score(y[idx], v[idx])

    aucs = {}
    for k, v in s.items():
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        aucs[k] = {"auc": float(roc_auc_score(y, v)), "ci_low": float(lo), "ci_high": float(hi)}
    diffs = {}
    for a, b in pairs:
        lo, hi = np.percentile(boot[a] - boot[b], [2.5, 97.5])
        diffs[f"{a} - {b}"] = {"diff": aucs[a]["auc"] - aucs[b]["auc"],
                               "ci_low": float(lo), "ci_high": float(hi)}
    return aucs, diffs
