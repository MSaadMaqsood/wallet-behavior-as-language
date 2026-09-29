
import json
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import plot_style as ps
from config import DATA_DIR, RESULTS_DIR, SEED
from evaluation import bootstrap_auc, load_corpus
from ngram_lm import KneserNeyLM
from step4_tokenize import VOCAB

WINDOW = 4
N_SHUFFLES = 5
C_GRID = [0.01, 0.1, 1, 10]

# figure groups, top to bottom: (title, {key: (label, uses order, uses labels)})
GROUPS = [
    ("Safe-behavior language model (n-gram, n = 6)", {
        "ngram": ("full score: main model", True, False),
        "ngram_eos_only": ("end token only", True, False),
        "ngram_actions": ("actions only (no end token)", True, False)}),
    ("Order and vocabulary controls", {
        "ngram_shuffled": ("n-gram LM on shuffled sentences", False, False),
        "unigram": ("unigram LM (bag of actions)", False, False),
        "unigram_actions": ("unigram LM, actions only", False, False),
        "ngram_base_vocab": ("n-gram LM, 4 actions (no asset class)", True, False),
        "length": ("sentence length only", False, False)}),
    (f"Matched windows: {WINDOW} actions ending mid-stream", {
        "win_ngram": ("n-gram LM", True, False),
        "win_ngram_shuffled": ("n-gram LM on shuffled windows", False, False),
        "win_unigram": ("unigram LM (bag of actions)", False, False),
        "win_logreg_ngrams": ("logistic regression, 1-3 action sequences", True, True),
        "win_logreg_counts": ("logistic regression, action counts", False, True)}),
    ("Supervised classifiers, full histories", {
        "logreg_ngrams": ("logistic regression, 1-3 action sequences", True, True),
        "logreg_counts": ("logistic regression, action counts", False, True)}),
]
MODELS = {k: v for _, g in GROUPS for k, v in g.items()}
WINDOW_KEYS = [k for k in MODELS if k.startswith("win_")]
PAIRS_FULL = [("ngram", "unigram"), ("ngram", "ngram_shuffled"),
              ("ngram_actions", "unigram_actions"), ("logreg_ngrams", "logreg_counts"),
              ("ngram", "ngram_actions"), ("ngram", "ngram_base_vocab")]
PAIRS_WINDOW = [("win_ngram", "win_unigram"), ("win_ngram", "win_ngram_shuffled"),
                ("win_logreg_ngrams", "win_logreg_counts")]


# --------------------------------------------------------------------------- #
def log_ppl(lm, sentence, with_eos=True):
    """Log-perplexity (same ranking as perplexity), optionally without <EOS>."""
    lp = lm.token_logprobs(sentence)
    if not with_eos:
        lp = lp[:-1]
    return -sum(lp) / len(lp)


def window_log_ppl(lm, window):
    lp = lm.window_logprobs(window)
    return -sum(lp) / len(lp)


def add_matched_windows(main, rng, w=WINDOW):
    """Equal-length windows that end mid-stream in both classes.
    Liquidated: the last w actions before the first liquidation.
    Safe: w consecutive actions ending at a random point before the history's
    last action, so a natural end is never included. Too-short wallets: None."""
    windows, ends = [], []
    for tokens, label in zip(main["sentence"].str.split(), main["label"]):
        end = None
        if label == 1 and len(tokens) >= w:
            end = len(tokens)
        elif label == 0 and len(tokens) >= w + 1:
            end = int(rng.integers(w, len(tokens)))  # window = tokens[end-w:end], end <= len-1
        windows.append(" ".join(tokens[end - w:end]) if end else None)
        ends.append(end)
    main = main.copy()
    main["window"] = windows
    main["window_end"] = pd.array(ends, dtype="Int64")  # position after the window's last action
    return main


def logistic(train, dev, test, ngram_range, text="sentence", size=True):
    """Supervised baseline: train on train wallets (both classes), choose C on dev,
    score test. Features: counts per action of every 1..k action sequence
    (+ log length when sentences differ in length)."""
    vec = CountVectorizer(ngram_range=ngram_range, token_pattern=r"\S+", lowercase=False)
    vec.fit(train[text])

    def features(df):
        counts = vec.transform(df[text])
        n_tokens = df[text].str.count(" ").to_numpy() + 1
        x = csr_matrix(counts.multiply(1.0 / n_tokens[:, None]))
        if size:
            x = hstack([x, np.log1p(df["n_actions"].to_numpy())[:, None]]).tocsr()
        return x

    scaler = StandardScaler(with_mean=False).fit(features(train))
    x_train, x_dev, x_test = (scaler.transform(features(d)) for d in (train, dev, test))
    best = None
    for c in C_GRID:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
        clf.fit(x_train, train["label"])
        auc = roc_auc_score(dev["label"], clf.decision_function(x_dev))
        if best is None or auc > best[1]:
            best = (clf, auc, c)
    info = {"C": best[2], "dev_auc": round(best[1], 4), "features": x_train.shape[1]}
    return best[0].decision_function(x_test), info


def leakage_screen(main):
    """AUC of simple one-number features on all main-sample wallets."""
    y = main["label"].to_numpy()
    tokens = main["sentence"].str.split()
    feats = {"length (actions)": main["n_actions"]}
    for t in VOCAB:
        count = tokens.map(lambda s, t=t: s.count(t))
        feats[f"count of {t}"] = count
        feats[f"share of {t}"] = count / main["length"]
        feats[f"first action is {t}"] = tokens.str[0] == t
        feats[f"last action is {t}"] = tokens.str[-1] == t
    df = pd.DataFrame([{"feature": k, "auc": roc_auc_score(y, v)} for k, v in feats.items()])
    df["distance_from_chance"] = (df["auc"] - 0.5).abs()
    df["suspicious"] = df["distance_from_chance"] >= 0.4
    return df.sort_values("distance_from_chance", ascending=False, ignore_index=True)


def plot_forest(aucs, path):
    rows, headers, y = [], [], 0.0
    for title, g in reversed(GROUPS):  # draw bottom-up
        for k in reversed(list(g)):
            rows.append((y, k))
            y += 1
        headers.append((y - 0.2, title))
        y += 1.0
    fig, ax = ps.plt.subplots(figsize=(8.5, 7.2))
    to_axis = ax.get_yaxis_transform()  # x in axes units, y in data units
    for yi, k in rows:
        a = aucs[k]
        color = ps.INK if k == "ngram" else ps.MUTED
        ax.plot([a["ci_low"], a["ci_high"]], [yi, yi], color=color)
        ax.plot(a["auc"], yi, "o", color=color, markersize=8,
                markeredgecolor=ps.SURFACE, markeredgewidth=2)
        ax.text(1.02, yi, f"{a['auc']:.3f}  [{a['ci_low']:.3f}–{a['ci_high']:.3f}]",
                transform=to_axis, va="center", color=ps.INK_2, fontsize=9)
    for hy, title in headers:
        ax.text(-0.015, hy, title, transform=to_axis, ha="right", va="center",
                fontsize=9.5, fontweight="bold", color=ps.INK)
        ax.axhline(hy + 0.45, color=ps.GRID, linewidth=1)
    ax.axvline(0.5, color=ps.AXIS, linewidth=1)
    ax.text(0.5, -0.9, " chance", color=ps.MUTED, fontsize=8.5, va="center")
    ax.set_yticks([r[0] for r in rows], [MODELS[r[1]][0] for r in rows])
    ax.set_ylim(-1.2, y - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC-AUC on test wallets (95% bootstrap interval)")
    ax.set_title("What carries the signal? Models and controls side by side")
    ps.save(fig, path)


# --------------------------------------------------------------------------- #
def main():
    ps.apply()
    rng = np.random.default_rng(SEED)
    corpus = load_corpus()
    main_s = add_matched_windows(corpus[corpus["sample_main"]], rng)
    split = {s: main_s[main_s["split"] == s] for s in ("train", "dev", "test")}
    test = split["test"].reset_index(drop=True)
    y = test["label"].to_numpy()
    sents = test["sentence"].str.split().tolist()

    with open(DATA_DIR / "ngram_lm.pkl", "rb") as f:
        models = pickle.load(f)
    lm, lm_base = models["sentence"], models["sentence_base"]
    safe_train = split["train"][split["train"]["label"] == 0]
    unigram = KneserNeyLM(1).fit(safe_train["sentence"].str.split().tolist())

    # ---- full histories: every test wallet ----
    s = {"ngram": [log_ppl(lm, t) for t in sents],
         "ngram_eos_only": [-lm.token_logprobs(t)[-1] for t in sents],
         "ngram_actions": [log_ppl(lm, t, with_eos=False) for t in sents]}
    shuffled = np.zeros(len(sents))
    for _ in range(N_SHUFFLES):
        shuffled += [log_ppl(lm, [t[i] for i in rng.permutation(len(t))]) for t in sents]
    s["ngram_shuffled"] = shuffled / N_SHUFFLES
    s["unigram"] = [log_ppl(unigram, t) for t in sents]
    s["unigram_actions"] = [log_ppl(unigram, t, with_eos=False) for t in sents]
    s["ngram_base_vocab"] = [log_ppl(lm_base, t.split()) for t in test["sentence_base"]]
    s["length"] = test["n_actions"].to_numpy()
    info = {}
    s["logreg_ngrams"], info["ngrams"] = logistic(split["train"], split["dev"], test, (1, 3))
    s["logreg_counts"], info["counts"] = logistic(split["train"], split["dev"], test, (1, 1))
    aucs, diffs = bootstrap_auc(y, s, pairs=PAIRS_FULL)

    # ---- matched windows: test wallets that have one ----
    has = {k: v[v["window"].notna()] for k, v in split.items()}
    wtest = has["test"].reset_index(drop=True)
    wins = wtest["window"].str.split().tolist()
    w = {"win_ngram": [window_log_ppl(lm, t) for t in wins],
         "win_unigram": [window_log_ppl(unigram, t) for t in wins]}
    shuffled = np.zeros(len(wins))
    for _ in range(N_SHUFFLES):
        shuffled += [window_log_ppl(lm, [t[i] for i in rng.permutation(len(t))]) for t in wins]
    w["win_ngram_shuffled"] = shuffled / N_SHUFFLES
    for key, rng_ in (("win_logreg_ngrams", (1, 3)), ("win_logreg_counts", (1, 1))):
        w[key], info[key] = logistic(has["train"], has["dev"], wtest, rng_, text="window",
                                     size=False)
    waucs, wdiffs = bootstrap_auc(wtest["label"].to_numpy(), w, pairs=PAIRS_WINDOW)
    aucs |= waucs
    diffs |= wdiffs
    screen = leakage_screen(main_s)

    # ---- save ----
    scores = pd.DataFrame({"wallet": test["wallet"], "label": y, **s})
    scores = scores.merge(pd.DataFrame({"wallet": wtest["wallet"], **w}), on="wallet", how="left")
    scores.to_csv(DATA_DIR / "control_scores.tsv", sep="\t", index=False)
    pd.DataFrame([{"model": k, "description": d, "uses_order": o, "uses_labels": l,
                   "test_wallets": "matched windows" if k in WINDOW_KEYS else "all", **aucs[k]}
                  for k, (d, o, l) in MODELS.items()]).to_csv(
        RESULTS_DIR / "step7_auc.csv", index=False, float_format="%.4f")
    pd.DataFrame([{"comparison": k, **v} for k, v in diffs.items()]).to_csv(
        RESULTS_DIR / "step7_differences.csv", index=False, float_format="%.4f")
    screen.to_csv(RESULTS_DIR / "step7_leakage_screen.csv", index=False, float_format="%.4f")
    plot_forest(aucs, RESULTS_DIR / "step7_auc_comparison.png")

    better = lambda a, b: diffs[f"{a} - {b}"]["ci_low"] > 0  # noqa: E731
    above_chance = lambda k: aucs[k]["ci_low"] > 0.5  # noqa: E731
    verdict = {
        "H3_lm_beats_bag_of_actions": better("ngram", "unigram"),
        "H3_lm_beats_shuffled": better("ngram", "ngram_shuffled"),
        "H3_supervised": better("logreg_ngrams", "logreg_counts"),
        "H3_matched_windows_lm": better("win_ngram", "win_unigram"),
        "H3_matched_windows_supervised": better("win_logreg_ngrams", "win_logreg_counts"),
        "H4_no_suspicious_single_feature": not bool(screen["suspicious"].any()),
        "H4_perplexity_without_end_token": above_chance("ngram_actions"),
        "H4_perplexity_matched_windows": above_chance("win_ngram"),
        "H4_supervised_matched_windows": above_chance("win_logreg_ngrams"),
    }
    summary = {"test_wallets": {"safe": int((y == 0).sum()), "liquidated": int((y == 1).sum())},
               "matched_windows": {"window": WINDOW,
                                   "test_safe": int((wtest["label"] == 0).sum()),
                                   "test_liquidated": int((wtest["label"] == 1).sum())},
               "aucs": aucs, "paired_differences": diffs, "logistic_regression": info,
               "leakage_screen_top5": screen.head(5).to_dict(orient="records"),
               "verdict": verdict}
    (RESULTS_DIR / "step7_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- report ----
    print(f"matched windows ({WINDOW} actions): test {summary['matched_windows']['test_safe']:,} "
          f"safe, {summary['matched_windows']['test_liquidated']:,} liquidated\n")
    for title, g in GROUPS:
        print(title)
        for k, (d, _, _) in g.items():
            a = aucs[k]
            print(f"  {d:<48} {a['auc']:.3f}  [{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
        print()
    print("paired differences")
    for k, v in diffs.items():
        print(f"  {k:<42} {v['diff']:+.3f}  [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")
    print(f"\nlogistic regression: {info}")
    print("\nleakage screen, strongest single features:")
    print(screen.head(5).to_string(index=False, float_format="%.3f"))
    print("\nverdict:")
    for k, v in verdict.items():
        print(f"  {k:<40} {v}")
    print("\nWrote data/control_scores.tsv, results/step7_*.csv, results/step7_auc_comparison.png, "
          "results/step7_summary.json")


if __name__ == "__main__":
    main()
