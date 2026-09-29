
import json
import math
import pickle
from collections import Counter

import numpy as np
import pandas as pd

import plot_style as ps
from config import DATA_DIR, RESULTS_DIR, SEED
from evaluation import load_corpus
from ngram_lm import EOS
from step7_controls import add_matched_windows

MAX_N = 3
PRIOR_MASS = 1000   # total pseudo-count of the informative Dirichlet prior
Z_SIG = 3.29        # two-sided p < 0.001
TOP = 8             # patterns per class in the figure
NAMED = {           # 4-action vocabulary
    "BORROW BORROW": "two borrows in a row (no repay in between)",
    "BORROW BORROW BORROW": "three borrows in a row",
    "SUPPLY BORROW": "supply, then borrow (lever up)",
    "BORROW WITHDRAW": "borrow, then take collateral out",
    "BORROW REPAY": "borrow, then repay",
    "REPAY BORROW": "repay, then borrow again",
    "REPAY WITHDRAW": "repay, then take collateral out (exit)",
}


# --------------------------------------------------------------------------- #
def ngram_sets(sentences, max_n=MAX_N):
    """The set of distinct 1..max_n-grams in every sentence."""
    out = []
    for toks in sentences:
        out.append({" ".join(toks[i:i + n])
                    for n in range(1, max_n + 1) for i in range(len(toks) - n + 1)})
    return out


def fightin_words(sets, labels):
    """Weighted log-odds with an informative Dirichlet prior (Monroe et al. 2008),
    on wallet counts, separately for every n-gram order."""
    labels = np.asarray(labels)
    wallets = {c: int((labels == c).sum()) for c in (0, 1)}
    count = {0: Counter(), 1: Counter()}
    for grams, label in zip(sets, labels):
        count[label].update(grams)
    rows = []
    for n in range(1, MAX_N + 1):
        c = {k: {g: v for g, v in count[k].items() if g.count(" ") == n - 1} for k in (0, 1)}
        pooled = Counter(c[0]) + Counter(c[1])
        total = sum(pooled.values())
        size = {k: sum(c[k].values()) for k in (0, 1)}
        for gram, y in pooled.items():
            a = PRIOR_MASS * y / total  # prior pseudo-count, proportional to the pooled count
            y1, y0 = c[1].get(gram, 0), c[0].get(gram, 0)
            delta = (math.log((y1 + a) / (size[1] + PRIOR_MASS - y1 - a))
                     - math.log((y0 + a) / (size[0] + PRIOR_MASS - y0 - a)))
            rows.append({"ngram": gram, "n": n,
                         "share_liquidated": y1 / wallets[1], "share_safe": y0 / wallets[0],
                         "log_odds": delta, "z": delta / math.sqrt(1 / (y1 + a) + 1 / (y0 + a))})
    return pd.DataFrame(rows)


def odds_ratio(has, labels):
    """Odds ratio (liquidated vs safe) of containing a pattern, 95% CI (Woolf)."""
    has, labels = np.asarray(has), np.asarray(labels)
    a, b = has[labels == 1].sum() + 0.5, (~has[labels == 1]).sum() + 0.5
    c, d = has[labels == 0].sum() + 0.5, (~has[labels == 0]).sum() + 0.5
    log_or, se = math.log(a * d / (b * c)), math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    return math.exp(log_or), math.exp(log_or - 1.96 * se), math.exp(log_or + 1.96 * se)


def contains(tokens, pattern):
    p = pattern.split()
    return any(tokens[i:i + len(p)] == p for i in range(len(tokens) - len(p) + 1))


# --------------------------------------------------------------------------- #
def pretty(gram):
    return "  →  ".join(gram.split())


def plot_contrast(table, path):
    top_liq = table.nlargest(TOP, "z_matched")
    top_safe = table.nsmallest(TOP, "z_matched")
    rows = pd.concat([top_liq, top_safe.iloc[::-1]])
    ys = np.arange(len(rows))[::-1]
    fig, ax = ps.plt.subplots(figsize=(8.5, 6))
    colors = [ps.LIQUIDATED if z > 0 else ps.SAFE for z in rows["z_matched"]]
    ax.barh(ys, rows["z_matched"], height=0.55, color=colors)
    ax.axvline(0, color=ps.AXIS, linewidth=1)
    to_axis = ax.get_yaxis_transform()
    for y, (_, r) in zip(ys, rows.iterrows()):
        ax.text(1.02, y, f"{r['share_liquidated_matched']:.1%} vs {r['share_safe_matched']:.1%}",
                transform=to_axis, va="center", color=ps.INK_2, fontsize=9)
    ax.text(1.02, len(rows) - 0.3, "windows containing it\nliquidated vs safe", transform=to_axis,
            va="bottom", color=ps.MUTED, fontsize=8.5)
    ax.set_yticks(ys, [pretty(g) for g in rows["ngram"]], fontsize=9)
    ax.axhline(TOP - 0.5, color=ps.GRID, linewidth=1)
    ax.grid(axis="y", visible=False)
    lim = rows["z_matched"].abs().max() * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("← typical of safe wallets        z-score (weighted log-odds)        "
                  "typical of liquidated wallets →")
    ax.set_title("The grammar of each class: most characteristic 2-3 action patterns\n"
                 "(matched 4-action windows ending mid-stream)")
    ps.save(fig, path)


def plot_examples(examples, path):
    fig, axes = ps.plt.subplots(len(examples), 1, figsize=(8.5, 5.8), sharey=True)
    top = max(max(ex["surprise_bits"]) for ex in examples)
    for ax, ex in zip(axes, examples):
        bits = ex["surprise_bits"]
        x = np.arange(len(bits))
        ax.bar(x, bits, width=0.55, color=ps.CLASS_COLOR[ex["label"]])
        ax.set_xticks(x, ex["tokens"], rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("surprise (bits)")
        ax.set_ylim(0, top * 1.45)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{ps.CLASS_NAME[ex['label']]} wallet {ex['wallet'][:8]}…{ex['wallet'][-4:]}"
                     f"  ·  perplexity {ex['perplexity']:.2f}", fontsize=10.5)
        # bracket over the characteristic pattern of the class
        a, b = ex["pattern_span"]
        yb = top * 1.18
        ax.plot([a - 0.3, a - 0.3, b - 0.7, b - 0.7], [yb - 0.15, yb, yb, yb - 0.15],
                color=ps.INK_2, linewidth=1)
        ax.text(a - 0.3, yb + 0.08, ex["pattern_note"], ha="left", va="bottom",
                fontsize=8.5, color=ps.INK_2)
    fig.suptitle("Where the safe-behavior model is surprised, token by token",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.935, "Bars: surprise of each token, −log₂ P. Bracket: the most "
             "characteristic pattern of the class (Fightin' Words, matched windows).",
             fontsize=8.5, color=ps.MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ps.save(fig, path)


def pick_example(test, lm, label, pattern, shares):
    """A typical test wallet of the class that contains `pattern` (6-10 actions,
    perplexity closest to its class median)."""
    cls = test[test["label"] == label]
    pool = cls[cls["length"].between(6, 10)
               & cls["sentence"].map(lambda s: contains(s.split(), pattern))]
    median = cls["ppl"].median()
    row = pool.loc[(pool["ppl"] - median).abs().idxmin()]
    tokens = row["sentence"].split()
    p = pattern.split()
    start = next(i for i in range(len(tokens)) if tokens[i:i + len(p)] == p)
    lp = lm.token_logprobs(tokens)
    typical = "liquidated" if label == 1 else "safe"
    return {"wallet": row["wallet"], "label": label, "tokens": tokens + [EOS],
            "surprise_bits": [-v / math.log(2) for v in lp], "perplexity": row["ppl"],
            "pattern": pattern, "pattern_span": (start, start + len(p)),
            "pattern_note": f"typical of {typical} wallets: in {shares[0]:.1%} of liquidated vs "
                            f"{shares[1]:.1%} of safe windows"}


# --------------------------------------------------------------------------- #
def main():
    ps.apply()
    rng = np.random.default_rng(SEED)
    corpus = load_corpus()
    main_s = add_matched_windows(corpus[corpus["sample_main"]], rng)  # the same windows as Step 7
    labels = main_s["label"].to_numpy()
    win = main_s[main_s["window"].notna()]
    win_labels = win["label"].to_numpy()

    full = fightin_words(ngram_sets(main_s["sentence"].str.split()), labels)
    matched = fightin_words(ngram_sets(win["window"].str.split()), win_labels)
    table = matched.merge(full, on=["ngram", "n"], how="outer", suffixes=("_matched", "_full"))
    table["robust"] = ((table["z_matched"].abs() > Z_SIG) & (table["z_full"].abs() > Z_SIG)
                       & (np.sign(table["z_matched"]) == np.sign(table["z_full"])))
    table = table.sort_values("z_matched", ascending=False, ignore_index=True)
    table.to_csv(RESULTS_DIR / "step8_ngram_contrast.csv", index=False, float_format="%.4f")
    bi_tri = table[table["n"] >= 2].dropna(subset=["z_matched"])
    plot_contrast(bi_tri, RESULTS_DIR / "step8_characteristic_ngrams.png")

    # ---- named patterns (4-action vocabulary) ----
    base_full = main_s["sentence_base"].str.split()
    base_win = win["window"].str.replace(r"_(STABLE|VOL)\b", "", regex=True).str.split()
    named = []
    for pattern, meaning in NAMED.items():
        row = {"pattern": pattern, "meaning": meaning}
        for name, sents, lab in (("matched", base_win, win_labels), ("full", base_full, labels)):
            has = np.array([contains(t, pattern) for t in sents])
            ratio, lo, hi = odds_ratio(has, lab)
            row |= {f"share_liquidated_{name}": has[lab == 1].mean(),
                    f"share_safe_{name}": has[lab == 0].mean(),
                    f"odds_ratio_{name}": ratio, f"ci_low_{name}": lo, f"ci_high_{name}": hi}
        named.append(row)
    named = pd.DataFrame(named)
    named.to_csv(RESULTS_DIR / "step8_named_patterns.csv", index=False, float_format="%.4f")

    # ---- two annotated example sentences (test wallets, full histories) ----
    with open(DATA_DIR / "ngram_lm.pkl", "rb") as f:
        lm = pickle.load(f)["sentence"]
    scores = pd.read_csv(DATA_DIR / "scores.tsv", sep="\t", usecols=["wallet", "ppl"])
    test = main_s[main_s["split"] == "test"].merge(scores, on="wallet")
    examples = []
    for label, r in ((0, bi_tri.iloc[-1]), (1, bi_tri.iloc[0])):
        shares = (r["share_liquidated_matched"], r["share_safe_matched"])
        examples.append(pick_example(test, lm, label, r["ngram"], shares))
    plot_examples(examples, RESULTS_DIR / "step8_examples.png")

    summary = {
        "wallets": {"safe": int((labels == 0).sum()), "liquidated": int((labels == 1).sum())},
        "matched_windows": {"safe": int((win_labels == 0).sum()),
                            "liquidated": int((win_labels == 1).sum())},
        "patterns_compared": int(len(table)),
        "significant_matched": int((table["z_matched"].abs() > Z_SIG).sum()),
        "robust_in_both": int(table["robust"].sum()),
        "top_liquidated_matched": bi_tri.head(TOP)[["ngram", "z_matched", "z_full"]]
        .to_dict(orient="records"),
        "top_safe_matched": bi_tri.tail(TOP).iloc[::-1][["ngram", "z_matched", "z_full"]]
        .to_dict(orient="records"),
        "named_patterns": named.to_dict(orient="records"),
        "examples": [{k: v for k, v in e.items()} for e in examples],
    }
    (RESULTS_DIR / "step8_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- report ----
    print(f"patterns compared: {len(table):,} | significant (matched): "
          f"{summary['significant_matched']:,} | robust in both settings: "
          f"{summary['robust_in_both']:,}\n")
    for title, part in (("typical of LIQUIDATED wallets (matched)", bi_tri.head(TOP)),
                        ("typical of SAFE wallets (matched)", bi_tri.tail(TOP).iloc[::-1])):
        print(title)
        for _, r in part.iterrows():
            print(f"  {pretty(r['ngram']):<60} z {r['z_matched']:+7.1f} (full {r['z_full']:+7.1f})"
                  f"  {r['share_liquidated_matched']:5.1%} vs {r['share_safe_matched']:5.1%}"
                  f"{'  robust' if r['robust'] else ''}")
        print()
    print("named patterns (odds ratio liquidated vs safe, matched | full):")
    for _, r in named.iterrows():
        print(f"  {r['pattern']:<22} {r['odds_ratio_matched']:5.2f} [{r['ci_low_matched']:.2f}-"
              f"{r['ci_high_matched']:.2f}] | {r['odds_ratio_full']:5.2f}   ({r['meaning']})")
    print("\nWrote results/step8_ngram_contrast.csv, results/step8_named_patterns.csv, "
          "results/step8_characteristic_ngrams.png, results/step8_examples.png, "
          "results/step8_summary.json")


if __name__ == "__main__":
    main()
