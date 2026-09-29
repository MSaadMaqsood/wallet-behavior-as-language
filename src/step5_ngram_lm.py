
import json
import pickle
import time

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, SEED
from ngram_lm import BOS, KneserNeyLM

SHARES = (0.6, 0.2, 0.2)  # train / dev / test
ORDERS = range(1, 11)
VOCABS = {"sentence": "8 action+class tokens", "sentence_base": "4 action tokens"}


# --------------------------------------------------------------------------- #
def make_splits(corpus, rng):
    """Random 60/20/20 split inside each group of wallets."""
    group = np.select(
        [corpus["sample_main"] & (corpus["label"] == 0),
         corpus["sample_main"] & (corpus["label"] == 1),
         corpus["sample_new"] & ~corpus["sample_main"]],
        ["safe", "liquidated", "liquidated_dust"], default="old_only_never_borrowed")
    split = np.empty(len(corpus), dtype=object)
    for g in sorted(set(group)):
        idx = rng.permutation(np.flatnonzero(group == g))
        a, b = int(SHARES[0] * len(idx)), int((SHARES[0] + SHARES[1]) * len(idx))
        split[idx[:a]], split[idx[a:b]], split[idx[b:]] = "train", "dev", "test"
    return pd.DataFrame({"wallet": corpus["wallet"], "group": group, "split": split})


def sum_to_one_error(lm, sentences, rng, n_contexts=1000):
    """Largest |1 - sum of P(w | context) over all predictable w| on random contexts."""
    worst = 0.0
    for i in rng.choice(len(sentences), size=n_contexts):
        seq = lm._pad(sentences[i])
        pos = rng.integers(lm.n - 1, len(seq))
        ctx = seq[pos - lm.n + 1:pos]
        worst = max(worst, abs(1 - sum(lm.prob(w, ctx) for w in lm.vocab)))
    return worst


def nltk_difference(train, dev, n=3):
    """Largest difference to NLTK's KneserNeyInterpolated under NLTK's settings."""
    from nltk.lm import KneserNeyInterpolated
    from nltk.util import everygrams

    ours = KneserNeyLM(n, discount=0.1, unigram_discount=False).fit(train)
    padded = [ours._pad(s) for s in train]
    ref = KneserNeyInterpolated(n, discount=0.1)
    # NLTK learns from the same padded sentences; n-grams ending in <BOS> are
    # never predicted, so they are left out, exactly as in our counts
    ref.fit([[g for g in everygrams(p, max_len=n) if g[-1] != BOS] for p in padded],
            [t for p in padded for t in p])
    diff = 0.0
    for s in dev:
        seq = ours._pad(s)
        for i in range(n - 1, len(seq)):
            ctx = seq[i - n + 1:i]
            diff = max(diff, abs(ours.prob(seq[i], ctx) - ref.score(seq[i], ctx)))
    return diff


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    corpus = pd.read_csv(DATA_DIR / "corpus.tsv", sep="\t")
    splits = make_splits(corpus, rng)
    corpus = corpus.merge(splits[["wallet", "split"]], on="wallet")

    safe = corpus[corpus["sample_main"] & (corpus["label"] == 0)]
    train_rows, dev_rows = safe[safe["split"] == "train"], safe[safe["split"] == "dev"]
    print(f"safe train {len(train_rows):,} sentences | safe dev {len(dev_rows):,} sentences\n")

    # ---- choose n on safe-dev perplexity ----
    rows, best = [], {}
    for vocab in VOCABS:
        train = train_rows[vocab].str.split().tolist()
        dev = dev_rows[vocab].str.split().tolist()
        for n in ORDERS:
            t0 = time.time()
            lm = KneserNeyLM(n).fit(train)
            pp = lm.perplexity(dev)
            rows.append({"vocab": vocab, "n": n, "dev_perplexity": round(pp, 4),
                         "discounts": {k: round(d, 3) for k, d in lm.discount.items()}})
            print(f"  {vocab:<14} n={n}  dev perplexity {pp:7.4f}   ({time.time() - t0:4.1f} s)")
            if vocab not in best or pp < best[vocab][1]:
                best[vocab] = (lm, pp)
        print()
    grid = pd.DataFrame(rows)

    # ---- correctness checks ----
    lm_main = best["sentence"][0]
    dev_main = dev_rows["sentence"].str.split().tolist()
    train_main = train_rows["sentence"].str.split().tolist()
    small = rng.choice(len(train_main), size=2000, replace=False)
    checks = {
        "probabilities_sum_to_one_max_error": float(sum_to_one_error(lm_main, dev_main, rng)),
        "max_difference_to_nltk_n3": float(nltk_difference(
            [train_main[i] for i in small], dev_main[:300])),
        "liquidated_wallets_in_lm_training": int((train_rows["label"] == 1).sum()),
    }

    # ---- save ----
    splits.to_csv(DATA_DIR / "splits.tsv", sep="\t", index=False)
    with open(DATA_DIR / "ngram_lm.pkl", "wb") as f:
        pickle.dump({v: m for v, (m, _) in best.items()}, f)
    grid.drop(columns="discounts").to_csv(RESULTS_DIR / "step5_dev_perplexity.csv", index=False)
    split_counts = splits.groupby(["group", "split"]).size().unstack().astype(int)
    summary = {
        "split_shares": dict(zip(("train", "dev", "test"), SHARES)),
        "split_counts": split_counts.to_dict(orient="index"),
        "lm_training": {"safe_train_sentences": len(train_rows),
                        "safe_dev_sentences": len(dev_rows)},
        "chosen": {v: {"n": m.n, "dev_perplexity": round(pp, 4),
                       "discounts": {k: round(d, 3) for k, d in m.discount.items()},
                       "note": VOCABS[v]} for v, (m, pp) in best.items()},
        "largest_n_tried": max(ORDERS),
        "checks": checks,
    }
    (RESULTS_DIR / "step5_summary.json").write_text(json.dumps(summary, indent=2))

    print("split sizes:\n" + split_counts[["train", "dev", "test"]].to_string() + "\n")
    for v, (m, pp) in best.items():
        flag = "  <- largest n tried" if m.n == max(ORDERS) else ""
        print(f"chosen {v:<14}: n = {m.n}, dev perplexity {pp:.4f}{flag}")
    print(f"\nchecks: {checks}")
    print("\nWrote data/splits.tsv, data/ngram_lm.pkl, results/step5_dev_perplexity.csv, "
          "results/step5_summary.json")


if __name__ == "__main__":
    main()
