
import json

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, SEED

MAX_LEN = 300
ACTIONS = ["SUPPLY", "BORROW", "REPAY", "WITHDRAW"]
VOCAB = [f"{a}_{c}" for a in ACTIONS for c in ("STABLE", "VOL")]
SPECIALS = ["<BOS>", "<EOS>"]
CLASS_NAME = {0: "safe", 1: "liquidated"}


def length_stats(n):
    q = n.quantile([0.5, 0.9]).to_numpy()
    return {"median": float(q[0]), "mean": round(float(n.mean()), 1), "p90": float(q[1]),
            "max": int(n.max())}


def token_frequencies(corpus, column, vocab):
    """Count of each token per class, and its share of all tokens of that class."""
    rows = []
    for label, group in corpus.groupby("label"):
        counts = group[column].str.split().explode().value_counts()
        total = counts.sum()
        for tok in vocab:
            rows.append({"vocab": column, "token": tok, "class": CLASS_NAME[label],
                         "count": int(counts.get(tok, 0)),
                         "share": round(counts.get(tok, 0) / total, 4)})
    table = pd.DataFrame(rows).pivot_table(index=["vocab", "token"], columns="class",
                                           values=["count", "share"])
    table.columns = [f"{v}_{c}" for v, c in table.columns]
    counts = [c for c in table.columns if c.startswith("count_")]
    table[counts] = table[counts].astype(int)
    table["share_ratio_liq_vs_safe"] = (table["share_liquidated"] / table["share_safe"]).round(2)
    return table.reset_index()


def examples(corpus, rng, per_class=4):
    lines = []
    for label in (0, 1):
        pool = corpus[(corpus["label"] == label) & corpus["n_actions"].between(6, 14)]
        for _, r in pool.sample(per_class, random_state=rng).iterrows():
            lines += [f"{CLASS_NAME[label].upper():<11} {r['wallet']}  ({r['n_actions']} actions)",
                      f"    <BOS> {r['sentence']} <EOS>", ""]
    return "\n".join(lines)


def main():
    tl = pd.read_parquet(DATA_DIR / "timelines.parquet",
                         columns=["wallet", "block", "event", "asset_class", "pre_cutoff"])
    wallets = pd.read_parquet(DATA_DIR / "wallets.parquet")

    # ---- leakage checks on what the model may see ----
    first_liq_block = tl[tl["event"] == "LIQUIDATION"].groupby("wallet")["block"].min()
    seen = tl[tl["pre_cutoff"]]  # timelines are sorted by wallet, block, log_index
    same_block = int((seen["block"] >= seen["wallet"].map(first_liq_block)).sum())

    # ---- tokens: ACTION_CLASS ----
    cls = seen["asset_class"].astype(str).str.replace("VOLATILE", "VOL")
    tokens = seen["event"].astype(str) + "_" + cls
    unexpected = sorted(set(tokens.unique()) - set(VOCAB))

    sentences = tokens.groupby(seen["wallet"], sort=False).agg(list)
    corpus = wallets[["wallet", "label", "sample_main", "sample_new", "sample_old"]].copy()
    toks = corpus["wallet"].map(sentences)
    empty = int(toks.isna().sum())
    corpus["n_actions"] = toks.str.len().fillna(0).astype(int)
    corpus["truncated"] = corpus["n_actions"] > MAX_LEN
    corpus["sentence"] = toks.map(lambda t: " ".join(t[-MAX_LEN:]))  # keep the LAST actions
    corpus["sentence_base"] = corpus["sentence"].str.replace(r"_(STABLE|VOL)\b", "", regex=True)
    corpus["length"] = corpus["sentence"].str.count(" ") + 1

    checks = {
        "unexpected_tokens": unexpected,  # e.g. a LIQUIDATION token would show up here
        "liquidation_token_in_corpus": bool(corpus["sentence"].str.contains("LIQUID").any()),
        "visible_actions_in_or_after_liquidation_block": same_block,
        "empty_sentences": empty,
    }
    assert not unexpected and not checks["liquidation_token_in_corpus"] and same_block == 0 \
        and empty == 0, f"leakage / tokenization check failed: {checks}"

    # ---- statistics on the main sample ----
    main_s = corpus[corpus["sample_main"]]
    freq = pd.concat([token_frequencies(main_s, "sentence", VOCAB),
                      token_frequencies(main_s, "sentence_base", ACTIONS)])
    summary = {
        "vocabulary": {"sentence": VOCAB + SPECIALS, "sentence_base": ACTIONS + SPECIALS},
        "max_len": MAX_LEN,
        "sentences": {name: {CLASS_NAME[l]: int(((corpus[flag]) & (corpus["label"] == l)).sum())
                             for l in (0, 1)}
                      for name, flag in (("main", "sample_main"), ("new", "sample_new"),
                                         ("old", "sample_old"))},
        "main_sample": {CLASS_NAME[l]: {"tokens": int(g["length"].sum()),
                                        "length_after_cap": length_stats(g["length"]),
                                        "truncated_wallets": int(g["truncated"].sum())}
                        for l, g in main_s.groupby("label")},
        "checks": checks,
    }

    corpus.to_csv(DATA_DIR / "corpus.tsv", sep="\t", index=False)
    freq.to_csv(RESULTS_DIR / "step4_token_frequencies.csv", index=False)
    (RESULTS_DIR / "step4_examples.txt").write_text(
        examples(main_s, np.random.default_rng(SEED)), encoding="utf-8")
    (RESULTS_DIR / "step4_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- report ----
    print(f"vocabulary: {len(VOCAB)} action tokens + {SPECIALS}  (base: {len(ACTIONS)} actions)")
    print(f"corpus    : {len(corpus):,} sentences | main sample "
          f"{summary['sentences']['main']['liquidated']:,} liquidated + "
          f"{summary['sentences']['main']['safe']:,} safe")
    for name, s in summary["main_sample"].items():
        L = s["length_after_cap"]
        print(f"  {name:<11} {s['tokens']:>9,} tokens | length median {L['median']:.0f}, "
              f"mean {L['mean']}, p90 {L['p90']:.0f}, max {L['max']} | "
              f"truncated {s['truncated_wallets']:,}")
    print("\ntoken share per class (main sample)")
    enriched = freq[freq["vocab"] == "sentence"]
    for _, r in enriched.iterrows():
        print(f"  {r['token']:<16} safe {r['share_safe']:6.1%}   liquidated "
              f"{r['share_liquidated']:6.1%}   ratio {r['share_ratio_liq_vs_safe']:.2f}")
    print(f"\nchecks: {checks}")
    print("\n" + (RESULTS_DIR / "step4_examples.txt").read_text(encoding="utf-8"))
    print("Wrote data/corpus.tsv, results/step4_token_frequencies.csv, "
          "results/step4_examples.txt, results/step4_summary.json")


if __name__ == "__main__":
    main()
