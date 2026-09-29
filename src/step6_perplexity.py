
import json
import math
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

import plot_style as ps
from config import DATA_DIR, RESULTS_DIR
from evaluation import bootstrap_auc, load_corpus, mann_whitney
from ngram_lm import KneserNeyLM


def perplexities(lm, sentences):
    """Per wallet: perplexity over actions + <EOS>, and over the actions only."""
    full, actions = [], []
    for s in sentences:
        lp = lm.token_logprobs(s)
        full.append(math.exp(-sum(lp) / len(lp)))
        actions.append(math.exp(-sum(lp[:-1]) / (len(lp) - 1)))
    return full, actions


def evaluate(df, score):
    y = df["label"].to_numpy()
    s = df[score].to_numpy()
    q = {c: [float(v) for v in np.percentile(s[y == c], [25, 50, 75])] for c in (0, 1)}
    boot = bootstrap_auc(y, {score: s})[0][score]
    return {"safe_wallets": int((y == 0).sum()), "liquidated_wallets": int((y == 1).sum()),
            "median_safe": q[0][1], "iqr_safe": [q[0][0], q[0][2]],
            "median_liquidated": q[1][1], "iqr_liquidated": [q[1][0], q[1][2]],
            **mann_whitney(y, s), **boot}


# --------------------------------------------------------------------------- #
def plot_distributions(df, path):
    fig, ax = ps.plt.subplots(figsize=(7.5, 4))
    logp = np.log10(df["ppl"])
    bins = np.linspace(logp.min(), np.percentile(logp, 99.5), 60)
    heights = {}
    for c in (0, 1):
        v = logp[df["label"] == c]
        h, _ = np.histogram(v, bins, weights=np.ones(len(v)) / len(v))  # share of the class
        heights[c] = h
        ax.stairs(h, bins, fill=True, color=ps.CLASS_COLOR[c], alpha=0.10)
        ax.stairs(h, bins, color=ps.CLASS_COLOR[c], linewidth=2,
                  label=f"{ps.CLASS_NAME[c]} (n = {len(v):,}, median {10 ** v.median():.1f})")
        ax.axvline(v.median(), color=ps.CLASS_COLOR[c], linewidth=1)

    # label the most common sentence of each class: they make the two spikes
    for c, text_xy in ((1, (np.log10(5.6), 0.085)), (0, (np.log10(1.42), 0.105))):
        g = df[df["label"] == c]
        sentence = g["sentence"].value_counts().index[0]
        share = {k: (df.loc[df["label"] == k, "sentence"] == sentence).mean() for k in (0, 1)}
        x = np.log10(g.loc[g["sentence"] == sentence, "ppl"].iloc[0])
        y = heights[c][min(np.searchsorted(bins, x) - 1, len(heights[c]) - 1)]
        tokens = sentence.split()
        wrapped = "\n".join("  ".join(tokens[i:i + 2]) for i in range(0, len(tokens), 2))
        ax.annotate(f"{wrapped}\n{share[1]:.1%} of liquidated · {share[0]:.1%} of safe",
                    xy=(x, y), xytext=text_xy, fontsize=8.5, color=ps.INK_2,
                    arrowprops={"arrowstyle": "-", "color": ps.MUTED, "lw": 0.8})

    ticks = [1.5, 2, 3, 4, 6, 10, 15, 25]
    ax.set_xticks(np.log10(ticks), [str(t) for t in ticks])
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel("Perplexity under the safe-behavior model (log scale)")
    ax.set_ylabel("Share of wallets")
    ax.yaxis.set_major_formatter(ps.plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Liquidated wallets surprise the model of safe behavior more")
    ax.legend(loc="upper right")
    ps.save(fig, path)


def plot_roc(df, result, path):
    fpr, tpr, _ = roc_curve(df["label"], df["ppl"])
    fig, ax = ps.plt.subplots(figsize=(4.2, 4.2))
    ax.plot([0, 1], [0, 1], color=ps.AXIS, linewidth=1)
    ax.plot(fpr, tpr, color=ps.INK)
    ax.text(0.97, 0.05, f"AUC {result['auc']:.3f}\n95% CI {result['ci_low']:.3f}–"
            f"{result['ci_high']:.3f}", ha="right", va="bottom", color=ps.INK_2)
    ax.text(0.62, 0.55, "chance", color=ps.MUTED, rotation=45, ha="center")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Safe wallets flagged (false-positive rate)",
           ylabel="Liquidated wallets caught (true-positive rate)")
    ax.set_title("Perplexity as a one-number risk score")
    ps.save(fig, path)


# --------------------------------------------------------------------------- #
def main():
    ps.apply()
    corpus = load_corpus()
    sentences = corpus["sentence"].str.split().tolist()
    with open(DATA_DIR / "ngram_lm.pkl", "rb") as f:
        lm_main = pickle.load(f)["sentence"]

    # comparison model: same n, trained on the OLD study's safe train wallets
    old_train = corpus["sample_old"] & (corpus["label"] == 0) & (corpus["split"] == "train")
    lm_old = KneserNeyLM(lm_main.n).fit([sentences[i] for i in np.flatnonzero(old_train)])

    corpus["ppl"], corpus["ppl_actions"] = perplexities(lm_main, sentences)
    corpus["ppl_old"], _ = perplexities(lm_old, sentences)
    scores = corpus[["wallet", "label", "group", "split", "sample_main", "sample_new",
                     "sample_old", "length", "ppl", "ppl_actions", "ppl_old"]]
    scores.to_csv(DATA_DIR / "scores.tsv", sep="\t", index=False)

    test = corpus[corpus["split"] == "test"]
    samples = {"main": (test[test["sample_main"]], "ppl"),
               "with_dust": (test[test["sample_new"]], "ppl"),
               "old_study": (test[test["sample_old"]], "ppl_old")}
    results = {name: evaluate(df, score) for name, (df, score) in samples.items()}

    table = pd.DataFrame(results).T
    table.to_csv(RESULTS_DIR / "step6_results.csv", float_format="%.4g")
    main_df = samples["main"][0]
    plot_distributions(main_df, RESULTS_DIR / "step6_perplexity_by_class.png")
    plot_roc(main_df, results["main"], RESULTS_DIR / "step6_roc.png")
    summary = {"model": {"n": lm_main.n, "trained_on": "safe train wallets of the main sample"},
               "results": results,
               "H1_supported": bool(results["main"]["p_value"] < 0.001
                                    and results["main"]["median_liquidated"]
                                    > results["main"]["median_safe"]),
               "H2_supported": bool(results["main"]["ci_low"] > 0.5)}
    (RESULTS_DIR / "step6_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"{'sample':<10} {'safe':>7} {'liq':>6}  {'median ppl safe':>15} {'liq':>6}  "
          f"{'p (MWU)':>9} {'r_rb':>6}  {'AUC [95% CI]':>22}")
    for name, r in results.items():
        print(f"{name:<10} {r['safe_wallets']:>7,} {r['liquidated_wallets']:>6,}  "
              f"{r['median_safe']:>15.2f} {r['median_liquidated']:>6.2f}  {r['p_value']:>9.1e} "
              f"{r['rank_biserial']:>6.3f}  {r['auc']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print(f"\nH1 supported: {summary['H1_supported']} | H2 supported: {summary['H2_supported']}")
    print("\nWrote data/scores.tsv, results/step6_results.csv, results/step6_*.png, "
          "results/step6_summary.json")


if __name__ == "__main__":
    main()
