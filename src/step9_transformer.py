
import copy
import json
import math
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import plot_style as ps
from config import DATA_DIR, RESULTS_DIR, SEED
from evaluation import bootstrap_auc, load_corpus
from step4_tokenize import VOCAB
from step7_controls import WINDOW, add_matched_windows

PAD, BOS, EOS = 0, 1, 2
ITOS = ["<PAD>", "<BOS>", "<EOS>"] + VOCAB
STOI = {t: i for i, t in enumerate(ITOS)}
D_MODEL, HEADS, LAYERS, FF, DROPOUT = 64, 4, 2, 128, 0.1
MAX_LEN = 302            # <BOS> + 300 actions + <EOS>
TOKENS_PER_BATCH = 8192  # padded tokens per batch
MAX_EPOCHS, PATIENCE, LR = 50, 3, 3e-3  # stop after 3 epochs without a better dev perplexity

SCORES = {"full": "full score", "eos": "end token only", "actions": "actions only (no end token)",
          "window": f"matched {WINDOW}-action windows"}


class TransformerLM(nn.Module):
    """Causal Transformer: every position attends only to the tokens before it."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(len(ITOS), D_MODEL, padding_idx=PAD)
        self.pos = nn.Embedding(MAX_LEN, D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, HEADS, FF, DROPOUT, batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.out = nn.Linear(D_MODEL, len(ITOS))

    def forward(self, x):
        t = x.size(1)
        h = self.tok(x) + self.pos(torch.arange(t))
        causal = nn.Transformer.generate_square_subsequent_mask(t)
        return self.out(self.norm(self.blocks(h, mask=causal, is_causal=True)))


# --------------------------------------------------------------------------- #
def encode(sentences):
    return [[BOS] + [STOI[t] for t in s.split()] + [EOS] for s in sentences]


def batches(seqs, rng=None):
    """Batches of sequences of similar length (little padding); shuffled if rng."""
    order = np.argsort([len(s) for s in seqs], kind="stable")
    groups, cur, longest = [], [], 0
    for i in order:
        if cur and max(longest, len(seqs[i])) * (len(cur) + 1) > TOKENS_PER_BATCH:
            groups.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, len(seqs[i]))
    groups.append(cur)
    if rng is not None:
        rng.shuffle(groups)
    for g in groups:
        x = torch.full((len(g), max(len(seqs[i]) for i in g)), PAD, dtype=torch.long)
        for r, i in enumerate(g):
            x[r, :len(seqs[i])] = torch.tensor(seqs[i])
        yield g, x


@torch.no_grad()
def target_logprobs(model, seqs):
    """Natural-log probability of every predicted token (actions + <EOS>)."""
    model.eval()
    out = [None] * len(seqs)
    for g, x in batches(seqs):
        logp = torch.log_softmax(model(x[:, :-1]), dim=-1)
        lp = logp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
        for r, i in enumerate(g):
            out[i] = lp[r, :len(seqs[i]) - 1].numpy()
    return out


def perplexity(logprobs):
    return math.exp(-sum(lp.sum() for lp in logprobs) / sum(len(lp) for lp in logprobs))


def train(model, train_seqs, dev_seqs):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    rng = np.random.default_rng(SEED)
    history, best_pp, best_state, bad = [], math.inf, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0, total, count = time.time(), 0.0, 0
        for _, x in batches(train_seqs, rng):
            logits = model(x[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = int((x[:, 1:] != PAD).sum())
            total, count = total + loss.item() * n, count + n
        dev_pp = perplexity(target_logprobs(model, dev_seqs))
        lr = opt.param_groups[0]["lr"]
        plateau.step(dev_pp)
        history.append({"epoch": epoch, "learning_rate": lr,
                        "train_perplexity": round(math.exp(total / count), 4),
                        "dev_perplexity": round(dev_pp, 4), "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {epoch:>2}: lr {lr:.5f} | train perplexity {math.exp(total / count):.4f} | "
              f"dev perplexity {dev_pp:.4f} | {time.time() - t0:.0f} s", flush=True)
        if dev_pp < best_pp:
            best_pp, best_state, bad = dev_pp, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return history, best_pp


def scores_from_logprobs(logprobs, window_end):
    """The four scores (higher = more surprising) from per-token log-probabilities.
    lp[j] predicts action j; the last entry predicts <EOS>."""
    out = {k: [] for k in SCORES}
    for lp, end in zip(logprobs, window_end):
        lp = np.asarray(lp)
        out["full"].append(-lp.mean())
        out["eos"].append(-lp[-1])
        out["actions"].append(-lp[:-1].mean())
        out["window"].append(-lp[end - WINDOW:end].mean() if not pd.isna(end) else np.nan)
    return out


def plot_comparison(aucs, path):
    fig, ax = ps.plt.subplots(figsize=(7.5, 3.4))
    ys = np.arange(len(SCORES))[::-1]
    for y, key in zip(ys, SCORES):
        for model, color, marker, dy in (("ngram", ps.INK, "o", 0.12),
                                         ("transformer", ps.MUTED, "s", -0.12)):
            a = aucs[f"{model}_{key}"]
            ax.plot([a["ci_low"], a["ci_high"]], [y + dy, y + dy], color=color)
            ax.plot(a["auc"], y + dy, marker, color=color, markersize=8,
                    markeredgecolor=ps.SURFACE, markeredgewidth=2,
                    label=("n-gram LM (n = 6)" if model == "ngram" else "Transformer LM")
                    if y == ys[0] else None)
    ax.axvline(0.5, color=ps.AXIS, linewidth=1)
    ax.set_yticks(ys, list(SCORES.values()))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC-AUC on test wallets (95% bootstrap interval)")
    ax.set_title("n-gram vs Transformer language model of safe behavior")
    ax.legend(loc="lower right")
    ps.save(fig, path)


# --------------------------------------------------------------------------- #
def main():
    ps.apply()
    torch.manual_seed(SEED)
    corpus = load_corpus()
    main_s = add_matched_windows(corpus[corpus["sample_main"]], np.random.default_rng(SEED))
    safe = main_s[main_s["label"] == 0]
    train_seqs = encode(safe[safe["split"] == "train"]["sentence"])
    dev_seqs = encode(safe[safe["split"] == "dev"]["sentence"])

    model = TransformerLM()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Transformer LM: {n_params:,} parameters | train {len(train_seqs):,} safe sentences")
    history, dev_pp = train(model, train_seqs, dev_seqs)
    torch.save(model.state_dict(), DATA_DIR / "transformer_lm.pt")

    with open(DATA_DIR / "ngram_lm.pkl", "rb") as f:
        lm = pickle.load(f)["sentence"]
    ngram_dev_pp = lm.perplexity([s.split() for s in safe[safe["split"] == "dev"]["sentence"]])

    # ---- score the test wallets with both models ----
    test = main_s[main_s["split"] == "test"].reset_index(drop=True)
    tf = scores_from_logprobs(target_logprobs(model, encode(test["sentence"])), test["window_end"])
    ng = scores_from_logprobs([lm.token_logprobs(s.split()) for s in test["sentence"]],
                              test["window_end"])
    scores = pd.DataFrame({"wallet": test["wallet"], "label": test["label"],
                           **{f"transformer_{k}": v for k, v in tf.items()},
                           **{f"ngram_{k}": v for k, v in ng.items()}})
    scores.to_csv(DATA_DIR / "transformer_scores.tsv", sep="\t", index=False)

    full_keys = [k for k in SCORES if k != "window"]
    aucs, diffs = bootstrap_auc(
        scores["label"], {f"{m}_{k}": scores[f"{m}_{k}"] for m in ("ngram", "transformer")
                          for k in full_keys},
        pairs=[(f"transformer_{k}", f"ngram_{k}") for k in full_keys])
    w = scores[scores["ngram_window"].notna()]
    waucs, wdiffs = bootstrap_auc(w["label"], {"ngram_window": w["ngram_window"],
                                               "transformer_window": w["transformer_window"]},
                                  pairs=[("transformer_window", "ngram_window")])
    aucs |= waucs
    diffs |= wdiffs

    pd.DataFrame(history).to_csv(RESULTS_DIR / "step9_training.csv", index=False)
    pd.DataFrame([{"score": SCORES[k], "model": m, **aucs[f"{m}_{k}"]}
                  for k in SCORES for m in ("ngram", "transformer")]).to_csv(
        RESULTS_DIR / "step9_auc.csv", index=False, float_format="%.4f")
    plot_comparison(aucs, RESULTS_DIR / "step9_model_comparison.png")
    summary = {"transformer": {"parameters": n_params, "d_model": D_MODEL, "heads": HEADS,
                               "layers": LAYERS, "epochs_trained": len(history),
                               "safe_dev_perplexity": round(dev_pp, 4)},
               "ngram_safe_dev_perplexity": round(ngram_dev_pp, 4),
               "aucs": aucs, "paired_differences_transformer_minus_ngram": diffs,
               "matched_window_test_wallets": int(len(w))}
    (RESULTS_DIR / "step9_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\nsafe dev perplexity: Transformer {dev_pp:.4f} | n-gram {ngram_dev_pp:.4f}")
    print(f"\n{'score':<32} {'n-gram':>22} {'Transformer':>22} {'difference':>24}")
    for k, name in SCORES.items():
        a, b = aucs[f"ngram_{k}"], aucs[f"transformer_{k}"]
        d = diffs[f"transformer_{k} - ngram_{k}"]
        print(f"{name:<32} {a['auc']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] "
              f"{b['auc']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}] "
              f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")
    print("\nWrote data/transformer_lm.pt, data/transformer_scores.tsv, results/step9_*.csv, "
          "results/step9_model_comparison.png, results/step9_summary.json")


if __name__ == "__main__":
    main()
