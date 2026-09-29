# Wallet Behavior as Language

Predicting DeFi liquidations with NLP sequence models. Each Aave v3 wallet's
history becomes a "sentence" of actions (`SUPPLY_VOL BORROW_STABLE REPAY_STABLE …`),
and a language model trained on safe wallets measures how surprising the
wallets that were later liquidated look. No prices and no health factor are used.


## Setup (Windows, Python 3.14)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\step0_check_env.py
```

The versions in `requirements.txt` are pinned. On this machine, Windows 11
*Smart App Control* blocks compiled files of some newer builds (pandas,
scipy, scikit-learn, pyarrow). `step0_check_env.py` runs every compiled code
path the project needs, so a blocked file shows up there first.


## Steps

| Step | Script | Output |
|---|---|---|
| 0 Environment check | `src/step0_check_env.py` | – |
| 1 Raw events (reused, verified) | `src/step1_raw_events.py` | `results/step1_raw_summary.json` |
| 1b ETH-gateway aWETH transfers (download) | `src/step1b_gateway_transfers.py` | `data/gateway_transfers.csv` |
| 2 Normalize → one tidy events table | `src/step2_normalize.py` | `data/events.parquet`, `results/step2_*` |
| 3 Wallet labels + timelines (true order, real timestamps) | `src/step3_wallet_timelines.py` | `data/timelines.parquet`, `data/wallets.parquet`, `results/step3_summary.json` |
| 4 Tokenization (behavior → sentences) | `src/step4_tokenize.py` | `data/corpus.tsv`, `results/step4_*` |
| 5 Train/dev/test split + n-gram language model of safe behavior | `src/step5_ngram_lm.py` (model: `src/ngram_lm.py`) | `data/splits.tsv`, `data/ngram_lm.pkl`, `results/step5_*` |
| 6 Perplexity scoring: H1, H2 | `src/step6_perplexity.py` | `data/scores.tsv`, `results/step6_*` |
| 7 Controls: order, end token, random cutoff, leakage screen: H3, H4 | `src/step7_controls.py` | `data/control_scores.tsv`, `results/step7_*` |
| 8 Grammar contrast: characteristic action patterns per class | `src/step8_grammar.py` | `results/step8_*` |
| 9 Transformer language model of safe behavior | `src/step9_transformer.py` | `data/transformer_lm.pt`, `results/step9_*` |
| 10 Dashboard (Streamlit) | `dashboard/app.py` | run: `streamlit run dashboard/app.py` |

Run every script from the project root. `data/` holds regenerated files and
is git-ignored; `results/` holds small summaries and is committed.

## Dashboard

```powershell
streamlit run dashboard/app.py
```

It has five tabs: Overview (hypotheses and results) · Perplexity (H1, H2) ·
Controls (H3, H4, n-gram vs Transformer) · Grammar (characteristic action
patterns) · Wallet explorer (any wallet's sentence, token-by-token surprise
and raw events). It reads the saved results and the files in `data/`, so run
Steps 1–9 first; it recomputes nothing except the explorer's surprise profile.


