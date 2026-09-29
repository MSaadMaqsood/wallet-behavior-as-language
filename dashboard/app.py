
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import plot_style as ps  # noqa: E402  (the colors of every figure)
from config import DATA_DIR, RESULTS_DIR  # noqa: E402
from ngram_lm import EOS  # noqa: E402,F401  (the pickled model needs this module)
from step7_controls import GROUPS  # noqa: E402

st.set_page_config(page_title="Wallet Behavior as Language", layout="wide")

FONT = "Segoe UI, system-ui, -apple-system, sans-serif"
NEEDED = [DATA_DIR / "scores.tsv", DATA_DIR / "corpus.tsv", DATA_DIR / "wallets.parquet",
          DATA_DIR / "timelines.parquet", DATA_DIR / "ngram_lm.pkl",
          RESULTS_DIR / "step7_auc.csv", RESULTS_DIR / "step8_ngram_contrast.csv",
          RESULTS_DIR / "step9_auc.csv"]
SAMPLES = {"main": ("Main sample (no dust liquidations)", "sample_main", "ppl"),
           "with_dust": ("With dust liquidations", "sample_new", "ppl"),
           "old_study": ("Old study's safe wallets (35% never borrowed)", "sample_old", "ppl_old")}


# --------------------------------------------------------------------------- #
# loading (cached: every file is read once per session)
# --------------------------------------------------------------------------- #
@st.cache_data
def load_json(name):
    return json.loads((RESULTS_DIR / name).read_text())


@st.cache_data
def load_csv(name):
    return pd.read_csv(RESULTS_DIR / name)


@st.cache_data
def load_wallets():
    scores = pd.read_csv(DATA_DIR / "scores.tsv", sep="\t")
    corpus = pd.read_csv(DATA_DIR / "corpus.tsv", sep="\t",
                         usecols=["wallet", "sentence", "n_actions", "truncated"])
    info = pd.read_parquet(DATA_DIR / "wallets.parquet",
                           columns=["wallet", "first_ts", "last_ts", "first_liquidation_ts",
                                    "n_liquidations", "dust_liquidation"])
    return scores.merge(corpus, on="wallet").merge(info, on="wallet")


@st.cache_resource
def load_model():
    with open(DATA_DIR / "ngram_lm.pkl", "rb") as f:
        return pickle.load(f)["sentence"]


@st.cache_data
def load_timeline(wallet):
    return pd.read_parquet(DATA_DIR / "timelines.parquet", filters=[("wallet", "==", wallet)])


# --------------------------------------------------------------------------- #
# chart helpers
# --------------------------------------------------------------------------- #
def styled(fig, height=380, **layout):
    fig.update_layout(height=height, paper_bgcolor=ps.SURFACE, plot_bgcolor=ps.SURFACE,
                      font={"family": FONT, "color": ps.INK_2, "size": 13},
                      margin={"l": 10, "r": 10, "t": 40, "b": 10},
                      hoverlabel={"bgcolor": "white", "font_family": FONT},
                      legend={"orientation": "h", "y": 1.12, "x": 0}, **layout)
    fig.update_xaxes(gridcolor=ps.GRID, linecolor=ps.AXIS, zeroline=False)
    fig.update_yaxes(gridcolor=ps.GRID, linecolor=ps.AXIS, zeroline=False)
    return fig


def fmt_auc(a):
    return f"{a['auc']:.3f} [{a['ci_low']:.3f}–{a['ci_high']:.3f}]"


def distribution_chart(df, score):
    """Share of each class's wallets per perplexity bin (log-spaced bins)."""
    logp = np.log10(df[score])
    edges = np.linspace(logp.min(), np.percentile(logp, 99.5), 60)
    fig = go.Figure()
    for c in (0, 1):
        v = logp[df["label"] == c]
        h, _ = np.histogram(v, edges, weights=np.ones(len(v)) / len(v))
        x = 10 ** edges
        fig.add_trace(go.Scatter(
            x=x, y=np.append(h, h[-1]), mode="lines", line_shape="hv",
            line={"color": ps.CLASS_COLOR[c], "width": 2}, fill="tozeroy",
            fillcolor=ps.CLASS_COLOR[c] + "1a",  # ~10% opacity
            name=f"{ps.CLASS_NAME[c]} (n = {len(v):,}, median {10 ** v.median():.2f})",
            customdata=np.column_stack([x, np.append(x[1:], x[-1])]),
            hovertemplate="perplexity %{customdata[0]:.2f}–%{customdata[1]:.2f}<br>"
                          "%{y:.1%} of wallets<extra>%{fullData.name}</extra>"))
    fig.update_xaxes(type="log", title="Perplexity under the safe-behavior model (log scale)")
    fig.update_yaxes(title="Share of wallets", tickformat=".0%")
    return styled(fig, 400)


def forest_chart(rows, height):
    """AUC with 95% interval per row, under a bold header per group; the main
    model in ink, everything else gray. Rows must come group by group."""
    fig = go.Figure()
    y, ticks, text, previous = [], [], [], None
    for group, label in zip(rows["group"], rows["label"]):
        if group != previous:  # a header line before each group
            ticks.append(len(ticks))
            text.append(f"<b>{group}</b>")
            previous = group
        y.append(len(ticks))
        ticks.append(len(ticks))
        text.append(label)
    colors = [ps.INK if m else ps.MUTED for m in rows["main"]]
    fig.add_trace(go.Scatter(
        x=rows["auc"], y=y, mode="markers",
        marker={"size": 10, "color": colors, "line": {"color": ps.SURFACE, "width": 2}},
        error_x={"type": "data", "symmetric": False, "array": rows["ci_high"] - rows["auc"],
                 "arrayminus": rows["auc"] - rows["ci_low"], "color": ps.MUTED,
                 "thickness": 2, "width": 0},
        customdata=rows[["ci_low", "ci_high", "note"]],
        hovertemplate="AUC %{x:.3f} [%{customdata[0]:.3f}–%{customdata[1]:.3f}]"
                      "<br>%{customdata[2]}<extra></extra>", showlegend=False))
    fig.add_vline(x=0.5, line={"color": ps.AXIS, "width": 1})
    fig.add_annotation(x=0.5, y=1, yref="paper", text="chance", showarrow=False,
                       xanchor="left", font={"color": ps.MUTED, "size": 11})
    fig.update_yaxes(range=[len(ticks) - 0.4, -0.8], showgrid=False, tickmode="array",
                     tickvals=ticks, ticktext=text)  # top to bottom, headers included
    fig.update_xaxes(title="ROC-AUC on test wallets (95% bootstrap interval)")
    return styled(fig, height)


# --------------------------------------------------------------------------- #
# tabs
# --------------------------------------------------------------------------- #
def overview(s1, s6, s7, s9, wallets):
    main = wallets[wallets["sample_main"]]
    c = st.columns(4)
    c[0].metric("Wallets (main sample)", f"{len(main):,}")
    c[1].metric("Liquidated", f"{int(main['label'].sum()):,}")
    c[2].metric("Safe: borrowed, never liquidated", f"{int((main['label'] == 0).sum()):,}")
    c[3].metric("Aave v3 events", f"{s1['total_rows'] / 1e6:.1f} M")

    r6, a7, d7 = s6["results"]["main"], s7["aucs"], s7["paired_differences"]
    leak = s7["leakage_screen_top5"][0]
    st.markdown("#### What the tests show")
    st.caption("All numbers on held-out test wallets. Matched windows = exactly 4 actions per "
               "wallet ending mid-stream (liquidated: the last 4 before the liquidation; safe: "
               "4 ending at a random point), which removes the end-of-history and length effects.")
    cards = [
        ["H1", "Liquidated wallets get a higher perplexity under a model of safe behavior",
         f"Full histories: median {r6['median_liquidated']:.2f} vs {r6['median_safe']:.2f}, "
         f"Mann-Whitney p = {r6['p_value']:.1e}. Matched windows: AUC {a7['win_ngram']['auc']:.3f}.",
         "Holds on full histories; gone on matched windows"],
        ["H2", "Perplexity alone separates the classes",
         f"AUC {fmt_auc(r6)} on full histories; {a7['win_ngram']['auc']:.3f} (n-gram) and "
         f"{s9['aucs']['transformer_window']['auc']:.3f} (Transformer) on matched windows. "
         f"End token alone: {a7['ngram_eos_only']['auc']:.3f}.",
         "Signal comes from how histories end"],
        ["H3", "The order of actions carries information beyond their counts",
         f"Matched windows: classifier on 1-3 action sequences {a7['win_logreg_ngrams']['auc']:.3f}"
         f" vs counts {a7['win_logreg_counts']['auc']:.3f} "
         f"(+{d7['win_logreg_ngrams - win_logreg_counts']['diff']:.3f}). Full histories: n-gram "
         f"LM +{d7['ngram - unigram']['diff']:.3f} over the bag-of-actions LM.",
         "Supported"],
        ["H4", "The signal is not leakage",
         f"No liquidation token, nothing from the liquidation block on; strongest single "
         f"feature: {leak['feature']} (AUC {leak['auc']:.2f}). The end-of-history effect is an "
         f"artifact of how histories are cut and is removed by the matched windows.",
         "No leakage; cutoff artifact found and controlled"],
    ]
    cols = st.columns(2)
    for i, (h, hypothesis, evidence, result) in enumerate(cards):
        with cols[i % 2].container(border=True):
            st.markdown(f"**{h} · {hypothesis}**")
            st.markdown(f"Result: **{result}**")
            st.caption(evidence)


def perplexity_tab(s6, wallets):
    choice = st.radio("Sample", list(SAMPLES), horizontal=True,
                      format_func=lambda k: SAMPLES[k][0])
    _, flag, score = SAMPLES[choice]
    df = wallets[(wallets["split"] == "test") & wallets[flag]]
    r = s6["results"][choice]
    c = st.columns(4)
    c[0].metric("Median perplexity · safe", f"{r['median_safe']:.2f}")
    c[1].metric("Median perplexity · liquidated", f"{r['median_liquidated']:.2f}")
    c[2].metric("Mann-Whitney p", f"{r['p_value']:.1e}", help="one-sided: liquidated > safe")
    c[3].metric("ROC-AUC", f"{r['auc']:.3f}",
                help=f"95% bootstrap interval {r['ci_low']:.3f}–{r['ci_high']:.3f}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Perplexity of test wallets by class** (H1)")
        st.plotly_chart(distribution_chart(df, score), width="stretch")
    with right:
        st.markdown("**Perplexity as a one-number risk score** (H2)")
        fpr, tpr, _ = roc_curve(df["label"], df[score])
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                                    line={"color": ps.AXIS, "width": 1}, hoverinfo="skip"),
                         go.Scatter(x=fpr, y=tpr, mode="lines", showlegend=False,
                                    line={"color": ps.INK, "width": 2},
                                    hovertemplate="safe flagged %{x:.1%}<br>liquidated caught "
                                                  "%{y:.1%}<extra></extra>")])
        fig.update_xaxes(title="Safe wallets flagged (false-positive rate)", range=[0, 1])
        fig.update_yaxes(title="Liquidated wallets caught (true-positive rate)", range=[0, 1])
        st.plotly_chart(styled(fig, 400), width="stretch")
    st.caption("The tall spikes are single common sentences: e.g. SUPPLY_VOL BORROW_STABLE "
               "(open a leveraged position, then nothing) is 12.6% of liquidated test wallets. "
               "The Controls tab shows where this signal comes from.")
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame(s6["results"]).T, width="stretch")


def controls_tab(s9):
    auc = load_csv("step7_auc.csv").set_index("model")
    short = ["Safe-behavior LM", "Order & vocabulary", "Matched windows", "Full histories"]
    rows = []
    for (_, group), group_name in zip(GROUPS, short):
        for key, (label, order, labels) in group.items():
            a = auc.loc[key]
            rows.append({"group": group_name, "label": label, "auc": a["auc"],
                         "ci_low": a["ci_low"], "ci_high": a["ci_high"], "main": key == "ngram",
                         "note": f"uses order: {'yes' if order else 'no'} · trained with labels: "
                                 f"{'yes' if labels else 'no'}"})
    rows = pd.DataFrame(rows)
    st.markdown("**Every model and control on the same test wallets** (H3, H4)")
    st.plotly_chart(forest_chart(rows, 620), width="stretch")
    st.caption("Main model in black. 'End token only' = how surprising it is that the history "
               "stops where it stops. Matched windows remove that effect and the length effect.")

    names = {key: label.replace("full score: main model", "main model")
             + (" (windows)" if key.startswith("win_") else "")
             for _, group in GROUPS for key, (label, _, _) in group.items()}
    diffs = load_csv("step7_differences.csv")
    diffs["comparison"] = [" − ".join(names[k] for k in c.split(" - ")) for c in diffs["comparison"]]
    left, right = st.columns(2)
    with left:
        st.markdown("**Paired differences in AUC** (same bootstrap resamples)")
        st.dataframe(diffs.round(3), hide_index=True, width="stretch")
    with right:
        st.markdown("**Leakage screen**: single features, strongest first")
        screen = load_csv("step7_leakage_screen.csv").head(10)
        st.dataframe(screen[["feature", "auc", "suspicious"]].round(3), hide_index=True,
                     width="stretch")

    st.markdown("**n-gram vs Transformer language model of safe behavior**")
    t = s9["transformer"]
    st.caption(f"Transformer: {t['parameters']:,} parameters, {t['layers']} layers, "
               f"{t['epochs_trained']} epochs. Safe-dev perplexity {t['safe_dev_perplexity']:.3f} "
               f"vs n-gram {s9['ngram_safe_dev_perplexity']:.3f} (lower = better model).")
    a9 = load_csv("step9_auc.csv")
    a9["group"] = a9["score"].str.replace(" (no end token)", "", regex=False)
    a9["label"] = a9["model"].map({"ngram": "n-gram", "transformer": "Transformer"})
    a9["main"] = False
    a9["note"] = a9["score"]
    st.plotly_chart(forest_chart(a9, 360), width="stretch")


def grammar_tab():
    table = load_csv("step8_ngram_contrast.csv")
    c = st.columns([2, 2, 3])
    setting = c[0].radio("Histories", ["matched", "full"], horizontal=True,
                         format_func=lambda s: {"matched": "Matched 4-action windows",
                                                "full": "Full histories"}[s])
    lengths = c[1].multiselect("Pattern length", [2, 3], default=[2, 3])
    top = c[2].slider("Patterns per class", 5, 20, 8)
    z, sl, ss = f"z_{setting}", f"share_liquidated_{setting}", f"share_safe_{setting}"
    t = table[table["n"].isin(lengths)].dropna(subset=[z])
    rows = pd.concat([t.nlargest(top, z), t.nsmallest(top, z).iloc[::-1]])
    rows["label"] = rows["ngram"].str.replace(" ", "  →  ")
    fig = go.Figure(go.Bar(
        x=rows[z], y=rows["label"], orientation="h",
        marker={"color": [ps.LIQUIDATED if v > 0 else ps.SAFE for v in rows[z]]},
        customdata=rows[[sl, ss]],
        hovertemplate="<b>%{y}</b><br>z = %{x:.1f}<br>in %{customdata[0]:.1%} of liquidated vs "
                      "%{customdata[1]:.1%} of safe<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="z-score (weighted log-odds):  ← safe  ·  liquidated →")
    st.plotly_chart(styled(fig, 150 + 28 * len(rows), barcornerradius=4),
                    width="stretch")
    st.caption("Fightin' Words (Monroe, Colaresi & Quinn 2008): log-odds of a pattern in "
               "liquidated vs safe wallets with an informative Dirichlet prior, divided by its "
               "standard error. Each wallet counts a pattern once. |z| > 3.29 means p < 0.001.")

    st.markdown("**Patterns named in the project description** (4-action vocabulary)")
    named = load_csv("step8_named_patterns.csv")
    show = pd.DataFrame({
        "pattern": named["pattern"], "meaning": named["meaning"],
        "liquidated (windows)": named["share_liquidated_matched"].map("{:.1%}".format),
        "safe (windows)": named["share_safe_matched"].map("{:.1%}".format),
        "odds ratio, windows [95% CI]": [f"{r.odds_ratio_matched:.2f} [{r.ci_low_matched:.2f}–"
                                         f"{r.ci_high_matched:.2f}]" for r in named.itertuples()],
        "odds ratio, full histories": named["odds_ratio_full"].round(2)})
    st.dataframe(show, hide_index=True, width="stretch")


def explorer_tab(wallets, s8):
    model = load_model()
    test = wallets[(wallets["split"] == "test") & wallets["sample_main"]]
    rng = np.random.default_rng(0)
    examples = [e["wallet"] for e in s8["examples"]]
    for label in (1, 0):
        pool = test[(test["label"] == label) & test["n_actions"].between(5, 15)]
        examples += pool["wallet"].iloc[rng.choice(len(pool), 3, replace=False)].tolist()
    by_wallet = wallets.set_index("wallet")

    def describe(w):
        r = by_wallet.loc[w]
        return f"{ps.CLASS_NAME[r['label']]} · {w[:10]}…{w[-4:]} · {r['n_actions']} actions"

    c = st.columns([2, 3])
    picked = c[0].selectbox("Example wallets (test set)", examples, format_func=describe)
    typed = c[1].text_input("…or paste any wallet address", placeholder="0x…").strip().lower()
    wallet = typed or picked
    if wallet not in by_wallet.index:
        st.warning("This address is not among the analyzed wallets (it may never have borrowed "
                   "on Aave v3, or it was only a helper contract).")
        return
    r = by_wallet.loc[wallet]

    tokens = r["sentence"].split()
    lp = model.token_logprobs(tokens)
    bits = [-v / np.log(2) for v in lp]
    safe_test = test.loc[test["label"] == 0, "ppl"]
    pct = (safe_test < r["ppl"]).mean()

    c = st.columns(5)
    c[0].metric("Class", ps.CLASS_NAME[r["label"]] + (" (dust)" if r["dust_liquidation"] else ""))
    c[1].metric("Split", r["split"])
    c[2].metric("Visible actions", f"{r['n_actions']:,}" + (" (last 300 used)" if r["truncated"]
                                                             else ""))
    c[3].metric("Perplexity", f"{r['ppl']:.2f}")
    c[4].metric("Perplexity percentile", f"{pct:.0%}",
                help="Share of safe test wallets with a lower perplexity")
    if r["split"] == "train" and r["label"] == 0:
        st.info("This safe wallet was part of the language model's training data, so its "
                "perplexity is optimistic.")

    st.markdown("**The wallet's sentence and the model's surprise, token by token**")
    fig = go.Figure(go.Bar(
        x=list(range(len(bits))), y=bits, marker={"color": ps.CLASS_COLOR[r["label"]]},
        customdata=np.column_stack([tokens + [EOS], np.exp(lp)]),
        hovertemplate="%{customdata[0]}<br>%{y:.2f} bits · P = %{customdata[1]:.3f}"
                      "<extra></extra>"))
    fig.update_xaxes(tickmode="array", tickvals=list(range(len(bits))),
                     ticktext=tokens + [EOS], tickangle=-35, showgrid=False)
    fig.update_yaxes(title="surprise (bits)")
    st.plotly_chart(styled(fig, 360, barcornerradius=4), width="stretch")

    events = load_timeline(wallet)
    events = events.assign(
        visible_to_model=events["pre_cutoff"],
        transaction="https://etherscan.io/tx/" + events["tx_hash"].astype(str))
    st.markdown(f"**All {len(events):,} recorded events** "
                "(after a liquidation, events are hidden from the models)")
    st.dataframe(events[["timestamp", "event", "symbol", "asset_class", "amount",
                         "visible_to_model", "via", "transaction"]],
                 hide_index=True, width="stretch",
                 column_config={"transaction": st.column_config.LinkColumn(display_text="view"),
                                "amount": st.column_config.NumberColumn(format="%.4f")})


# --------------------------------------------------------------------------- #
def main():
    st.title("Wallet Behavior as Language")
    missing = [str(p.relative_to(ROOT)) for p in NEEDED if not p.exists()]
    if missing:
        st.error("Run Steps 1-9 first. Missing: " + ", ".join(missing))
        st.stop()

    s1, s6, s7, s8, s9 = (load_json(f"step{i}_{n}.json") for i, n in
                          ((1, "raw_summary"), (6, "summary"), (7, "summary"), (8, "summary"),
                           (9, "summary")))
    start, end = (pd.Timestamp(d.replace(" UTC", "")).strftime("%b %Y") for d in s1["dates_utc"])
    st.caption(f"Aave v3 lending wallets on Ethereum, {start} – {end}, read as sentences of "
               "actions · NLP sequence models · no prices, no health factor")
    wallets = load_wallets()
    tabs = st.tabs(["Overview", "Perplexity (H1, H2)", "Controls (H3, H4)", "Grammar",
                    "Wallet explorer"])
    with tabs[0]:
        overview(s1, s6, s7, s9, wallets)
    with tabs[1]:
        perplexity_tab(s6, wallets)
    with tabs[2]:
        controls_tab(s9)
    with tabs[3]:
        grammar_tab()
    with tabs[4]:
        explorer_tab(wallets, s8)


if __name__ == "__main__":
    main()
