
import json

import numpy as np
import pandas as pd

from chain_api import block_timestamp
from config import DATA_DIR, OLD_PROJECT, POOL_DEPLOY_BLOCK, RESULTS_DIR, SEED

SAFE_RATIO = None        # None = all never-liquidated borrowers
DUST_USD = 10            # a first liquidation covering less stablecoin debt is "dust"
ANCHOR_EVERY = 50_000    # blocks between exact timestamps (~7 days)
N_TEST_BLOCKS = 20       # random blocks to measure the interpolation error
FOLLOW_UP_DAYS = 90      # reported: safe wallets that borrowed only recently

# The old timelines kept same-block actions in file order: all supplies, then
# withdrawals, borrows, repays. Used only to measure what the true order changes.
OLD_FILE_ORDER = {"SUPPLY": 0, "WITHDRAW": 1, "BORROW": 2, "REPAY": 3, "LIQUIDATION": 4}


# --------------------------------------------------------------------------- #
def block_times(last_block):
    """Exact timestamps of anchor blocks (cached) + a few random test blocks."""
    path = DATA_DIR / "block_times.csv"
    if path.exists():
        return pd.read_csv(path)
    anchors = sorted({*range(POOL_DEPLOY_BLOCK, last_block, ANCHOR_EVERY), last_block})
    rng = np.random.default_rng(SEED)
    tests = sorted(int(b) for b in rng.integers(POOL_DEPLOY_BLOCK, last_block, N_TEST_BLOCKS))
    rows = [{"block": b, "timestamp": block_timestamp(b), "kind": kind}
            for kind, blocks in (("anchor", anchors), ("test", tests)) for b in blocks]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def interpolate_time(blocks, times):
    anchors = times[times["kind"] == "anchor"]
    return np.interp(blocks, anchors["block"], anchors["timestamp"]).round().astype("int64")


def wallets_reordered(seen):
    """Wallets whose action sequence differs between the true order and the old
    file order: some block holds an action that the old order moved earlier."""
    w, b = seen["wallet"].to_numpy(), seen["block"].to_numpy()
    rank = seen["event"].map(OLD_FILE_ORDER).to_numpy()
    same_block = (w[1:] == w[:-1]) & (b[1:] == b[:-1])
    return set(w[1:][same_block & (rank[1:] < rank[:-1])])


def length_stats(n):
    q = n.quantile([0.5, 0.9, 0.99]).round(1).to_numpy()
    return {"wallets": int(len(n)), "median": float(q[0]), "mean": round(float(n.mean()), 1),
            "p90": float(q[1]), "p99": float(q[2]), "max": int(n.max())}


# --------------------------------------------------------------------------- #
def main():
    ev = pd.read_parquet(DATA_DIR / "events.parquet")
    ev = ev[ev["owner_source"] != "contract"]  # unresolved helper-contract rows are not wallets

    # ---- labels ----
    liq_events = ev[ev["event"] == "LIQUIDATION"]
    liquidated = set(liq_events["wallet"])
    borrowers = set(ev.loc[ev["event"] == "BORROW", "wallet"])
    safe_pool = sorted(borrowers - liquidated)
    if SAFE_RATIO is None:
        safe = set(safe_pool)
    else:
        n = min(len(safe_pool), int(SAFE_RATIO * len(liquidated)))
        safe = set(np.random.default_rng(SEED).choice(safe_pool, size=n, replace=False))

    old = pd.read_csv(OLD_PROJECT / "clean_ready" / "wallet_set.csv")
    old_safe = set(old.loc[old["label"] == 0, "user"])
    assert set(old.loc[old["label"] == 1, "user"]) == liquidated, "liquidated sets differ"

    wallets = pd.DataFrame({"wallet": sorted(liquidated | safe | old_safe)})
    wallets["label"] = wallets["wallet"].isin(liquidated).astype("int8")
    wallets["sample_new"] = wallets["wallet"].isin(liquidated | safe)
    wallets["sample_old"] = wallets["wallet"].isin(liquidated | old_safe)

    # ---- timelines ----
    times = block_times(int(ev["block"].max()))
    tl = ev[ev["wallet"].isin(set(wallets["wallet"]))].copy()
    tl["timestamp"] = pd.to_datetime(interpolate_time(tl["block"].to_numpy(), times),
                                     unit="s", utc=True)
    tl["label"] = tl["wallet"].isin(liquidated).astype("int8")

    order = tl["block"] * 100_000 + tl["log_index"]  # on-chain order as one number
    first_liq = order[tl["event"] == "LIQUIDATION"].groupby(tl["wallet"]).min()
    cutoff = tl["wallet"].map(first_liq)
    tl["pre_cutoff"] = cutoff.isna() | (order < cutoff)  # never true for a liquidation row
    tl = tl.sort_values(["wallet", "block", "log_index"], ignore_index=True)
    cols = ["wallet", "label", "block", "log_index", "timestamp", "event", "asset_class",
            "symbol", "amount", "via", "owner_source", "pre_cutoff", "tx_hash"]
    tl = tl[cols]

    # ---- per-wallet numbers ----
    seen = tl[tl["pre_cutoff"]]
    g = seen.groupby("wallet")
    wallets = wallets.set_index("wallet")
    wallets["n_actions"] = g.size().reindex(wallets.index).fillna(0).astype(int)
    wallets["first_ts"] = g["timestamp"].min()
    wallets["last_ts"] = g["timestamp"].max()
    wallets["first_borrow_ts"] = seen[seen["event"] == "BORROW"].groupby("wallet")["timestamp"].min()
    liq_rows = tl[tl["event"] == "LIQUIDATION"]
    first = liq_rows.drop_duplicates("wallet").set_index("wallet")  # tl is in on-chain order
    wallets["first_liquidation_ts"] = first["timestamp"]
    wallets["n_liquidations"] = liq_rows.groupby("wallet").size() \
        .reindex(wallets.index).fillna(0).astype(int)
    wallets["first_liq_debt"] = first["amount"]
    wallets["first_liq_debt_class"] = first["asset_class"].astype(str)
    wallets["dust_liquidation"] = ((wallets["first_liq_debt_class"] == "STABLE")
                                   & (wallets["first_liq_debt"] < DUST_USD))
    stable_borrows = seen[(seen["event"] == "BORROW") & (seen["asset_class"] == "STABLE")]
    wallets["max_stable_borrow"] = stable_borrows.groupby("wallet")["amount"].max() \
        .reindex(wallets.index).fillna(0.0)
    wallets["sample_main"] = wallets["sample_new"] & ~wallets["dust_liquidation"]
    wallets = wallets.reset_index()

    # ---- checks + summary ----
    data_end = tl["timestamp"].max()
    new = wallets[wallets["sample_new"]]
    old_w = wallets[wallets["sample_old"]]
    t = times[times["kind"] == "test"]
    interp_err = np.abs(interpolate_time(t["block"].to_numpy(), times) - t["timestamp"].to_numpy())
    recent = new[(new["label"] == 0)
                 & ((data_end - new["first_borrow_ts"]).dt.days < FOLLOW_UP_DAYS)]
    reordered = wallets_reordered(seen[seen["wallet"].isin(set(new["wallet"]))])
    liq_new = new[new["label"] == 1]
    stable_liq = liq_new[liq_new["first_liq_debt_class"] == "STABLE"]
    main = wallets[wallets["sample_main"]]
    summary = {
        "wallets_main_sample": {"liquidated": int(main["label"].sum()),
                                "safe": int((main["label"] == 0).sum()),
                                "excluded_dust_liquidations": int(liq_new["dust_liquidation"].sum())},
        "wallets_new_sample": {"liquidated": int(new["label"].sum()),
                               "safe": int((new["label"] == 0).sum()),
                               "safe_pool_never_liquidated_borrowers": len(safe_pool),
                               "safe_ratio": SAFE_RATIO or "all"},
        "wallets_old_sample": {"liquidated": int(old_w["label"].sum()),
                               "safe": int((old_w["label"] == 0).sum()),
                               "safe_never_borrowed": int(((old_w["label"] == 0)
                                                           & old_w["first_borrow_ts"].isna()).sum())},
        "timeline_rows": len(tl),
        "actions_visible_to_models": int(tl["pre_cutoff"].sum()),
        "new_sample_wallets_reordered_vs_old_timelines": len(reordered),
        "data_window_utc": [str(tl["timestamp"].min()), str(data_end)],
        "sentence_length_new": {"safe": length_stats(new.loc[new["label"] == 0, "n_actions"]),
                                "liquidated": length_stats(new.loc[new["label"] == 1, "n_actions"])},
        "sentence_length_old": {"safe": length_stats(old_w.loc[old_w["label"] == 0, "n_actions"])},
        "first_liquidation_size": {
            "stablecoin_debt_share": round(len(stable_liq) / len(liq_new), 3),
            "stablecoin_debt_covered_below": {f"${c:,}": round(float((stable_liq["first_liq_debt"] < c)
                                                                     .mean()), 3)
                                              for c in (1, 10, 100, 1000)},
            f"dust_liquidations_below_${DUST_USD}": int(liq_new["dust_liquidation"].sum()),
        },
        "checks": {
            "liquidated_with_empty_history": int(((new["label"] == 1) & (new["n_actions"] == 0)).sum()),
            "liquidated_without_borrow_before": int(((new["label"] == 1)
                                                     & new["first_borrow_ts"].isna()).sum()),
            "timestamp_max_error_seconds": int(interp_err.max()),
            f"safe_first_borrow_last_{FOLLOW_UP_DAYS}_days": int(len(recent)),
        },
    }

    tl.to_parquet(DATA_DIR / "timelines.parquet", index=False, compression="zstd")
    wallets.to_parquet(DATA_DIR / "wallets.parquet", index=False)
    (RESULTS_DIR / "step3_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- report ----
    s = summary
    print(f"main sample: {s['wallets_main_sample']['liquidated']:,} liquidated + "
          f"{s['wallets_main_sample']['safe']:,} safe (without "
          f"{s['wallets_main_sample']['excluded_dust_liquidations']:,} dust liquidations)")
    print(f"new sample : {s['wallets_new_sample']['liquidated']:,} liquidated + "
          f"{s['wallets_new_sample']['safe']:,} safe (all never-liquidated borrowers: "
          f"{len(safe_pool):,})")
    print(f"old sample : {s['wallets_old_sample']['liquidated']:,} liquidated + "
          f"{s['wallets_old_sample']['safe']:,} safe, of which "
          f"{s['wallets_old_sample']['safe_never_borrowed']:,} never borrowed")
    print(f"timelines  : {len(tl):,} rows, {s['actions_visible_to_models']:,} visible to models "
          f"| window {s['data_window_utc'][0][:10]} -> {s['data_window_utc'][1][:10]}")
    print(f"true order : {len(reordered):,} of {len(new):,} wallets "
          f"({len(reordered) / len(new):.1%}) read differently than in the old timelines")
    print("\nactions per wallet (what a model sees)")
    for name, st in [("new safe", s["sentence_length_new"]["safe"]),
                     ("new liquidated", s["sentence_length_new"]["liquidated"]),
                     ("old safe", s["sentence_length_old"]["safe"])]:
        print(f"  {name:<15} median {st['median']:>5.0f}  mean {st['mean']:>6}  "
              f"p90 {st['p90']:>5.0f}  p99 {st['p99']:>6.0f}  max {st['max']:>6,}")
    fl = s["first_liquidation_size"]
    print(f"\nfirst liquidation, stablecoin debt covered (share of {len(stable_liq):,}): "
          + ", ".join(f"<{k} {v:.1%}" for k, v in fl["stablecoin_debt_covered_below"].items()))
    print(f"dust liquidations (< ${DUST_USD}): {fl[f'dust_liquidations_below_${DUST_USD}']:,} "
          f"of {len(liq_new):,} liquidated wallets")
    print(f"\nchecks: {s['checks']}")
    print("\nWrote data/timelines.parquet, data/wallets.parquet, results/step3_summary.json")


if __name__ == "__main__":
    main()
