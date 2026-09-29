
import json

import pandas as pd

from config import CLEAN_DIR, DATA_DIR, ETH_GATEWAYS, RAW_DIR, RESULTS_DIR

# event -> (clean file, owner column, caller column, asset column, amount column)
SOURCES = {
    "SUPPLY":      ("supply.csv",       "onBehalfOf", "user",    "reserve",   "amount_real"),
    "WITHDRAW":    ("withdraw.csv",     "user",       None,      "reserve",   "amount_real"),
    "BORROW":      ("borrow.csv",       "onBehalfOf", "user",    "reserve",   "amount_real"),
    "REPAY":       ("repay.csv",        "user",       "repayer", "reserve",   "amount_real"),
    "LIQUIDATION": ("liquidations.csv", "user",       None,      "debtAsset", "debtToCover_real"),
}
RAW_FILES = {"SUPPLY": "raw_supply.csv", "WITHDRAW": "raw_withdraw.csv",
             "BORROW": "raw_borrow.csv", "REPAY": "raw_repay.csv",
             "LIQUIDATION": "raw_liquidationcall.csv"}

# MKR's symbol() returns bytes32, which the old normalize.py could not read
SYMBOL_FIXES = {"0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2": "MKR"}

# STABLE = pegged to a fiat currency, including yield-bearing versions (sDAI,
# sUSDe, syrupUSDT) and Pendle principal tokens (PT-...) of USD stablecoins.
# Everything else - ETH and BTC variants, governance tokens, gold (XAUt) - is VOLATILE.
STABLECOINS = {"usdc", "usdt", "dai", "gho", "lusd", "crvusd", "pyusd", "frax", "usds",
               "rlusd", "usdg", "usdtb", "musd", "usde", "eusde", "susde", "sdai",
               "syrupusdt", "eurc"}

# A caller that executes actions for at least this many different wallets is a
# helper contract, not a user.
MIN_WALLETS_SERVED = 100

# Verified contract names from Blockscout (looked up 2026-09-28), for the report
CONTRACT_NAMES = {
    **{g: "WrappedTokenGatewayV3 (ETH gateway)" for g in ETH_GATEWAYS},
    "0xadc0a53095a0af87f3aa29fe0715b5c28016364e": "ParaSwapLiquiditySwapAdapter",
    "0x872fbcb1b582e8cd0d0dd4327fbfa0b4c2730995": "ParaSwapLiquiditySwapAdapter",
    "0x35bb522b102326ea3f1141661df4626c87000e3e": "ParaSwapRepayAdapter",
    "0x02e7b8511831b1b02d9018215a0f8f500ea5c6b3": "ParaSwapRepayAdapter",
    "0x1809f186d680f239420b56948c58f8dbbcdf1e18": "ParaSwapRepayAdapter",
    "0x8761e0370f94f68db8eaa731f4fc581f6ad0bd68": "ParaSwapDebtSwapAdapterV3GHO",
    "0xd7852e139a7097e119623de0751ae53a61efb442": "ParaSwapDebtSwapAdapterV3GHO",
    "0x8f30adaa6950b31f675bf8a709bc23f55aa24735": "ParaSwapDebtSwapAdapterV3GHO",
    "0xb748952c7bc638f31775245964707bcc5ddfabfc": "MigrationHelperMainnet",
    "0x40b72bb73b1e9380629b205e7880fa409ae7fcc9": "VaultDepositooorV2",
    "0x49d3fc00f3acf80fabcb42d7681667b20f60889a": "MulticallRouter",
}

STR = "string[pyarrow]"  # compact string storage for 4M+ addresses and hashes


def asset_class(symbol):
    s = symbol.lower()
    return "STABLE" if s in STABLECOINS or (s.startswith("pt-") and "usd" in s) else "VOLATILE"


# --------------------------------------------------------------------------- #
def load_events():
    """Read the five clean files into one common schema."""
    frames = []
    for event, (fname, owner, caller, asset, amount) in SOURCES.items():
        str_cols = [c for c in ("tx_hash", owner, caller, asset) if c]
        df = pd.read_csv(CLEAN_DIR / fname, engine="pyarrow",
                         usecols=["block", "log_index", amount] + str_cols,
                         dtype={c: STR for c in str_cols})
        frames.append(pd.DataFrame({
            "block": df["block"].astype("int64"),
            "log_index": df["log_index"].astype("int32"),
            "tx_hash": "0x" + df["tx_hash"],  # the old pull stored hashes without 0x
            "event": event,
            "wallet": df[owner],
            "caller": df[caller] if caller else pd.Series(pd.NA, index=df.index, dtype=STR),
            "asset": df[asset],
            "amount": df[amount].astype("float64"),
        }))
    ev = pd.concat(frames, ignore_index=True)
    ev["event"] = ev["event"].astype(pd.CategoricalDtype(list(SOURCES)))
    return ev


def build_asset_table(ev):
    meta = json.loads((CLEAN_DIR / "token_meta.json").read_text())
    assets = pd.DataFrame([{"asset": addr, "symbol": SYMBOL_FIXES.get(addr, m["symbol"]),
                            "decimals": m["decimals"]} for addr, m in meta.items()])
    assets["asset_class"] = assets["symbol"].map(asset_class)
    assets["events"] = assets["asset"].map(ev["asset"].value_counts()).fillna(0).astype(int)
    return assets.sort_values(["asset_class", "events"], ascending=[True, False],
                              ignore_index=True)


def find_helper_contracts(ev):
    """Callers that execute Supply/Borrow/Repay for many different wallets."""
    acted = ev.loc[(ev["caller"].notna() & (ev["caller"] != ev["wallet"])).fillna(False),
                   ["caller", "wallet"]]
    served = acted.groupby("caller")["wallet"].nunique()
    return served[served >= MIN_WALLETS_SERVED].sort_values(ascending=False)


def load_gateway_owners():
    """(tx_hash, gateway) -> the one wallet that sent aWETH to that gateway in
    that transaction, from step1b. Transactions with several senders are skipped."""
    path = DATA_DIR / "gateway_transfers.csv"
    if not path.exists():
        print("NOTE: data/gateway_transfers.csv missing - run step1b to repair "
              "ETH-gateway withdrawals")
        return {}
    t = pd.read_csv(path, usecols=["tx_hash", "gateway", "owner"]).drop_duplicates()
    t = t[~t.duplicated(["tx_hash", "gateway"], keep=False)]
    return dict(zip(zip(t["tx_hash"], t["gateway"]), t["owner"]))


def repair_contract_withdrawals(ev, contracts, gateway_owners):
    """Give withdrawals executed by a helper contract back to the real wallet."""
    by_contract = ev["wallet"].isin(contracts)
    contract_withdraw = by_contract & (ev["event"] == "WITHDRAW")

    # a) the only real wallet acting in the same transaction (the liquidated
    #    wallet does not act in the liquidator's transaction, so liquidations
    #    are left out)
    real = ev.loc[~by_contract & (ev["event"] != "LIQUIDATION"), ["tx_hash", "wallet"]]
    real = real.drop_duplicates()
    n_real = real["tx_hash"].map(real["tx_hash"].value_counts())
    single = real[n_real == 1].set_index("tx_hash")["wallet"]
    owner = ev["tx_hash"].map(single).where(contract_withdraw)
    source = pd.Series("same_tx", index=ev.index).where(owner.notna())

    # b) ETH gateways: the wallet that sent its aWETH to the gateway
    todo = contract_withdraw & owner.isna()
    keys = zip(ev.loc[todo, "tx_hash"], ev.loc[todo, "wallet"])
    owner.loc[todo] = [gateway_owners.get(k) for k in keys]
    owner = owner.where(~owner.isin(contracts))  # never hand an event to another contract
    source = source.mask(todo & owner.notna(), "atoken_transfer")

    fixed = owner.notna()
    stats = pd.DataFrame({
        "withdraws_owned": ev.loc[contract_withdraw, "wallet"].value_counts(),
        "repaired_same_tx": ev.loc[source == "same_tx", "wallet"].value_counts(),
        "repaired_atoken_transfer": ev.loc[source == "atoken_transfer", "wallet"].value_counts(),
    }).fillna(0).astype(int)
    stats["unresolved"] = stats["withdraws_owned"] - stats["repaired_same_tx"] \
        - stats["repaired_atoken_transfer"]

    # `via` keeps the helper contract that executed the action
    via = ev["caller"].where(ev["caller"].isin(contracts))
    via = via.mask(by_contract, ev["wallet"])
    ev["via"] = via.astype("category")

    ev["owner_source"] = source.where(fixed, "event")
    ev.loc[by_contract & ~fixed, "owner_source"] = "contract"
    ev["owner_source"] = ev["owner_source"].astype("category")
    ev.loc[fixed, "wallet"] = owner[fixed].to_numpy()
    return ev.drop(columns="caller"), stats


# --------------------------------------------------------------------------- #
def main():
    print("Loading clean files ...")
    ev = load_events()
    raw_rows = {e: len(pd.read_csv(RAW_DIR / f, usecols=["block"], engine="pyarrow"))
                for e, f in RAW_FILES.items()}
    clean_rows = ev["event"].value_counts().to_dict()

    # 1-3: assets
    assets = build_asset_table(ev)
    missing_meta = sorted(set(ev["asset"].unique()) - set(assets["asset"]))
    ev["symbol"] = ev["asset"].map(dict(zip(assets["asset"], assets["symbol"]))).astype("category")
    ev["asset_class"] = ev["asset"].map(dict(zip(assets["asset"], assets["asset_class"]))) \
        .astype("category")

    # 4-5: owners
    served = find_helper_contracts(ev)
    ev, repair = repair_contract_withdrawals(ev, set(served.index), load_gateway_owners())
    contracts = pd.DataFrame({"name": [CONTRACT_NAMES.get(a, "(unverified contract)")
                                       for a in served.index],
                              "wallets_served": served}).join(repair)
    contracts[repair.columns] = contracts[repair.columns].fillna(0).astype(int)
    contracts.index.name = "address"

    # 6: order + checks
    ev = ev.sort_values(["block", "log_index"], ignore_index=True)
    addr_ok = ev["wallet"].str.fullmatch(r"0x[0-9a-f]{40}").all() and \
        ev["asset"].str.fullmatch(r"0x[0-9a-f]{40}").all()
    checks = {
        "duplicate_logs": int(ev.duplicated(["block", "log_index"]).sum()),
        "addresses_valid": bool(addr_ok),
        "assets_without_metadata": missing_meta,
        "amount_missing": int(ev["amount"].isna().sum()),
        "amount_negative": int((ev["amount"] < 0).sum()),
    }

    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    ev.to_parquet(DATA_DIR / "events.parquet", index=False, compression="zstd")
    assets.to_csv(RESULTS_DIR / "step2_asset_classes.csv", index=False)
    contracts.to_csv(RESULTS_DIR / "step2_contracts.csv")

    real = ev[ev["owner_source"] != "contract"]
    top = real["wallet"].value_counts().head(5)
    summary = {
        "events": {e: int(n) for e, n in clean_rows.items()},
        "raw_vs_clean_rows": {e: [raw_rows[e], int(clean_rows[e])] for e in SOURCES},
        "wallets": int(real["wallet"].nunique()),
        "wallets_per_event": {e: int(n) for e, n in
                              real.groupby("event", observed=True)["wallet"].nunique().items()},
        "owner_source": {k: int(v) for k, v in ev["owner_source"].value_counts().items()},
        "helper_contracts": len(contracts),
        "assets": {"total": len(assets), **assets["asset_class"].value_counts().to_dict()},
        "stable_share_by_event": {e: round(float(s), 3) for e, s in
                                  (ev["asset_class"] == "STABLE").groupby(ev["event"],
                                                                         observed=True).mean().items()},
        "most_active_wallets": {w: int(n) for w, n in top.items()},
        "checks": checks,
    }
    (RESULTS_DIR / "step2_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- report ----
    print(f"\n{'event':<12} {'raw':>10} {'clean':>10} {'wallets':>9} {'stable %':>9}")
    for e in SOURCES:
        print(f"{e:<12} {raw_rows[e]:>10,} {clean_rows[e]:>10,} "
              f"{summary['wallets_per_event'][e]:>9,} "
              f"{100 * summary['stable_share_by_event'][e]:>8.1f}%")
    print(f"\nassets: {len(assets)} ({summary['assets'].get('STABLE', 0)} STABLE, "
          f"{summary['assets'].get('VOLATILE', 0)} VOLATILE) | missing metadata: {len(missing_meta)}")
    w = contracts[repair.columns].sum()
    print(f"helper contracts: {len(contracts)} | withdrawals they executed: {w.withdraws_owned:,}\n"
          f"  repaired from the same transaction: {w.repaired_same_tx:,}\n"
          f"  repaired from the aWETH transfer  : {w.repaired_atoken_transfer:,}\n"
          f"  unresolved                        : {w.unresolved:,}")
    print(f"distinct wallets: {summary['wallets']:,}")
    print(f"checks: {checks}")
    print(f"\nWrote data/events.parquet ({len(ev):,} rows), results/step2_asset_classes.csv, "
          f"results/step2_contracts.csv, results/step2_summary.json")


if __name__ == "__main__":
    main()
