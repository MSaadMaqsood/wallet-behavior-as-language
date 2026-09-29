

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from web3 import Web3

from aave_config import get_web3, ERC20_ABI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("normalize")

JUNK = {"0x0000000000000000000000000000000000000000",
        "0x000000000000000000000000000000000000dead"}

# raw file -> (output name, list of (amount_col, asset_col) to normalize)
FILES = {
    "raw_supply.csv":          ("supply.csv",       [("amount", "reserve")]),
    "raw_withdraw.csv":        ("withdraw.csv",     [("amount", "reserve")]),
    "raw_borrow.csv":          ("borrow.csv",       [("amount", "reserve")]),
    "raw_repay.csv":           ("repay.csv",        [("amount", "reserve")]),
    "raw_liquidationcall.csv": ("liquidations.csv", [("debtToCover", "debtAsset"),
                                                     ("liquidatedCollateralAmount", "collateralAsset")]),
}
ADDR_COLS = ["reserve", "user", "onBehalfOf", "to", "repayer",
             "collateralAsset", "debtAsset", "liquidator"]


# --------------------------------------------------------------------------- #
def collect_token_addresses(in_dir):
    tokens = set()
    for raw_name, (_, pairs) in FILES.items():
        p = Path(in_dir) / raw_name
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype=str)
        for _, asset_col in pairs:
            if asset_col in df.columns:
                tokens.update(df[asset_col].dropna().str.lower().unique())
    tokens.discard(None)
    tokens -= JUNK
    return sorted(tokens)


def load_cache(path):
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def fetch_token_meta(w3, addr):
    """Return (symbol, decimals) via eth_call; robust to non-standard tokens."""
    try:
        c = w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)
        decimals = int(c.functions.decimals().call())
    except Exception:
        decimals = 18  # sensible default
    try:
        symbol = c.functions.symbol().call()
        if isinstance(symbol, bytes):
            symbol = symbol.decode("utf-8", "ignore").rstrip("\x00")
    except Exception:
        symbol = addr[:8]
    return symbol, decimals


def build_token_meta(w3, tokens, cache_path):
    cache = load_cache(cache_path)
    changed = 0
    for i, addr in enumerate(tokens, 1):
        if addr in cache:
            continue
        sym, dec = fetch_token_meta(w3, addr)
        cache[addr] = {"symbol": sym, "decimals": dec}
        changed += 1
        log.info("  [%d/%d] %s -> %s (%d decimals)", i, len(tokens), addr, sym, dec)
    if changed:
        Path(cache_path).write_text(json.dumps(cache, indent=2))
    log.info("Token metadata ready: %d tokens (%d newly fetched)", len(cache), changed)
    return cache


# --------------------------------------------------------------------------- #
def normalize_file(in_dir, out_dir, raw_name, out_name, pairs, meta):
    p = Path(in_dir) / raw_name
    if not p.exists():
        log.warning("missing %s — skipping", raw_name)
        return
    df = pd.read_csv(p, dtype=str)
    n0 = len(df)

    # clean addresses
    for c in ADDR_COLS:
        if c in df.columns:
            df[c] = df[c].str.lower()

    # numeric block/log
    for c in ["block", "log_index"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalize each (amount, asset) pair
    for amount_col, asset_col in pairs:
        if amount_col not in df.columns or asset_col not in df.columns:
            continue
        dec = df[asset_col].map(lambda a: meta.get(a, {}).get("decimals", 18))
        sym = df[asset_col].map(lambda a: meta.get(a, {}).get("symbol", "?"))
        # raw integer -> real units (use float; amounts can be huge)
        raw = pd.to_numeric(df[amount_col], errors="coerce")
        df[amount_col + "_real"] = raw / (10.0 ** dec)
        df[asset_col + "_symbol"] = sym

    # drop junk-address rows on the primary owner field
    owner = next((c for c in ["onBehalfOf", "user"] if c in df.columns), None)
    if owner:
        df = df[~df[owner].isin(JUNK)]

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / out_name
    df.to_csv(out_path, index=False)
    log.info("%s: %d -> %d rows | wrote %s", raw_name, n0, len(df), out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="raw")
    ap.add_argument("--out", dest="out_dir", default="clean")
    ap.add_argument("--cache", default=None, help="token metadata cache path")
    args = ap.parse_args()

    cache_path = args.cache or str(Path(args.out_dir) / "token_meta.json")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    log.info("Collecting token addresses from raw events…")
    tokens = collect_token_addresses(args.in_dir)
    log.info("Found %d unique tokens.", len(tokens))

    w3 = get_web3()
    meta = build_token_meta(w3, tokens, cache_path)

    for raw_name, (out_name, pairs) in FILES.items():
        normalize_file(args.in_dir, args.out_dir, raw_name, out_name, pairs, meta)

    log.info("Done. Normalized CSVs in '%s/'.", args.out_dir)


if __name__ == "__main__":
    main()
