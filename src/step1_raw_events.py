
import argparse
import json
import urllib.error
from datetime import datetime, timezone

import pandas as pd

from chain_api import RpcError, block_timestamp, get_logs
from config import POOL_ADDRESS, POOL_DEPLOY_BLOCK, RAW_DIR, RESULTS_DIR, ROOT

# event -> (file, columns written by pipeline/pull_events.py)
RAW_FILES = {
    "Supply": ("raw_supply.csv",
               ["block", "tx_hash", "log_index", "reserve", "user", "onBehalfOf", "amount"]),
    "Withdraw": ("raw_withdraw.csv",
                 ["block", "tx_hash", "log_index", "reserve", "user", "to", "amount"]),
    "Borrow": ("raw_borrow.csv",
               ["block", "tx_hash", "log_index", "reserve", "user", "onBehalfOf", "amount",
                "interestRateMode", "borrowRate"]),
    "Repay": ("raw_repay.csv",
              ["block", "tx_hash", "log_index", "reserve", "user", "repayer", "amount",
               "useATokens"]),
    "LiquidationCall": ("raw_liquidationcall.csv",
                        ["block", "tx_hash", "log_index", "collateralAsset", "debtAsset", "user",
                         "debtToCover", "liquidatedCollateralAmount", "liquidator",
                         "receiveAToken"]),
}
KEY_COLS = ["block", "tx_hash", "log_index"]

# keccak("LiquidationCall(address,address,address,uint256,uint256,address,bool)")
LIQUIDATION_TOPIC = "0xe413a321e8681d831f4dbccbca790d2952b56f977908e45be37335533e005286"


def utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="skip the public-node cross-check")
    args = ap.parse_args()

    # relative path only: this summary is committed to a public repository
    source = RAW_DIR.relative_to(ROOT).as_posix() if RAW_DIR.is_relative_to(ROOT) else RAW_DIR.name
    summary, keys = {"source": source, "events": {}}, []
    print(f"Raw files: {source}\n")
    print(f"{'event':<16} {'rows':>10} {'first block':>12} {'last block':>12} {'missing':>8}")
    for event, (fname, columns) in RAW_FILES.items():
        path = RAW_DIR / fname
        header = list(pd.read_csv(path, nrows=0).columns)
        assert header == columns, f"{fname}: unexpected columns {header}"

        df = pd.read_csv(path, usecols=KEY_COLS, dtype={"tx_hash": str}, engine="pyarrow")
        missing = int(df[KEY_COLS].isna().sum().sum())
        keys.append(df[["block", "log_index"]])
        summary["events"][event] = {
            "file": fname, "rows": len(df), "missing_key_values": missing,
            "first_block": int(df.block.min()), "last_block": int(df.block.max()),
            "transactions": int(df.tx_hash.nunique()),
        }
        s = summary["events"][event]
        print(f"{event:<16} {s['rows']:>10,} {s['first_block']:>12,} {s['last_block']:>12,} {missing:>8}")

    all_keys = pd.concat(keys, ignore_index=True)
    first, last = int(all_keys.block.min()), int(all_keys.block.max())
    summary["total_rows"] = len(all_keys)
    summary["block_range"] = [first, last]
    summary["checks"] = {
        "duplicate_logs": int(all_keys.duplicated().sum()),
        "blocks_before_pool_deploy": int((all_keys.block < POOL_DEPLOY_BLOCK).sum()),
        "missing_key_values": sum(e["missing_key_values"] for e in summary["events"].values()),
    }
    print(f"{'TOTAL':<16} {len(all_keys):>10,} {first:>12,} {last:>12,}")
    print(f"\nduplicate logs (block, log_index): {summary['checks']['duplicate_logs']}")
    print(f"blocks before Pool deployment    : {summary['checks']['blocks_before_pool_deploy']}")

    if not args.offline:
        print("\nCross-check with a public Ethereum node ...")
        try:
            summary["dates_utc"] = [utc(block_timestamp(first)), utc(block_timestamp(last))]
            print(f"data window     : {summary['dates_utc'][0]}  ->  {summary['dates_utc'][1]}")
            n_liq = len(get_logs(first, last, POOL_ADDRESS, [LIQUIDATION_TOPIC], window=500_000))
            ours = summary["events"]["LiquidationCall"]["rows"]
            summary["checks"]["liquidations_public_node"] = n_liq
            summary["checks"]["liquidations_match"] = n_liq == ours
            print(f"LiquidationCall : ours {ours:,} | public node {n_liq:,} | "
                  f"{'MATCH' if n_liq == ours else 'MISMATCH'}")
        except (RpcError, urllib.error.URLError, TimeoutError) as e:
            summary["checks"]["liquidations_public_node"] = f"skipped: {e}"
            print(f"skipped ({e}) - run the script again later")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "step1_raw_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
