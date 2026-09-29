

from __future__ import annotations
import argparse
import csv
import logging
import time
from pathlib import Path

from web3._utils.events import get_event_data
from eth_utils import event_abi_to_log_topic

from aave_config import get_web3, POOL_ADDRESS, POOL_DEPLOY_BLOCK, EVENT_ABIS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pull")

CSV_FIELDS = {
    "Supply":   ["block", "tx_hash", "log_index", "reserve", "user", "onBehalfOf", "amount"],
    "Withdraw": ["block", "tx_hash", "log_index", "reserve", "user", "to", "amount"],
    "Borrow":   ["block", "tx_hash", "log_index", "reserve", "user", "onBehalfOf",
                 "amount", "interestRateMode", "borrowRate"],
    "Repay":    ["block", "tx_hash", "log_index", "reserve", "user", "repayer",
                 "amount", "useATokens"],
    "LiquidationCall": ["block", "tx_hash", "log_index", "collateralAsset", "debtAsset",
                        "user", "debtToCover", "liquidatedCollateralAmount",
                        "liquidator", "receiveAToken"],
}


class _RangeTooBig(Exception):
    pass


def build_topic_map(w3):
    m = {}
    for name, abi in EVENT_ABIS.items():
        topic0 = "0x" + event_abi_to_log_topic(abi).hex()
        m[name] = (topic0, abi)
    return m


def row_from_event(name, block, tx_hash, log_index, args):
    base = {"block": block, "tx_hash": tx_hash, "log_index": log_index}
    for f in CSV_FIELDS[name]:
        if f in base:
            continue
        v = args.get(f)
        if isinstance(v, (bytes, bytearray)):
            v = "0x" + v.hex()
        base[f] = v
    return base


def get_logs_with_retry(w3, params, tries=5):
    delay = 1.0
    for attempt in range(tries):
        try:
            return w3.eth.get_logs(params)
        except Exception as e:
            msg = str(e).lower()
            if ("10 block range" in msg or "block range should work" in msg
                    or "up to a 10 block" in msg):
                # free-tier hard cap — signal caller to shrink to 10
                raise _RangeTooBig()
            if ("more than" in msg or "response size" in msg
                    or "limit" in msg or "range" in msg or "too large" in msg):
                raise _RangeTooBig()
            log.warning("get_logs error (attempt %d): %s", attempt + 1, str(e)[:140])
            time.sleep(delay); delay *= 2
    raise RuntimeError("get_logs failed after retries")


def pull_range(w3, topic_map, start, end, chunk, out_dir, sleep_s):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    writers, files, counts = {}, {}, {name: 0 for name in EVENT_ABIS}
    for name in EVENT_ABIS:
        fh = open(out / f"raw_{name.lower()}.csv", "w", newline="")
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS[name]); w.writeheader()
        writers[name], files[name] = w, fh

    name_by_topic0 = {t: name for name, (t, _) in topic_map.items()}

    total_blocks = end - start + 1
    lo = start
    last_report = time.time()
    while lo <= end:
        hi = min(lo + chunk - 1, end)
        # NOTE: no topic filter — pull ALL Pool events in range, classify in Python.
        params = {"fromBlock": lo, "toBlock": hi, "address": POOL_ADDRESS}
        try:
            logs = get_logs_with_retry(w3, params)
        except _RangeTooBig:
            new_chunk = max(1, (hi - lo + 1) // 2)
            if new_chunk == chunk:
                new_chunk = max(1, chunk - 1)
            log.info("Range %d-%d rejected; shrinking chunk %d -> %d",
                     lo, hi, chunk, new_chunk)
            chunk = new_chunk
            continue

        for raw in logs:
            t0 = raw["topics"][0]
            t0 = "0x" + t0.hex() if isinstance(t0, (bytes, bytearray)) else t0
            name = name_by_topic0.get(t0)
            if name is None:
                continue
            _, abi = topic_map[name]
            try:
                ev = get_event_data(w3.codec, abi, raw)
            except Exception as e:
                log.warning("decode failed (%s) blk %s: %s", name, raw["blockNumber"], str(e)[:80])
                continue
            txh = raw["transactionHash"]
            txh = txh.hex() if hasattr(txh, "hex") else txh
            writers[name].writerow(
                row_from_event(name, raw["blockNumber"], txh, raw["logIndex"], dict(ev["args"])))
            counts[name] += 1

        # progress every ~5s
        if time.time() - last_report > 5:
            done = hi - start + 1
            log.info("…%d/%d blocks (%.1f%%) | %s", done, total_blocks,
                     100 * done / total_blocks,
                     ", ".join(f"{k}={v}" for k, v in counts.items()))
            last_report = time.time()

        lo = hi + 1
        if sleep_s:
            time.sleep(sleep_s)

    for fh in files.values():
        fh.close()
    log.info("FINISHED. Totals: %s", counts)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="start block (default: deploy block)")
    ap.add_argument("--end", default="latest")
    ap.add_argument("--last", type=int, default=None,
                    help="convenience: pull only the last N blocks (overrides --start)")
    ap.add_argument("--chunk", type=int, default=10,
                    help="block range per request (free tier max = 10)")
    ap.add_argument("--out", default="raw")
    ap.add_argument("--sleep", type=float, default=0.05,
                    help="seconds to sleep between calls (raise if rate-limited)")
    args = ap.parse_args()

    w3 = get_web3()
    latest = w3.eth.block_number
    log.info("Connected. chain=%d latest=%d", w3.eth.chain_id, latest)

    end = latest if args.end == "latest" else int(args.end)
    if args.last is not None:
        start = end - args.last + 1
    elif args.start is not None:
        start = int(args.start)
    else:
        start = POOL_DEPLOY_BLOCK

    n = end - start + 1
    est_calls = (n + args.chunk - 1) // args.chunk
    log.info("Range %d -> %d (%d blocks) | chunk=%d | ~%d requests | out='%s'",
             start, end, n, args.chunk, est_calls, args.out)
    if est_calls > 20000:
        log.warning("~%d requests is a LOT for the free tier. Consider --last "
                    "for a smaller test, or PAYG for the full pull.", est_calls)

    topic_map = build_topic_map(w3)
    pull_range(w3, topic_map, start, end, args.chunk, args.out, args.sleep)


if __name__ == "__main__":
    main()