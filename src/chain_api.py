
import json
import time
import urllib.error
import urllib.request

RPC = "https://gateway.tenderly.co/public/mainnet"
HEADERS = {"User-Agent": "Mozilla/5.0 (research script)", "Content-Type": "application/json"}
# internal error / rate limit / busy: retry - unless the message says the answer is too big
TRANSIENT = {-32603, -32005, -32000}


class RpcError(Exception):
    pass


def hexint(s):
    return int(s, 16) if s not in ("", "0x") else 0  # nodes may write 0 as "0x"


def rpc(method, params, tries=6):
    """One JSON-RPC call. Network trouble and busy nodes are retried with a
    growing pause; any other node error is raised as RpcError."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(RPC, data=body, headers=HEADERS)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                reply = json.load(r)
            if "error" not in reply:
                return reply["result"]
            err = reply["error"]
            msg = str(err.get("message", "")).lower()
            too_big = any(w in msg for w in ("result", "range", "too many", "response size"))
            if err.get("code") not in TRANSIENT or too_big:
                raise RpcError(err)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass  # HTTP 429/5xx, timeouts, garbled replies
        if attempt < tries - 1:
            time.sleep(2 ** attempt)
    raise RpcError(f"{method}: node kept failing after {tries} tries")


def get_logs(from_block, to_block, address, topics, window=100_000, pause=0.2):
    """All logs from `address` matching `topics` in [from_block, to_block].
    The range is queried window by window; a window the node refuses (too many
    results) is split in half and tried again."""
    todo = [(lo, min(lo + window - 1, to_block)) for lo in range(from_block, to_block + 1, window)]
    todo.reverse()  # pop() from the end -> ascending block order
    logs, started, last_report = [], time.time(), 0.0
    while todo:
        lo, hi = todo.pop()
        try:
            logs += rpc("eth_getLogs", [{"address": address, "fromBlock": hex(lo),
                                         "toBlock": hex(hi), "topics": topics}])
        except RpcError:
            if lo == hi:
                raise
            mid = (lo + hi) // 2
            todo += [(mid + 1, hi), (lo, mid)]
        if time.time() - last_report > 30:  # progress line every ~30 s
            done = (hi - from_block + 1) / (to_block - from_block + 1)
            print(f"  ... {done:5.1%} of blocks, {len(logs):,} logs, "
                  f"{time.time() - started:.0f} s", flush=True)
            last_report = time.time()
        time.sleep(pause)
    return logs


def block_timestamp(block):
    """Unix timestamp of a block."""
    return hexint(rpc("eth_getBlockByNumber", [hex(block), False])["timestamp"])
