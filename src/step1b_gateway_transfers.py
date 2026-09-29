
import json

import pandas as pd

from chain_api import get_logs, hexint
from config import AWETH, DATA_DIR, ETH_GATEWAYS, RESULTS_DIR, ROOT

# keccak("Transfer(address,address,uint256)") - the ERC-20 Transfer event
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def pad(address):
    return "0x" + address[2:].rjust(64, "0")  # an address as a 32-byte topic


def main():
    first, last = json.loads((RESULTS_DIR / "step1_raw_summary.json").read_text())["block_range"]

    # topics: Transfer, any sender, recipient = one of the gateways
    logs = get_logs(first, last, AWETH, [TRANSFER_TOPIC, None, [pad(g) for g in ETH_GATEWAYS]])
    df = pd.DataFrame({
        "tx_hash": [l["transactionHash"].lower() for l in logs],
        "block": [hexint(l["blockNumber"]) for l in logs],
        "log_index": [hexint(l["logIndex"]) for l in logs],
        "owner": ["0x" + l["topics"][1][-40:].lower() for l in logs],
        "gateway": ["0x" + l["topics"][2][-40:].lower() for l in logs],
        "amount": [hexint(l["data"]) / 1e18 for l in logs],  # aWETH has 18 decimals
    }).sort_values(["block", "log_index"], ignore_index=True)

    DATA_DIR.mkdir(exist_ok=True)
    out = DATA_DIR / "gateway_transfers.csv"
    df.to_csv(out, index=False)
    print(df["gateway"].value_counts().rename("transfers").to_string())
    print(f"\nWrote {out.relative_to(ROOT)} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
