
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The Aave data was downloaded by our earlier Business Analytics project. We
# reuse its files read-only. Set OLD_PROJECT_DIR if that folder lives elsewhere.
OLD_PROJECT = Path(os.environ.get(
    "OLD_PROJECT_DIR",
    ROOT / "On-Chain DeFi Liquidation-Risk — Behavioral Archetypes & Early Prediction",
))
RAW_DIR = OLD_PROJECT / "raw"      # decoded Pool events (Step 1 input)
CLEAN_DIR = OLD_PROJECT / "clean"  # normalized events + token_meta.json (Step 2 input)

DATA_DIR = ROOT / "data"        # generated data, git-ignored
RESULTS_DIR = ROOT / "results"  # summaries, tables and figures, committed

SEED = 42

# Aave v3 Ethereum, Core market
POOL_ADDRESS = "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2"
POOL_DEPLOY_BLOCK = 16_291_127

# Aave's ETH gateways (WrappedTokenGatewayV3). They withdraw ETH in their own
# name after receiving the wallet's aWETH; see src/step1b_gateway_transfers.py.
AWETH = "0x4d5f47fa6a74757f35c14fd3a6ef8e3c9bc514e8"
ETH_GATEWAYS = [
    "0xd322a49006fc828f9b5b37ab215f99b4e5cab19c",
    "0xd01607c3c5ecaba394d8be377a08590149325722",
    "0x893411580e590d62ddbca8a703d61cc4a8c7b2b9",
    "0xa434d495249abe33e031fe71a969b81f3c07950d",
]
