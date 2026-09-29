

import os
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# RPC
# --------------------------------------------------------------------------- #
ALCHEMY_URL = os.getenv("ALCHEMY_URL")

# --------------------------------------------------------------------------- #
# Aave v3 Ethereum mainnet (verified addresses)
# --------------------------------------------------------------------------- #
POOL_ADDRESS = Web3.to_checksum_address("0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2")
DATA_PROVIDER = Web3.to_checksum_address("0x7B4EB56E7CD4b454BA8ff71E4518426369a138a3")
ORACLE = Web3.to_checksum_address("0x54586bE62E3c3580375aE3723C145253060Ca0C2")

# Pool proxy deployed ~ block 16291127 (late Jan 2023). Events start around here.
POOL_DEPLOY_BLOCK = 16291127

# --------------------------------------------------------------------------- #
# Event ABIs (only the events we pull). Indexed flags matter for decoding.
# Signatures from aave-v3-core IPool.
# --------------------------------------------------------------------------- #
EVENT_ABIS = {
    "Supply": {
        "anonymous": False,
        "name": "Supply",
        "type": "event",
        "inputs": [
            {"indexed": True,  "name": "reserve",      "type": "address"},
            {"indexed": False, "name": "user",         "type": "address"},
            {"indexed": True,  "name": "onBehalfOf",   "type": "address"},
            {"indexed": False, "name": "amount",       "type": "uint256"},
            {"indexed": True,  "name": "referralCode", "type": "uint16"},
        ],
    },
    "Withdraw": {
        "anonymous": False,
        "name": "Withdraw",
        "type": "event",
        "inputs": [
            {"indexed": True,  "name": "reserve", "type": "address"},
            {"indexed": True,  "name": "user",    "type": "address"},
            {"indexed": True,  "name": "to",      "type": "address"},
            {"indexed": False, "name": "amount",  "type": "uint256"},
        ],
    },
    "Borrow": {
        "anonymous": False,
        "name": "Borrow",
        "type": "event",
        "inputs": [
            {"indexed": True,  "name": "reserve",           "type": "address"},
            {"indexed": False, "name": "user",              "type": "address"},
            {"indexed": True,  "name": "onBehalfOf",        "type": "address"},
            {"indexed": False, "name": "amount",            "type": "uint256"},
            {"indexed": False, "name": "interestRateMode",  "type": "uint8"},
            {"indexed": False, "name": "borrowRate",        "type": "uint256"},
            {"indexed": True,  "name": "referralCode",      "type": "uint16"},
        ],
    },
    "Repay": {
        "anonymous": False,
        "name": "Repay",
        "type": "event",
        "inputs": [
            {"indexed": True,  "name": "reserve",    "type": "address"},
            {"indexed": True,  "name": "user",       "type": "address"},
            {"indexed": True,  "name": "repayer",    "type": "address"},
            {"indexed": False, "name": "amount",     "type": "uint256"},
            {"indexed": False, "name": "useATokens", "type": "bool"},
        ],
    },
    "LiquidationCall": {
        "anonymous": False,
        "name": "LiquidationCall",
        "type": "event",
        "inputs": [
            {"indexed": True,  "name": "collateralAsset",            "type": "address"},
            {"indexed": True,  "name": "debtAsset",                  "type": "address"},
            {"indexed": True,  "name": "user",                       "type": "address"},
            {"indexed": False, "name": "debtToCover",                "type": "uint256"},
            {"indexed": False, "name": "liquidatedCollateralAmount", "type": "uint256"},
            {"indexed": False, "name": "liquidator",                 "type": "address"},
            {"indexed": False, "name": "receiveAToken",              "type": "bool"},
        ],
    },
}

# minimal ERC20 ABI for decimals()/symbol() (Task 3)
ERC20_ABI = [
    {"constant": True, "inputs": [], "name": "decimals",
     "outputs": [{"name": "", "type": "uint8"}], "type": "function"},
    {"constant": True, "inputs": [], "name": "symbol",
     "outputs": [{"name": "", "type": "string"}], "type": "function"},
]


# --------------------------------------------------------------------------- #
def get_web3() -> Web3:
    if not ALCHEMY_URL:
        raise RuntimeError("ALCHEMY_URL not set. "
                           "export ALCHEMY_URL='https://eth-mainnet.g.alchemy.com/v2/<KEY>'")
    w3 = Web3(Web3.HTTPProvider(ALCHEMY_URL, request_kwargs={"timeout": 60}))
    if not w3.is_connected():
        raise RuntimeError("Web3 could not connect — check ALCHEMY_URL.")
    return w3


if __name__ == "__main__":
    w3 = get_web3()
    print("Connected:", w3.is_connected())
    print("Chain id :", w3.eth.chain_id, "(1 = Ethereum mainnet)")
    print("Latest   :", w3.eth.block_number)
    print("Pool     :", POOL_ADDRESS)
    print("Deploy blk:", POOL_DEPLOY_BLOCK)
