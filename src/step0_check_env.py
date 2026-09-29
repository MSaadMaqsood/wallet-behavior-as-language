
import importlib.metadata as md
import io
import sys
import tempfile
from pathlib import Path

from config import CLEAN_DIR, RAW_DIR

PACKAGES = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "nltk", "torch",
            "matplotlib", "plotly", "streamlit"]


def parquet_roundtrip():
    import pandas as pd
    df = pd.DataFrame({"block": [1, 2], "event": ["SUPPLY", "BORROW"]})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.parquet"
        df.to_parquet(path)
        assert pd.read_parquet(path).equals(df)
    return "write + read"


def csv_pyarrow_engine():
    import pandas as pd
    df = pd.read_csv(io.BytesIO(b"block,log_index\n16496817,436\n"), engine="pyarrow")
    assert df.iloc[0, 0] == 16496817
    return "fast CSV reader"


def statistics():
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    p = mannwhitneyu([1, 2, 3], [4, 5, 6]).pvalue
    return f"roc_auc={auc:.2f}, mann-whitney p={p:.2f}"


def ngram_lm():
    from nltk.lm import KneserNeyInterpolated
    from nltk.lm.preprocessing import padded_everygram_pipeline
    train, vocab = padded_everygram_pipeline(2, [["SUPPLY", "BORROW", "REPAY"]])
    lm = KneserNeyInterpolated(2)
    lm.fit(train, vocab)
    return f"P(BORROW | SUPPLY) = {lm.score('BORROW', ['SUPPLY']):.2f}"


def transformer():
    import torch
    layer = torch.nn.TransformerEncoderLayer(16, 2, 32, batch_first=True)
    layer(torch.randn(2, 5, 16)).sum().backward()
    return f"forward + backward pass (torch {torch.__version__})"


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    with tempfile.TemporaryDirectory() as tmp:
        fig.savefig(Path(tmp) / "t.png")
    plt.close(fig)
    return "PNG saved"


def dashboard_libs():
    import plotly.graph_objects  # noqa: F401
    import streamlit  # noqa: F401
    return "streamlit + plotly import"


def data_folder():
    for folder in (RAW_DIR, CLEAN_DIR):
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} not found (set OLD_PROJECT_DIR)")
    n_raw = len(list(RAW_DIR.glob("raw_*.csv")))
    n_clean = len(list(CLEAN_DIR.glob("*.csv")))
    return f"{n_raw} raw + {n_clean} clean CSV files"


CHECKS = [("parquet files", parquet_roundtrip), ("CSV via pyarrow", csv_pyarrow_engine),
          ("statistics", statistics), ("n-gram LM (nltk)", ngram_lm),
          ("Transformer (torch)", transformer),
          ("figures", figure), ("dashboard", dashboard_libs),
          ("old project data", data_folder)]


def main():
    print(f"Python {sys.version.split()[0]}")
    for pkg in PACKAGES:
        try:
            print(f"  {pkg:<13} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:<13} NOT INSTALLED")

    print("\nChecks:")
    failed = 0
    for name, check in CHECKS:
        try:
            print(f"  OK    {name:<18} {check()}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name:<18} {type(e).__name__}: {e}")

    print("\nEnvironment ready." if not failed else f"\n{failed} check(s) failed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
