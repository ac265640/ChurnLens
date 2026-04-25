"""
scripts/download_data.py
Download the Telco Customer Churn dataset to data/raw/.
Usage: python scripts/download_data.py
"""

import os
import sys
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# IBM Watson original — mirrored on GitHub, stable URL
MIRROR_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def download_via_mirror() -> None:
    print(f"Downloading from mirror:\n  {MIRROR_URL}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MIRROR_URL, OUTPUT_FILE)
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"Saved  → {OUTPUT_FILE}")
    print(f"Size   → {size_kb:.1f} KB")
    if size_kb < 900:
        print("WARNING: File size is suspiciously small — download may be incomplete.")
        sys.exit(1)


def try_kaggle() -> bool:
    """Attempt Kaggle API download. Returns True on success."""
    try:
        import kaggle  # noqa: F401
        print("Kaggle credentials found — attempting Kaggle API download...")
        ret = os.system(
            f'kaggle datasets download -d blastchar/telco-customer-churn '
            f'--unzip -p "{RAW_DIR}"'
        )
        if ret == 0 and OUTPUT_FILE.exists():
            print("Kaggle download successful.")
            return True
    except Exception as e:
        print(f"Kaggle API unavailable ({e}). Falling back to mirror.")
    return False


def main() -> None:
    if OUTPUT_FILE.exists():
        size_kb = OUTPUT_FILE.stat().st_size / 1024
        print(f"Dataset already present at {OUTPUT_FILE} ({size_kb:.1f} KB). Skipping.")
        sys.exit(0)

    if not try_kaggle():
        download_via_mirror()

    print("\nDone. Run: pytest tests/test_loader.py -v")


if __name__ == "__main__":
    main()
