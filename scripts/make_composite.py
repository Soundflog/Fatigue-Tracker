import argparse
from afc.harmonize import build_composite

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    build_composite(args.config)
