import os, yaml, numpy as np, pandas as pd
import matplotlib.pyplot as plt

def summarize(csv_path: str, out_root: str, title: str):
    df = pd.read_csv(csv_path)
    agg = df.groupby("method").agg({
        "F1-macro":["mean","std"],
        "BalancedAcc":["mean","std"],
        "ROC-AUC":["mean","std"],
        "PR-AUC":["mean","std"],
        "Brier":["mean","std"],
    }).reset_index()
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.values]
    out_csv = os.path.join(out_root, "summary_metrics.csv")
    agg.to_csv(out_csv, index=False)

    for metric in ["F1-macro_mean","PR-AUC_mean","ROC-AUC_mean","Brier_mean"]:
        plt.figure(figsize=(10,5))
        x = np.arange(len(agg))
        plt.bar(x, agg[metric].values)
        plt.xticks(x, agg["method_"].values, rotation=45, ha="right")
        plt.ylabel(metric.replace("_mean",""))
        plt.title(f"{title}: {metric.replace('_mean','')}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_root, f"{metric}.png"), dpi=200)
        plt.close()

if __name__ == "__main__":
    import argparse, yaml, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    res_root = cfg["paths"]["results_root"]
    os.makedirs(res_root, exist_ok=True)
    summarize(os.path.join(res_root, "benchmarks_loso.csv"), res_root, "Benchmarks LOSO")
