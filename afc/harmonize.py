import os, yaml, numpy as np, pandas as pd
from .io_zenodo import load_zenodo_sessions
from .io_4tu import load_4tu_sessions
from .io_fatigueset import load_fatigueset_sessions
from .io_common import sliding_windows, simple_features

def build_windows_and_features(df: pd.DataFrame, win_sec: float, hop_sec: float, min_coverage: float,
                               feature_channels):
    windows_rows, feats_rows = [], []
    for _, row in df.iterrows():
        sid, sess, domain = row["sid"], row["sess"], row["domain"]
        fs = float(row["fs"])
        channels = [c for c in feature_channels if c in row and hasattr(row[c], "__len__")]
        if not channels: 
            continue
        L = len(row[channels[0]])
        for start, end in sliding_windows(L, fs, win_sec, hop_sec):
            ys = row.get("label", None)
            if ys is None or len(ys) < end: 
                continue
            yseg = np.asarray(ys[start:end])
            if np.mean(~np.isnan(yseg)) < min_coverage: 
                continue
            vals, cnt = np.unique(yseg[~np.isnan(yseg)], return_counts=True)
            if len(vals)==0: 
                continue
            y = int(vals[np.argmax(cnt)])
            X = np.stack([np.asarray(row[c][start:end]) for c in channels], axis=1)
            windows_rows.append({"sid":sid,"sess":sess,"domain":domain,"fs":fs,"y":y,"channels":channels,"X":X})
            feat = {"sid":sid,"sess":sess,"domain":domain,"y":y}
            for ci, cname in enumerate(channels):
                f = simple_features(X[:,ci], fs)
                for k,v in f.items():
                    feat[f"{cname}_{k}"] = v
            feats_rows.append(feat)
    windows = pd.DataFrame(windows_rows)
    feats = pd.DataFrame(feats_rows)
    return windows, feats


def _save_parquet_or_csv(df: pd.DataFrame, parquet_path: str, csv_path: str):
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Saved parquet: {parquet_path}")
    except Exception as e:
        print(f"WARN: parquet failed ({e}); falling back to CSV: {csv_path}")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def build_composite(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_root = cfg["paths"]["raw_root"]
    out_root = cfg["paths"]["out_root"]
    use_sets = cfg["processing"]["use_sets"]
    win_sec = float(cfg["processing"]["win_sec"])
    hop_sec = float(cfg["processing"]["hop_sec"])
    min_cov = float(cfg["processing"]["min_coverage"])

    dfs = []
    if "zenodo" in use_sets:
        dfs.append(load_zenodo_sessions(raw_root))
    if "4tu" in use_sets:
        dfs.append(load_4tu_sessions(raw_root))
    if "fatigueset" in use_sets:
        dfs.append(load_fatigueset_sessions(raw_root))
    if not dfs:
        raise RuntimeError("Нет источников для сборки.")

    df_all = pd.concat(dfs, ignore_index=True)
    feat_channels = ["ax","ay","az","gx","gy","gz","bvp","eda","temp"]
    windows, feats = build_windows_and_features(df_all, win_sec, hop_sec, min_cov, feat_channels)

    os.makedirs(os.path.join(out_root, "windows"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "features"), exist_ok=True)
    # сохранить окна
    for idx, row in windows.iterrows():
        np.savez_compressed(os.path.join(out_root, "windows", f"win_{idx:07d}.npz"),
                            sid=row["sid"], sess=row["sess"], domain=row["domain"],
                            fs=row["fs"], y=row["y"], channels=np.array(row["channels"]), X=row["X"])
    # Метаданные окон
    meta = windows.copy()
    if "X" in meta.columns:
        meta = meta.drop(columns=["X"])
    _save_parquet_or_csv(meta,
                         os.path.join(out_root, "windows", "windows_meta.parquet"),
                         os.path.join(out_root, "windows", "windows_meta.csv"))

    # Признаки
    if feats is None or feats.empty:
        feats = pd.DataFrame(columns=["sid", "sess", "domain", "y"])
    _save_parquet_or_csv(feats,
                         os.path.join(out_root, "features", "features.parquet"),
                         os.path.join(out_root, "features", "features.csv"))

    print("Windows DF shape:", windows.shape, "Features DF shape:", feats.shape)
