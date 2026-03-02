import os, glob
import numpy as np
import pandas as pd

def load_zenodo_sessions(raw_root: str) -> pd.DataFrame:
    rows = []
    base = os.path.join(raw_root, "zenodo")
    for f in glob.glob(os.path.join(base, "**", "*.csv"), recursive=True):
        df = pd.read_csv(f)
        fs = float(df.get("fs", pd.Series([256])).iloc[0])
        sid = str(df.get("subject_id", pd.Series(["SUNK"])).iloc[0])
        sess = str(df.get("session", pd.Series(["UNK"])).iloc[0])
        label = df.get("label", pd.Series([0]*len(df))).to_numpy()
        ax = df.get("acc_x", pd.Series([np.nan]*len(df))).to_numpy()
        ay = df.get("acc_y", pd.Series([np.nan]*len(df))).to_numpy()
        az = df.get("acc_z", pd.Series([np.nan]*len(df))).to_numpy()
        gx = df.get("gyr_x", pd.Series([np.nan]*len(df))).to_numpy()
        gy = df.get("gyr_y", pd.Series([np.nan]*len(df))).to_numpy()
        gz = df.get("gyr_z", pd.Series([np.nan]*len(df))).to_numpy()
        t = np.arange(len(ax))/fs
        rows.append(pd.DataFrame({
            "sid":[sid], "sess":[sess], "fs":[fs], "domain":["zenodo"], "t":[t],
            "ax":[ax], "ay":[ay], "az":[az], "gx":[gx], "gy":[gy], "gz":[gz], "label":[label]
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sid","sess","fs","domain","t","ax","ay","az","gx","gy","gz","label"])
