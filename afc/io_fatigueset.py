import os, glob
import numpy as np
import pandas as pd

def load_fatigueset_sessions(raw_root: str) -> pd.DataFrame:
    rows = []
    base = os.path.join(raw_root, "fatigueset")
    for f in glob.glob(os.path.join(base, "**", "*.csv"), recursive=True):
        df = pd.read_csv(f)
        fs = float(df.get("fs", pd.Series([64])).iloc[0])
        sid = str(df.get("subject_id", pd.Series(["SUNK"])).iloc[0])
        sess = str(df.get("session", pd.Series(["UNK"])).iloc[0])
        bvp = df.get("bvp", pd.Series([np.nan]*len(df))).to_numpy()
        eda = df.get("eda", pd.Series([np.nan]*len(df))).to_numpy()
        temp = df.get("temp", pd.Series([np.nan]*len(df))).to_numpy()
        ax = df.get("acc_x", pd.Series([np.nan]*len(df))).to_numpy()
        ay = df.get("acc_y", pd.Series([np.nan]*len(df))).to_numpy()
        az = df.get("acc_z", pd.Series([np.nan]*len(df))).to_numpy()
        label = df.get("label", pd.Series([0]*len(df))).to_numpy()
        t = np.arange(len(bvp))/fs
        rows.append(pd.DataFrame({
            "sid":[sid], "sess":[sess], "fs":[fs], "domain":["fatigueset"], "t":[t],
            "bvp":[bvp], "eda":[eda], "temp":[temp], "ax":[ax], "ay":[ay], "az":[az], "label":[label]
        }))
    cols = ["sid","sess","fs","domain","t","bvp","eda","temp","ax","ay","az","label"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)
