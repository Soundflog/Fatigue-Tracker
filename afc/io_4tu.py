import os, glob
import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_4tu_sessions(raw_root: str) -> pd.DataFrame:
    rows = []
    base = os.path.join(raw_root, "4tu")
    for f in glob.glob(os.path.join(base, "**", "*.mat"), recursive=True):
        try:
            mat = loadmat(f, squeeze_me=True, struct_as_record=False)
        except Exception:
            continue
        fs = float(mat.get("fs", 100))
        sid = os.path.basename(f).split("_")[0]
        sess = os.path.basename(f)
        acc = mat.get("acc"); gyr = mat.get("gyr")
        if acc is None or gyr is None:
            continue
        ax, ay, az = acc[:,0], acc[:,1], acc[:,2]
        gx, gy, gz = gyr[:,0], gyr[:,1], gyr[:,2]
        t = np.arange(len(ax))/fs
        label = np.zeros(len(ax), dtype=int)  # заполните по фазам протокола
        rows.append(pd.DataFrame({
            "sid":[sid], "sess":[sess], "fs":[fs], "domain":["4tu"], "t":[t],
            "ax":[ax], "ay":[ay], "az":[az], "gx":[gx], "gy":[gy], "gz":[gz], "label":[label]
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["sid","sess","fs","domain","t","ax","ay","az","gx","gy","gz","label"])
