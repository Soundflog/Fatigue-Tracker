import numpy as np
from typing import Dict

def resample_multi(channel_dict: Dict[str, np.ndarray], old_fs: float, new_fs: float) -> Dict[str, np.ndarray]:
    if abs(old_fs - new_fs) < 1e-6:
        return channel_dict
    ratio = new_fs / old_fs
    out = {}
    for k, x in channel_dict.items():
        n_new = int(np.round(len(x) * ratio))
        t_old = np.linspace(0, len(x)-1, num=len(x))
        t_new = np.linspace(0, len(x)-1, num=n_new)
        out[k] = np.interp(t_new, t_old, x)
    return out

def sliding_windows(data_len: int, fs: float, win_sec: float, hop_sec: float):
    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    for start in range(0, max(1, data_len - win + 1), hop):
        yield start, start + win

def simple_features(x: np.ndarray, fs: float):
    x = np.asarray(x).ravel()
    eps = 1e-9
    mean = float(np.mean(x))
    std = float(np.std(x) + eps)
    rms = float(np.sqrt(np.mean(x**2)))
    zc = float(((x[:-1] * x[1:]) < 0).mean()) if len(x) > 1 else 0.0
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs) if len(x)>0 else np.array([0.0])
    psd = (np.abs(X)**2) / max(len(x), 1)
    def band(a,b):
        m = (freqs >= a) & (freqs < b)
        return float(psd[m].sum() / (psd.sum() + eps))
    b1 = band(0.1, 1.5); b2 = band(1.5, 5.0); b3 = band(5.0, 15.0); b4 = band(15.0, 30.0)
    return {"mean": mean, "std": std, "rms": rms, "zc_rate": zc,
            "band_0_1p5": b1, "band_1p5_5": b2, "band_5_15": b3, "band_15_30": b4}
