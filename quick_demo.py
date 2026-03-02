import os, csv, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, brier_score_loss
import matplotlib.pyplot as plt

# --- 1) Чтение ваших CSV без заголовков (игнор первых 2 токенов в каждой строке)
def read_series_csv(path):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            # отбрасываем первые 2 поля ('4','F') и переводим остальные в float
            toks = row[2:]
            for t in toks:
                try:
                    vals.append(float(t))
                except:
                    pass
    return np.asarray(vals, dtype=float)

# --- 2) Ресэмплинг к единой длине (на случай, если длины рядов различаются)
def resample_to_length(x, new_len):
    if len(x) == 0:
        return np.zeros(new_len, dtype=float)
    old_idx = np.linspace(0, 1, num=len(x))
    new_idx = np.linspace(0, 1, num=new_len)
    return np.interp(new_idx, old_idx, x)

# --- 3) Оконование и простые признаки
def sliding(start, stop, win, hop):
    i = start
    while i + win <= stop:
        yield i, i + win
        i += hop

def band_energy(x, fs, a, b):
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0/fs)
    psd = (np.abs(X)**2) / max(len(x), 1)
    m = (freqs >= a) & (freqs < b)
    s = psd[m].sum()
    return float(s / (psd.sum() + 1e-9))

def extract_features(acc_win, ibi_win, fs):
    x = acc_win
    feats = {}
    feats["acc_mean"] = float(np.mean(x))
    feats["acc_std"]  = float(np.std(x) + 1e-9)
    feats["acc_rms"]  = float(np.sqrt(np.mean(x**2)))
    feats["acc_zc"]   = float(((x[:-1]*x[1:]) < 0).mean()) if len(x) > 1 else 0.0
    feats["acc_b01"]  = band_energy(x, fs, 0.1, 1.5)
    feats["acc_b15"]  = band_energy(x, fs, 1.5, 5.0)
    feats["acc_b5"]   = band_energy(x, fs, 5.0, 15.0)
    feats["acc_b15_30"] = band_energy(x, fs, 15.0, 30.0)
    ibi = ibi_win
    feats["hr_mean"]  = float(60000.0 / (np.mean(ibi) + 1e-6))  # формула для bpm из IBI (мс)
    feats["ibi_sdnn"] = float(np.std(ibi) + 1e-9)
    feats["ibi_rmssd"]= float(np.sqrt(np.mean(np.diff(ibi)**2)) if len(ibi) > 1 else 0.0)
    return feats

# --- 4) FILES
acc_path = r"data\raw\zenodo\Accel_mag_all.csv"
ibi_path = r"data\raw\fatigueset\01\chest_bb_interval.csv"

acc_raw = read_series_csv(acc_path)
ibi_raw = read_series_csv(ibi_path)
# нормируем длину, например до 40 000 отсчётов
T = 40000
acc = resample_to_length(acc_raw, T)
ibi = resample_to_length(ibi_raw, T)

# --- 5)  первые 60% времени — класс 0, последние 40% — класс 1
y_full = np.zeros(T, dtype=int)
y_full[int(0.60*T):] = 1

# --- 6) Формируем окна и табличные признаки
# опорная частота для окон/спектра
fs = 50.0
win_sec = 3.0
hop_sec = 1.0
win = int(round(win_sec*fs))
hop = int(round(hop_sec*fs))

rows = []
for s, e in sliding(0, T, win, hop):
    feats = extract_features(acc[s:e], ibi[s:e], fs)
    feats["y"] = int(round(np.mean(y_full[s:e])))
    rows.append(feats)

df = pd.DataFrame(rows)
X = df.drop(columns=["y"]).values
y = df["y"].values

# --- 7) Три простых модели и k-fold (без LOSO, т.к. субъект по сути один)
def metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    out = {
        "F1-macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "BalancedAcc": balanced_accuracy_score(y_true, y_pred)
    }
    try: out["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    except: out["ROC-AUC"] = np.nan
    try: out["PR-AUC"] = average_precision_score(y_true, y_prob)
    except: out["PR-AUC"] = np.nan
    try: out["Brier"] = brier_score_loss(y_true, y_prob)
    except: out["Brier"] = np.nan
    return out

def run_model(name, fit_fn, pred_fn, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=77)
    rowz = []
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        p = pred_fn(fit_fn(Xtr, ytr), Xte)
        m = metrics(yte, p); m.update({"method":name, "split":f"KFold_{i}"})
        rowz.append(m)
    return pd.DataFrame(rowz)

# Модели
scaler = StandardScaler()
def fit_lr(Xtr, ytr):
    Xn = scaler.fit_transform(Xtr)
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xn, ytr)
    return (scaler, lr)
def pred_lr(model, Xte):
    sc, lr = model
    return lr.predict_proba(sc.transform(Xte))[:,1]

def fit_rf(Xtr, ytr):
    rf = RandomForestClassifier(n_estimators=300, random_state=5332, n_jobs=-1)
    rf.fit(Xtr, ytr); return rf
def pred_rf(model, Xte):
    return model.predict_proba(Xte)[:,1]

def fit_gbdt(Xtr, ytr):
    gb = GradientBoostingClassifier(random_state=2342)
    gb.fit(Xtr, ytr); return gb
def pred_gbdt(model, Xte):
    return model.predict_proba(Xte)[:,1]

res = []
res.append(run_model("LR",   fit_lr,   pred_lr,   X, y))
res.append(run_model("RF",   fit_rf,   pred_rf,   X, y))
res.append(run_model("GBDT", fit_gbdt, pred_gbdt, X, y))
res = pd.concat(res, ignore_index=True)

summary = res.groupby("method").agg(
    F1_macro_mean=("F1-macro","mean"), F1_macro_std=("F1-macro","std"),
    BalancedAcc_mean=("BalancedAcc","mean"),
    ROC_AUC_mean=("ROC-AUC","mean"),
    PR_AUC_mean=("PR-AUC","mean"),
    Brier_mean=("Brier","mean")
).reset_index()

os.makedirs("results_demo", exist_ok=True)
df.to_csv("results_demo/demo_features.csv", index=False, encoding="utf-8-sig")
res.to_csv("results_demo/demo_results_splits.csv", index=False, encoding="utf-8-sig")
summary.to_csv("results_demo/demo_results_summary.csv", index=False, encoding="utf-8-sig")

# --- 8) Пара графиков для отчёта
plt.figure(figsize=(7,4))
plt.bar(range(len(summary)), summary["F1_macro_mean"].values)
plt.xticks(range(len(summary)), summary["method"].values)
plt.ylabel("F1-macro (mean)"); plt.title("Демо: F1-macro"); plt.tight_layout()
plt.savefig("results_demo/demo_f1_macro.png", dpi=180)

plt.figure(figsize=(7,4))
plt.bar(range(len(summary)), summary["PR_AUC_mean"].values)
plt.xticks(range(len(summary)), summary["method"].values)
plt.ylabel("PR-AUC (mean)"); plt.title("Демо: PR-AUC"); plt.tight_layout()
plt.savefig("results_demo/demo_prauc.png", dpi=180)

print("OK. Файлы сохранены в папке results_demo/")
