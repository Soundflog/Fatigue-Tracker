import numpy as np, pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score, average_precision_score, brier_score_loss

def loso_splits(df: pd.DataFrame, id_col: str = "sid"):
    for sid in sorted(df[id_col].unique()):
        tr = df[id_col] != sid
        te = df[id_col] == sid
        yield sid, tr.values, te.values

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    m = {}
    m["F1-macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    m["BalancedAcc"] = balanced_accuracy_score(y_true, y_pred)
    try: m["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    except Exception: m["ROC-AUC"] = np.nan
    try: m["PR-AUC"] = average_precision_score(y_true, y_prob)
    except Exception: m["PR-AUC"] = np.nan
    try: m["Brier"] = brier_score_loss(y_true, y_prob)
    except Exception: m["Brier"] = np.nan
    return m
