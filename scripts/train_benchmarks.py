import os, yaml, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from afc.splits_metrics import compute_metrics
from afc.models_tabular import fit_lr, predict_lr, fit_rf, predict_rf

def load_features(cfg):
    feats_parquet = os.path.join(cfg["paths"]["out_root"], "features", "features.parquet")
    feats_csv     = os.path.join(cfg["paths"]["out_root"], "features", "features.csv")
    if os.path.exists(feats_parquet):
        df = pd.read_parquet(feats_parquet)
    elif os.path.exists(feats_csv):
        df = pd.read_csv(feats_csv)
    else:
        raise FileNotFoundError("Не найден features.parquet или features.csv — выполните make_composite.")
    return df

def ensure_xy(df):
    y = df["y"].astype(int).values
    feat_cols = [c for c in df.columns if c not in ["y","sid","sess","domain"]]
    X = df[feat_cols].fillna(0.0).values
    return X, y, feat_cols

def train_eval_split(name, Xtr, ytr, Xte, yte, rows):
    # защита: в обучении и тесте должны быть оба класса
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return rows  # пропускаем такой сплит (иначе упадут модели/метрики)
    # LR
    lr = fit_lr(Xtr, ytr)
    p_lr = predict_lr(lr, Xte)
    m_lr = compute_metrics(yte, p_lr); m_lr.update({"method":"Bench: Logistic Regression","split":name}); rows.append(m_lr)
    # RF
    rf = fit_rf(Xtr, ytr)
    p_rf = predict_rf(rf, Xte)
    m_rf = compute_metrics(yte, p_rf); m_rf.update({"method":"Bench: Random Forest","split":name}); rows.append(m_rf)
    return rows

def loso_protocol(df, rows):
    # LOSO по субъектам, если субъектов >= 2
    subjects = df["sid"].astype(str).unique().tolist()
    for sid in subjects:
        te_mask = (df["sid"].astype(str) == str(sid)).values
        tr_mask = ~te_mask
        X, y, _ = ensure_xy(df)
        Xtr, ytr = X[tr_mask], y[tr_mask]
        Xte, yte = X[te_mask], y[te_mask]
        if Xtr.shape[0]==0 or Xte.shape[0]==0:
            continue
        rows = train_eval_split(f"LOSO_sid={sid}", Xtr, ytr, Xte, yte, rows)
    return rows

def kfold_protocol(df, rows, n_splits=5):
    # Fallback: стратифицированный KFold внутри одного субъекта/малого набора
    X, y, _ = ensure_xy(df)
    if len(np.unique(y)) < 2 or len(y) < 10:
        raise RuntimeError("Недостаточно данных/классов для KFold. Проверьте метки и сборку.")
    skf = StratifiedKFold(n_splits=min(n_splits, np.unique(y, return_counts=True)[1].min(), 5), shuffle=True, random_state=42)
    for i, (tr, te) in enumerate(skf.split(X, y), 1):
        rows = train_eval_split(f"KFold_{i}", X[tr], y[tr], X[te], y[te], rows)
    return rows

def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["paths"]["results_root"], exist_ok=True)

    df = load_features(cfg)
    # Санити-чек
    df["sid"] = df["sid"].astype(str)
    print("Rows:", len(df), "Subjects:", df["sid"].nunique(), "Label dist:", df["y"].value_counts().to_dict())

    rows = []
    if df["sid"].nunique() >= 2:
        rows = loso_protocol(df, rows)
    else:
        print("WARN: обнаружен один субъект. Переключаюсь на Stratified KFold.")
        rows = kfold_protocol(df, rows, n_splits=1)

    if not rows:
        raise RuntimeError("Нет валидных разбиений для обучения/оценки. Проверьте метки и распределение.")

    out_csv = os.path.join(cfg["paths"]["results_root"], "benchmarks_loso.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
