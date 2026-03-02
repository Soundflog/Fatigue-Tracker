"""
Скрипт обучения 1D CNN для детекции утомления спортсменов.

Использование:
    python scripts/train_deep.py --config config.yaml of
    
Требования:
    - PyTorch >= 1.12.0
    - Собранные окна в data/processed/windows/
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Проверка наличия PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARN: PyTorch не установлен. Установите: pip install torch")

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if TORCH_AVAILABLE:
    from afc.models_cnn import (
        FatigueCNN1D, 
        FatigueDataset, 
        FatigueTrainer,
        compute_class_weights,
        save_model,
        personalize_model
    )
    from afc.splits_metrics import compute_metrics


def load_windows_meta(cfg: dict) -> pd.DataFrame:
    """Загрузка метаданных окон."""
    meta_parquet = os.path.join(cfg["paths"]["out_root"], "windows", "windows_meta.parquet")
    meta_csv = os.path.join(cfg["paths"]["out_root"], "windows", "windows_meta.csv")
    
    if os.path.exists(meta_parquet):
        return pd.read_parquet(meta_parquet)
    elif os.path.exists(meta_csv):
        return pd.read_csv(meta_csv)
    else:
        raise FileNotFoundError("Не найден windows_meta — выполните make_composite.py")


def load_all_windows(cfg: dict, meta: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Загрузка всех окон в память.
    
    Returns:
        X: массив данных (N, T, C)
        y: метки (N,)
        meta: метаданные
    """
    windows_dir = os.path.join(cfg["paths"]["out_root"], "windows")
    
    X_list, y_list = [], []
    valid_indices = []
    
    for idx, row in meta.iterrows():
        fname = f"win_{idx:07d}.npz"
        fpath = os.path.join(windows_dir, fname)
        
        if os.path.exists(fpath):
            data = np.load(fpath, allow_pickle=True)
            X_list.append(data['X'])
            y_list.append(int(data['y']))
            valid_indices.append(idx)
    
    if not X_list:
        raise RuntimeError("Нет валидных окон для загрузки")
    
    # Определение общего формата
    shapes = [x.shape for x in X_list]
    max_len = max(s[0] for s in shapes)
    n_channels = shapes[0][1]
    
    # Паддинг до максимальной длины
    X = np.zeros((len(X_list), max_len, n_channels), dtype=np.float32)
    for i, x in enumerate(X_list):
        X[i, :x.shape[0], :] = x
    
    y = np.array(y_list, dtype=np.float32)
    meta_valid = meta.loc[valid_indices].reset_index(drop=True)
    
    print(f"Загружено окон: {len(X)}, shape: {X.shape}")
    print(f"Распределение меток: {np.bincount(y.astype(int))}")
    
    return X, y, meta_valid


def loso_train_eval(X: np.ndarray, 
                    y: np.ndarray, 
                    meta: pd.DataFrame,
                    cfg: dict,
                    device: str) -> List[Dict]:
    """
    Leave-One-Subject-Out обучение и валидация.
    
    Returns:
        results: список словарей с метриками для каждого субъекта
    """
    results = []
    subjects = meta['sid'].unique().tolist()
    
    print(f"\nLOSO: {len(subjects)} субъектов")
    print("=" * 60)
    
    for i, test_sid in enumerate(subjects):
        print(f"\n[{i+1}/{len(subjects)}] Test subject: {test_sid}")
        
        # Разбиение
        test_mask = (meta['sid'] == test_sid).values
        train_mask = ~test_mask
        
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # Проверка на наличие обоих классов
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  Пропуск: недостаточно классов (train: {np.unique(y_train)}, test: {np.unique(y_test)})")
            continue
        
        # Нормализация по train
        mean = X_train.mean(axis=(0, 1), keepdims=True)
        std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std
        
        # Datasets & Loaders
        train_dataset = FatigueDataset(X=X_train_norm, y=y_train)
        test_dataset = FatigueDataset(X=X_test_norm, y=y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Модель
        in_channels = X.shape[2]
        model = FatigueCNN1D(in_channels=in_channels)
        
        # Веса классов
        class_weights = compute_class_weights(y_train)
        
        # Trainer
        trainer = FatigueTrainer(
            model=model,
            device=device,
            learning_rate=0.001,
            class_weights=class_weights
        )
        
        # Обучение
        trainer.fit(
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=50,
            early_stopping_patience=10,
            verbose=False
        )
        
        # Финальная оценка
        _, metrics = trainer.validate(test_loader)
        metrics['subject'] = test_sid
        metrics['n_train'] = len(y_train)
        metrics['n_test'] = len(y_test)
        results.append(metrics)
        
        print(f"  F1-macro: {metrics['f1_macro']:.4f} | "
              f"ROC-AUC: {metrics['roc_auc']:.4f} | "
              f"Balanced Acc: {metrics['balanced_acc']:.4f}")
    
    return results


def cross_dataset_train_eval(X: np.ndarray,
                              y: np.ndarray,
                              meta: pd.DataFrame,
                              cfg: dict,
                              device: str) -> Dict:
    """
    Кросс-датасетная валидация.
    
    Train на одних датасетах, test на другом.
    """
    train_domains = cfg.get("splits", {}).get("cross_dataset", {}).get("train", [])
    test_domains = cfg.get("splits", {}).get("cross_dataset", {}).get("test", [])
    
    if not train_domains or not test_domains:
        return {}
    
    print(f"\nCross-Dataset: train={train_domains}, test={test_domains}")
    
    train_mask = meta['domain'].isin(train_domains).values
    test_mask = meta['domain'].isin(test_domains).values
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("  Пропуск: нет данных для cross-dataset")
        return {}
    
    # Нормализация
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std
    
    # Datasets
    train_dataset = FatigueDataset(X=X_train_norm, y=y_train)
    test_dataset = FatigueDataset(X=X_test_norm, y=y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Модель
    model = FatigueCNN1D(in_channels=X.shape[2])
    class_weights = compute_class_weights(y_train)
    
    trainer = FatigueTrainer(
        model=model,
        device=device,
        learning_rate=0.001,
        class_weights=class_weights
    )
    
    trainer.fit(
        train_loader=train_loader,
        val_loader=test_loader,
        epochs=100,
        early_stopping_patience=15,
        verbose=True
    )
    
    _, metrics = trainer.validate(test_loader)
    metrics['train_domains'] = str(train_domains)
    metrics['test_domains'] = str(test_domains)
    
    print(f"\nCross-Dataset Results:")
    print(f"  F1-macro: {metrics['f1_macro']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    
    return metrics


def main(config_path: str):
    """Главная функция обучения."""
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch не установлен. Установите: pip install torch")
        return
    
    # Загрузка конфигурации
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    os.makedirs(cfg["paths"]["results_root"], exist_ok=True)
    
    # Устройство
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Устройство: {device}")
    
    # Загрузка данных
    print("\nЗагрузка данных...")
    meta = load_windows_meta(cfg)
    X, y, meta = load_all_windows(cfg, meta)
    
    results_all = []
    
    # LOSO протокол
    if cfg.get("splits", {}).get("loso", True):
        loso_results = loso_train_eval(X, y, meta, cfg, device)
        
        for r in loso_results:
            r['protocol'] = 'LOSO'
            r['method'] = '1D-CNN'
        
        results_all.extend(loso_results)
        
        # Агрегация
        if loso_results:
            print("\n" + "=" * 60)
            print("LOSO Summary (1D-CNN):")
            df_loso = pd.DataFrame(loso_results)
            for col in ['f1_macro', 'roc_auc', 'pr_auc', 'balanced_acc', 'brier']:
                if col in df_loso.columns:
                    print(f"  {col}: {df_loso[col].mean():.4f} ± {df_loso[col].std():.4f}")
    
    # Cross-dataset протокол
    cd_result = cross_dataset_train_eval(X, y, meta, cfg, device)
    if cd_result:
        cd_result['protocol'] = 'CrossDataset'
        cd_result['method'] = '1D-CNN'
        results_all.append(cd_result)
    
    # Сохранение результатов
    if results_all:
        out_csv = os.path.join(cfg["paths"]["results_root"], "deep_results.csv")
        pd.DataFrame(results_all).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\nРезультаты сохранены: {out_csv}")
    
    # Сохранение финальной модели (обучение на всех данных)
    print("\nОбучение финальной модели на всех данных...")
    
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - mean) / std
    
    full_dataset = FatigueDataset(X=X_norm, y=y)
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
    
    final_model = FatigueCNN1D(in_channels=X.shape[2])
    class_weights = compute_class_weights(y)
    
    trainer = FatigueTrainer(
        model=final_model,
        device=device,
        learning_rate=0.001,
        class_weights=class_weights
    )
    
    trainer.fit(
        train_loader=full_loader,
        epochs=100,
        early_stopping_patience=20,
        verbose=True
    )
    
    model_path = os.path.join(cfg["paths"]["results_root"], "fatigue_cnn1d.pth")
    save_model(final_model, model_path, metadata={
        'norm_mean': mean.tolist(),
        'norm_std': std.tolist(),
        'n_samples': len(y),
        'class_distribution': np.bincount(y.astype(int)).tolist()
    })
    
    print("\n✓ Обучение завершено!")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Обучение 1D CNN для детекции утомления")
    ap.add_argument("--config", required=True, help="Путь к config.yaml")
    args = ap.parse_args()
    main(args.config)