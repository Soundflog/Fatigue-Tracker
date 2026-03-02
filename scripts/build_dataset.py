#!/usr/bin/env python3
"""
Сборка композиционного датасета из Zenodo и 4TU.

Данные уже сегментированы по шагам (strides):
- Zenodo: CSV файлы, 180 точек/шаг @ 256 Hz
- 4TU: MAT файлы, 150 точек/шаг @ 240 Hz

Выход: NPZ файл с унифицированными шагами (100 точек, 6 каналов).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from tqdm import tqdm

# Добавляем корень проекта в path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Конфигурация
# =============================================================================

TARGET_STRIDE_LEN = 100  # точек на шаг после ресемплинга
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]

ZENODO_FILES = {
    "ax": "Accel_X_all.csv",
    "ay": "Accel_Y_all.csv",
    "az": "Accel_Z_all.csv",
    "gx": "Gyro_X_all.csv",
    "gy": "Gyro_Y_all.csv",
    "gz": "Gyro_Z_all.csv",
}


# =============================================================================
# Утилиты
# =============================================================================

def resample_stride(stride: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ресемплирование одного шага до target_len точек.
    
    Args:
        stride: shape (original_len,) — один канал одного шага
        target_len: целевое количество точек
    
    Returns:
        shape (target_len,)
    """
    original_len = len(stride)
    if original_len == target_len:
        return stride
    
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, stride, kind='linear', fill_value='extrapolate')
    return f(x_new)


def normalize_per_subject(X: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """
    Z-score нормализация по субъектам.
    
    Args:
        X: shape (N_strides, stride_len, n_channels)
        pids: shape (N_strides,) — ID субъектов
    
    Returns:
        нормализованный X
    """
    X_norm = X.copy().astype(np.float32)
    
    for pid in np.unique(pids):
        mask = pids == pid
        subj_data = X_norm[mask]  # (n_subj_strides, stride_len, n_channels)
        
        # Статистики по всем шагам субъекта (по временной оси и шагам)
        for ch in range(subj_data.shape[2]):
            ch_data = subj_data[:, :, ch].flatten()
            mean = ch_data.mean()
            std = ch_data.std() + 1e-8
            X_norm[mask, :, ch] = (subj_data[:, :, ch] - mean) / std
    
    return X_norm


# =============================================================================
# Загрузка Zenodo
# =============================================================================

def load_zenodo(data_dir: Path) -> dict:
    """
    Загрузка шагов из CSV файлов Zenodo.
    
    Структура CSV (без заголовка):
    - Столбец 0: participant_id (1-19)
    - Столбец 1: label ('F' или 'NF')
    - Столбцы 2-181: временной ряд (180 точек)
    
    Returns:
        dict с X, y, pids, domain
    """
    print("\n📂 Загрузка Zenodo...")
    
    # Проверяем наличие файлов
    first_file = data_dir / ZENODO_FILES["ax"]
    if not first_file.exists():
        raise FileNotFoundError(f"Файл не найден: {first_file}")
    
    # Загружаем первый файл для получения метаданных
    df_first = pd.read_csv(first_file, header=None)
    n_strides = len(df_first)
    stride_len = df_first.shape[1] - 2  # минус pid и label
    
    print(f"   Найдено шагов: {n_strides}")
    print(f"   Длина шага: {stride_len} точек")
    
    # Метаданные
    pids = df_first.iloc[:, 0].values.astype(int)
    labels_str = df_first.iloc[:, 1].values
    labels = (labels_str == 'F').astype(np.int8)  # F=1, NF=0
    
    print(f"   Уникальных субъектов: {len(np.unique(pids))}")
    print(f"   Баланс классов: {labels.sum()}/{len(labels)} ({labels.mean()*100:.1f}% F)")
    
    # Загружаем все каналы
    channels_data = {}
    for ch_name, fname in ZENODO_FILES.items():
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"   ⚠️ Канал {ch_name} не найден, заполняем нулями")
            channels_data[ch_name] = np.zeros((n_strides, stride_len))
        else:
            df = pd.read_csv(fpath, header=None)
            channels_data[ch_name] = df.iloc[:, 2:].values.astype(np.float32)
    
    # Ресемплирование до TARGET_STRIDE_LEN
    print(f"   Ресемплирование {stride_len} → {TARGET_STRIDE_LEN} точек...")
    X = np.zeros((n_strides, TARGET_STRIDE_LEN, len(CHANNELS)), dtype=np.float32)
    
    for i in tqdm(range(n_strides), desc="   Zenodo strides", leave=False):
        for ch_idx, ch_name in enumerate(CHANNELS):
            X[i, :, ch_idx] = resample_stride(channels_data[ch_name][i], TARGET_STRIDE_LEN)
    
    return {
        'X': X,
        'y': labels,
        'pids': np.array([f"zenodo_{p}" for p in pids]),
        'domain': np.array(['zenodo'] * n_strides),
    }


# =============================================================================
# Загрузка 4TU
# =============================================================================

def load_4tu(data_dir: Path, segment: str = 'pelvis') -> dict:
    """
    Загрузка шагов из MAT файлов 4TU.
    
    Args:
        data_dir: путь к директории с MAT файлами
        segment: сегмент тела для использования ('pelvis', 'sternum', etc.)
    
    Returns:
        dict с X, y, pids, domain
    """
    print("\n📂 Загрузка 4TU...")
    
    try:
        import scipy.io
    except ImportError:
        raise ImportError("scipy не установлен. Выполните: pip install scipy")
    
    stride_files = sorted(data_dir.glob('p*_strides.mat'))
    if not stride_files:
        raise FileNotFoundError(f"MAT файлы не найдены в {data_dir}")
    
    print(f"   Найдено файлов: {len(stride_files)}")
    
    all_X = []
    all_labels = []
    all_pids = []
    
    for mat_file in tqdm(stride_files, desc="   4TU subjects"):
        pid = int(mat_file.stem.split('_')[0][1:])  # p001 -> 1
        
        mat = scipy.io.loadmat(mat_file)
        strides = mat['strides'][0, 0]
        
        # Получаем список доступных полей
        field_names = strides.dtype.names
        
        # Ищем данные для выбранного сегмента
        acc_field = f'{segment}_nacc'
        angvel_field = f'{segment}_angvel'
        
        if acc_field not in field_names:
            print(f"   ⚠️ Сегмент {segment} не найден для p{pid:03d}, пропускаем")
            continue
        
        acc_data = strides[acc_field]      # (150, N_strides) или (150, N, 3)
        angvel_data = strides[angvel_field] if angvel_field in field_names else None
        
        # Определяем размерность
        if acc_data.ndim == 2:
            # (150, N_strides) — один канал, магнитуда
            n_strides = acc_data.shape[1]
            stride_len = acc_data.shape[0]
            
            # Создаём 6 каналов (дублируем для ax/ay/az, gx/gy/gz)
            X_subj = np.zeros((n_strides, TARGET_STRIDE_LEN, 6), dtype=np.float32)
            
            for i in range(n_strides):
                # Акселерометр (только магнитуда в ax, остальные нули)
                X_subj[i, :, 0] = resample_stride(acc_data[:, i], TARGET_STRIDE_LEN)
                
                # Гироскоп (если есть)
                if angvel_data is not None:
                    X_subj[i, :, 3] = resample_stride(angvel_data[:, i], TARGET_STRIDE_LEN)
        
        elif acc_data.ndim == 3:
            # (150, N_strides, 3) — три компоненты
            n_strides = acc_data.shape[1]
            stride_len = acc_data.shape[0]
            
            X_subj = np.zeros((n_strides, TARGET_STRIDE_LEN, 6), dtype=np.float32)
            
            for i in range(n_strides):
                for axis in range(min(3, acc_data.shape[2])):
                    X_subj[i, :, axis] = resample_stride(acc_data[:, i, axis], TARGET_STRIDE_LEN)
                
                if angvel_data is not None and angvel_data.ndim == 3:
                    for axis in range(min(3, angvel_data.shape[2])):
                        X_subj[i, :, 3 + axis] = resample_stride(angvel_data[:, i, axis], TARGET_STRIDE_LEN)
        else:
            print(f"   ⚠️ Неизвестная размерность для p{pid:03d}: {acc_data.shape}")
            continue
        
        # Разметка: первые 50% = pre-fatigue (0), последние 50% = post-fatigue (1)
        # TODO: улучшить разметку на основе имён файлов raw данных
        labels = np.zeros(n_strides, dtype=np.int8)
        labels[n_strides // 2:] = 1
        
        all_X.append(X_subj)
        all_labels.append(labels)
        all_pids.extend([f"4tu_{pid}"] * n_strides)
    
    if not all_X:
        raise ValueError("Не удалось загрузить ни одного субъекта из 4TU")
    
    X = np.vstack(all_X)
    y = np.concatenate(all_labels)
    pids = np.array(all_pids)
    
    print(f"   Всего шагов: {len(y)}")
    print(f"   Уникальных субъектов: {len(np.unique(pids))}")
    print(f"   Баланс классов: {y.sum()}/{len(y)} ({y.mean()*100:.1f}% fatigue)")
    
    return {
        'X': X,
        'y': y,
        'pids': pids,
        'domain': np.array(['4tu'] * len(y)),
    }


# =============================================================================
# Сборка композиционного датасета
# =============================================================================

def build_composite_dataset(
    raw_dir: Path,
    output_path: Path,
    use_zenodo: bool = True,
    use_4tu: bool = True,
    segment_4tu: str = 'pelvis',
):
    """Сборка и сохранение композиционного датасета."""
    
    print("=" * 60)
    print("СБОРКА КОМПОЗИЦИОННОГО ДАТАСЕТА")
    print("=" * 60)
    
    datasets = []
    
    # Загрузка Zenodo
    if use_zenodo:
        zenodo_dir = raw_dir / 'zenodo'
        if zenodo_dir.exists():
            try:
                zenodo_data = load_zenodo(zenodo_dir)
                datasets.append(zenodo_data)
            except Exception as e:
                print(f"❌ Ошибка загрузки Zenodo: {e}")
        else:
            print(f"⚠️ Директория Zenodo не найдена: {zenodo_dir}")
    
    # Загрузка 4TU
    if use_4tu:
        ftu_dir = raw_dir / '4tu'
        if ftu_dir.exists():
            try:
                ftu_data = load_4tu(ftu_dir, segment=segment_4tu)
                datasets.append(ftu_data)
            except Exception as e:
                print(f"❌ Ошибка загрузки 4TU: {e}")
        else:
            print(f"⚠️ Директория 4TU не найдена: {ftu_dir}")
    
    if not datasets:
        raise ValueError("Не удалось загрузить ни один датасет!")
    
    # Объединение
    print("\n📦 Объединение датасетов...")
    X = np.vstack([d['X'] for d in datasets])
    y = np.concatenate([d['y'] for d in datasets])
    pids = np.concatenate([d['pids'] for d in datasets])
    domains = np.concatenate([d['domain'] for d in datasets])
    
    print(f"   Всего шагов: {len(y)}")
    print(f"   Форма X: {X.shape}")
    
    # Нормализация по субъектам
    print("\n🔧 Нормализация по субъектам...")
    X = normalize_per_subject(X, pids)
    
    # Проверка на NaN/Inf
    if np.isnan(X).any():
        print("   ⚠️ Обнаружены NaN, заменяем на 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Сохранение
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        output_path,
        X=X.astype(np.float32),
        y=y.astype(np.int8),
        pids=pids,
        domains=domains,
        channels=np.array(CHANNELS),
        target_stride_len=TARGET_STRIDE_LEN,
    )
    
    print("\n" + "=" * 60)
    print("✅ ДАТАСЕТ ГОТОВ")
    print("=" * 60)
    print(f"   Файл: {output_path}")
    print(f"   Размер: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Шагов: {len(y)}")
    print(f"   Форма X: {X.shape}")
    print(f"   Субъектов: {len(np.unique(pids))}")
    print(f"   Label=0: {(y == 0).sum()}")
    print(f"   Label=1: {(y == 1).sum()}")
    print(f"   Баланс: {y.mean()*100:.1f}% fatigue")
    
    # Статистика по доменам
    print("\n📊 По доменам:")
    for domain in np.unique(domains):
        mask = domains == domain
        print(f"   {domain}: {mask.sum()} шагов, "
              f"{y[mask].sum()}/{mask.sum()} ({y[mask].mean()*100:.1f}% fatigue)")
    
    return {
        'X': X,
        'y': y,
        'pids': pids,
        'domains': domains,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Сборка композиционного датасета из Zenodo и 4TU"
    )
    parser.add_argument(
        '--raw-dir', 
        type=Path, 
        default=PROJECT_ROOT / 'data' / 'raw',
        help="Путь к директории с сырыми данными"
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        default=PROJECT_ROOT / 'data' / 'processed' / 'composite_strides.npz',
        help="Путь для сохранения результата"
    )
    parser.add_argument(
        '--no-zenodo', 
        action='store_true',
        help="Не использовать Zenodo"
    )
    parser.add_argument(
        '--no-4tu', 
        action='store_true',
        help="Не использовать 4TU"
    )
    parser.add_argument(
        '--segment', 
        type=str, 
        default='pelvis',
        choices=['pelvis', 'sternum', 'lfoot', 'rfoot', 'lll', 'rll', 'lul', 'rul'],
        help="Сегмент тела для 4TU (default: pelvis)"
    )
    
    args = parser.parse_args()
    
    build_composite_dataset(
        raw_dir=args.raw_dir,
        output_path=args.output,
        use_zenodo=not args.no_zenodo,
        use_4tu=not args.no_4tu,
        segment_4tu=args.segment,
    )


if __name__ == '__main__':
    main()
