#!/usr/bin/env python3
"""
Сборка композиционного датасета из Zenodo, 4TU, PhysioNet и WSD4FEDSRM (v4.0).

Источники:
- Zenodo: CSV, stride-segmented, 180 точек/шаг @ 256 Hz, IMU (acc+gyro)
- 4TU: MAT, stride-segmented, 150 точек/шаг @ 240 Hz, IMU (acc+gyro)
- PhysioNet: Empatica E4 CSV, непрерывные записи,
             ACC(32Hz)+BVP(64Hz)+EDA(4Hz)+TEMP(4Hz)+HR(1Hz)
- WSD4FEDSRM: IMU Sternum (100Hz) + PPG (200Hz), Borg RPE labeling

Выход: NPZ с двумя тензорами:
  X_imu      (N, 100, 6)  — ax, ay, az, gx, gy, gz
  X_physio   (N, 100, 4)  — bvp, eda, temp, hr
  y          (N,)
  pids       (N,)
  domains    (N,)
  has_physio (N,)          — True для PhysioNet/WSD4FEDSRM, False для Zenodo/4TU
"""

import argparse
import sys
from datetime import datetime
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

TARGET_STRIDE_LEN = 100  # точек на окно после ресемплинга

IMU_CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
PHYSIO_CHANNELS = ["bvp", "eda", "temp", "hr"]

ZENODO_FILES = {
    "ax": "Accel_X_all.csv",
    "ay": "Accel_Y_all.csv",
    "az": "Accel_Z_all.csv",
    "gx": "Gyro_X_all.csv",
    "gy": "Gyro_Y_all.csv",
    "gz": "Gyro_Z_all.csv",
}

# PhysioNet: исключённые сессии из data_constraints.txt
PHYSIONET_EXCLUDE = {
    "AEROBIC": {"S12"},           # отсутствует
    "ANAEROBIC": set(),
}

PHYSIONET_WINDOW_SEC = 5.0    # длина окна в секундах
PHYSIONET_STRIDE_SEC = 5.0    # шаг окна — без overlap (v4.0: было 2.5)
PHYSIONET_FATIGUE_RATIO = 0.5 # последние 50% фаз → fatigue


# =============================================================================
# Утилиты
# =============================================================================

def resample_stride(stride: np.ndarray, target_len: int) -> np.ndarray:
    """
    Ресемплирование одного сигнала до target_len точек.

    Args:
        stride: shape (original_len,) — один канал
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
    Z-score нормализация по субъектам (поканально).

    Args:
        X: shape (N, T, C)
        pids: shape (N,)

    Returns:
        нормализованный X (float32)
    """
    X_norm = X.copy().astype(np.float32)

    for pid in np.unique(pids):
        mask = pids == pid
        subj_data = X_norm[mask]

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
        dict с X_imu, X_physio, y, pids, domain, has_physio
    """
    print("\n📂 Загрузка Zenodo...")

    first_file = data_dir / ZENODO_FILES["ax"]
    if not first_file.exists():
        raise FileNotFoundError(f"Файл не найден: {first_file}")

    df_first = pd.read_csv(first_file, header=None)
    n_strides = len(df_first)
    stride_len = df_first.shape[1] - 2

    print(f"   Найдено шагов: {n_strides}")
    print(f"   Длина шага: {stride_len} точек")

    # Метаданные
    pids = df_first.iloc[:, 0].values.astype(int)
    labels_str = df_first.iloc[:, 1].values
    labels = (labels_str == 'F').astype(np.int8)

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

    # Ресемплирование
    print(f"   Ресемплирование {stride_len} → {TARGET_STRIDE_LEN} точек...")
    X_imu = np.zeros((n_strides, TARGET_STRIDE_LEN, 6), dtype=np.float32)

    for i in tqdm(range(n_strides), desc="   Zenodo strides", leave=False):
        for ch_idx, ch_name in enumerate(IMU_CHANNELS):
            X_imu[i, :, ch_idx] = resample_stride(channels_data[ch_name][i], TARGET_STRIDE_LEN)

    # Zenodo не имеет физиологических каналов
    X_physio = np.zeros((n_strides, TARGET_STRIDE_LEN, 4), dtype=np.float32)

    return {
        'X_imu': X_imu,
        'X_physio': X_physio,
        'y': labels,
        'pids': np.array([f"zenodo_{p}" for p in pids]),
        'domain': np.array(['zenodo'] * n_strides),
        'has_physio': np.zeros(n_strides, dtype=bool),
    }


# =============================================================================
# Загрузка 4TU
# =============================================================================

def load_4tu(data_dir: Path, segment: str = 'pelvis') -> dict:
    """
    Загрузка шагов из MAT файлов 4TU.

    Returns:
        dict с X_imu, X_physio, y, pids, domain, has_physio
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
        field_names = strides.dtype.names

        acc_field = f'{segment}_nacc'
        angvel_field = f'{segment}_angvel'

        if acc_field not in field_names:
            print(f"   ⚠️ Сегмент {segment} не найден для p{pid:03d}, пропускаем")
            continue

        acc_data = strides[acc_field]
        angvel_data = strides[angvel_field] if angvel_field in field_names else None

        # 4TU _nacc fields are acceleration norms (scalar per time-step)
        # Also load jerk for additional gait kinematic channel
        jerk_field = f'{segment}_jerk'
        jerk_data = strides[jerk_field] if jerk_field in field_names else None

        if acc_data.ndim == 2:
            n_strides = acc_data.shape[1]
            X_subj = np.zeros((n_strides, TARGET_STRIDE_LEN, 6), dtype=np.float32)

            for i in range(n_strides):
                X_subj[i, :, 0] = resample_stride(acc_data[:, i], TARGET_STRIDE_LEN)
                if jerk_data is not None and jerk_data.ndim >= 2:
                    X_subj[i, :, 1] = resample_stride(jerk_data[:, i], TARGET_STRIDE_LEN)
                if angvel_data is not None:
                    X_subj[i, :, 3] = resample_stride(angvel_data[:, i], TARGET_STRIDE_LEN)

        elif acc_data.ndim == 3:
            n_strides = acc_data.shape[1]
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

        # 4TU protocol: 4000m pre-fatigue + 1200m post-fatigue
        # Check if postfatigue trial exists for this subject
        postfatigue_exists = any(data_dir.glob(f'p{pid:03d}_HDSL_postfatigue*.mat'))
        if not postfatigue_exists:
            print(f"   ⚠️ p{pid:03d}: нет postfatigue файла, пропускаем")
            continue

        # Last ~23% of strides are post-fatigue (1200m out of 5200m total)
        fatigue_frac = 1200.0 / (4000.0 + 1200.0)
        n_fatigue = max(1, int(n_strides * fatigue_frac))
        labels = np.zeros(n_strides, dtype=np.int8)
        labels[-n_fatigue:] = 1

        all_X.append(X_subj)
        all_labels.append(labels)
        all_pids.extend([f"4tu_{pid}"] * n_strides)

    if not all_X:
        raise ValueError("Не удалось загрузить ни одного субъекта из 4TU")

    X_imu = np.vstack(all_X)
    y = np.concatenate(all_labels)
    pids = np.array(all_pids)
    n_total = len(y)

    X_physio = np.zeros((n_total, TARGET_STRIDE_LEN, 4), dtype=np.float32)

    print(f"   Всего шагов: {n_total}")
    print(f"   Уникальных субъектов: {len(np.unique(pids))}")
    print(f"   Баланс классов: {y.sum()}/{n_total} ({y.mean()*100:.1f}% fatigue)")

    return {
        'X_imu': X_imu,
        'X_physio': X_physio,
        'y': y,
        'pids': pids,
        'domain': np.array(['4tu'] * n_total),
        'has_physio': np.zeros(n_total, dtype=bool),
    }


# =============================================================================
# Загрузка PhysioNet
# =============================================================================

def load_empatica_csv(filepath: Path) -> tuple:
    """
    Читает CSV файл формата Empatica E4.

    Формат:
    - Строка 1: начальное время (datetime) | для ACC — 3 столбца
    - Строка 2: частота дискретизации (Hz)
    - Строки 3+: значения сигнала

    Returns:
        (start_time: datetime, fs: float, data: np.ndarray)
    """
    with open(filepath, 'r') as f:
        header_line = f.readline().strip()
        fs_line = f.readline().strip()

    # Начальное время (берём первый элемент — для ACC их 3)
    start_str = header_line.split(',')[0].strip()
    try:
        start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        start_time = datetime.utcfromtimestamp(float(start_str))

    fs = float(fs_line.split(',')[0].strip())
    data = pd.read_csv(filepath, header=None, skiprows=2).values.astype(np.float32)

    return start_time, fs, data


def load_tags(filepath: Path) -> list:
    """
    Читает tags.csv — UTC-метки фаз протокола.

    Returns:
        Список datetime объектов
    """
    tags = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tags.append(datetime.strptime(line, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                try:
                    tags.append(datetime.utcfromtimestamp(float(line)))
                except Exception:
                    continue
    return tags


def assign_fatigue_labels_physionet(
    n_windows: int,
    window_centers_sec: np.ndarray,
    tags_sec: np.ndarray,
    fatigue_ratio: float = 0.5,
) -> np.ndarray:
    """
    Разметка окон PhysioNet по протоколу.

    Стратегия: первые (1-fatigue_ratio) от первого до последнего тега = NF,
    остальные = F.

    Returns:
        labels: shape (n_windows,) — 0/1
    """
    labels = np.zeros(n_windows, dtype=np.int8)

    if len(tags_sec) >= 2:
        proto_start = tags_sec[0]
        proto_end = tags_sec[-1]
        proto_duration = proto_end - proto_start

        if proto_duration > 0:
            fatigue_threshold = proto_start + proto_duration * (1 - fatigue_ratio)
            for i in range(n_windows):
                if window_centers_sec[i] >= fatigue_threshold:
                    labels[i] = 1
        else:
            mid = n_windows // 2
            labels[mid:] = 1
    else:
        mid = n_windows // 2
        labels[mid:] = 1

    return labels


def extract_physionet_windows(
    acc_data: np.ndarray,
    bvp_data: np.ndarray,
    eda_data: np.ndarray,
    temp_data: np.ndarray,
    hr_data: np.ndarray,
    fs_acc: float,
    fs_bvp: float,
    fs_eda: float,
    fs_temp: float,
    fs_hr: float,
    window_sec: float = 5.0,
    stride_sec: float = 2.5,
    target_len: int = 100,
) -> tuple:
    """
    Скользящее окно и ресемплирование PhysioNet сигналов.

    Returns:
        X_imu: (N_win, target_len, 6) — ACC в каналах 0-2, остальные 0
        X_physio: (N_win, target_len, 4) — BVP, EDA, TEMP, HR
        window_centers_sec: (N_win,)
    """
    durations = []
    if len(acc_data) > 0:
        durations.append(len(acc_data) / fs_acc)
    if len(bvp_data) > 0:
        durations.append(len(bvp_data) / fs_bvp)
    if len(eda_data) > 0:
        durations.append(len(eda_data) / fs_eda)
    if len(temp_data) > 0:
        durations.append(len(temp_data) / fs_temp)
    if len(hr_data) > 0:
        durations.append(len(hr_data) / fs_hr)

    if not durations:
        return np.empty((0, target_len, 6)), np.empty((0, target_len, 4)), np.empty(0)

    total_sec = min(durations)
    window_starts = np.arange(0, total_sec - window_sec + 1e-6, stride_sec)
    n_win = len(window_starts)

    if n_win == 0:
        return np.empty((0, target_len, 6)), np.empty((0, target_len, 4)), np.empty(0)

    X_imu = np.zeros((n_win, target_len, 6), dtype=np.float32)
    X_physio = np.zeros((n_win, target_len, 4), dtype=np.float32)
    centers = window_starts + window_sec / 2

    for w in range(n_win):
        t_start = window_starts[w]
        t_end = t_start + window_sec

        # IMU: ACC (3 оси) → каналы 0,1,2; gx,gy,gz = 0 (нет гироскопа)
        i0 = int(t_start * fs_acc)
        i1 = min(int(t_end * fs_acc), len(acc_data))
        if i1 > i0:
            seg = acc_data[i0:i1]
            if seg.ndim == 1:
                seg = seg.reshape(-1, 1)
            for c in range(min(seg.shape[1], 3)):
                X_imu[w, :, c] = resample_stride(seg[:, c], target_len)

        # Physio: BVP → канал 0
        i0 = int(t_start * fs_bvp)
        i1 = min(int(t_end * fs_bvp), len(bvp_data))
        if i1 > i0:
            X_physio[w, :, 0] = resample_stride(bvp_data[i0:i1].flatten(), target_len)

        # Physio: EDA → канал 1
        i0 = int(t_start * fs_eda)
        i1 = min(int(t_end * fs_eda), len(eda_data))
        if i1 > i0:
            X_physio[w, :, 1] = resample_stride(eda_data[i0:i1].flatten(), target_len)

        # Physio: TEMP → канал 2
        i0 = int(t_start * fs_temp)
        i1 = min(int(t_end * fs_temp), len(temp_data))
        if i1 > i0:
            X_physio[w, :, 2] = resample_stride(temp_data[i0:i1].flatten(), target_len)

        # Physio: HR → канал 3
        i0 = int(t_start * fs_hr)
        i1 = min(int(t_end * fs_hr), len(hr_data))
        if i1 > i0:
            X_physio[w, :, 3] = resample_stride(hr_data[i0:i1].flatten(), target_len)

    return X_imu, X_physio, centers


def load_physionet(
    data_dir: Path,
    protocols: list = None,
    window_sec: float = 5.0,
    stride_sec: float = 2.5,
    fatigue_ratio: float = 0.5,
) -> dict:
    """
    Загрузка данных PhysioNet (Empatica E4) со скользящим окном.

    Returns:
        dict с X_imu, X_physio, y, pids, domain, has_physio
    """
    print("\n📂 Загрузка PhysioNet...")

    if protocols is None:
        protocols = ["AEROBIC", "ANAEROBIC"]

    all_X_imu = []
    all_X_physio = []
    all_y = []
    all_pids = []

    for protocol in protocols:
        proto_dir = data_dir / protocol
        if not proto_dir.exists():
            print(f"   ⚠️ Протокол {protocol} не найден: {proto_dir}")
            continue

        excluded = PHYSIONET_EXCLUDE.get(protocol, set())
        subj_dirs = sorted([d for d in proto_dir.iterdir() if d.is_dir()])
        print(f"\n   📁 Протокол {protocol}: {len(subj_dirs)} участников")

        for subj_dir in tqdm(subj_dirs, desc=f"   {protocol}"):
            subj_id = subj_dir.name
            subj_id_clean = subj_id.replace('_', '')
            base_id = subj_id.split('_')[0]

            if base_id in excluded:
                continue

            acc_path = subj_dir / 'ACC.csv'
            if not acc_path.exists():
                continue

            try:
                start_acc, fs_acc, acc_data = load_empatica_csv(acc_path)

                bvp_path = subj_dir / 'BVP.csv'
                if bvp_path.exists():
                    _, fs_bvp, bvp_data = load_empatica_csv(bvp_path)
                else:
                    fs_bvp, bvp_data = 64.0, np.empty(0)

                eda_path = subj_dir / 'EDA.csv'
                if eda_path.exists():
                    _, fs_eda, eda_data = load_empatica_csv(eda_path)
                else:
                    fs_eda, eda_data = 4.0, np.empty(0)

                temp_path = subj_dir / 'TEMP.csv'
                if temp_path.exists():
                    _, fs_temp, temp_data = load_empatica_csv(temp_path)
                else:
                    fs_temp, temp_data = 4.0, np.empty(0)

                hr_path = subj_dir / 'HR.csv'
                if hr_path.exists():
                    _, fs_hr, hr_data = load_empatica_csv(hr_path)
                else:
                    fs_hr, hr_data = 1.0, np.empty(0)

                tags_path = subj_dir / 'tags.csv'
                tags = load_tags(tags_path) if tags_path.exists() else []
                tags_sec = np.array([(t - start_acc).total_seconds() for t in tags])
                tags_sec = tags_sec[tags_sec >= 0]

                X_imu, X_physio, centers = extract_physionet_windows(
                    acc_data=acc_data,
                    bvp_data=bvp_data,
                    eda_data=eda_data,
                    temp_data=temp_data,
                    hr_data=hr_data,
                    fs_acc=fs_acc,
                    fs_bvp=fs_bvp,
                    fs_eda=fs_eda,
                    fs_temp=fs_temp,
                    fs_hr=fs_hr,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    target_len=TARGET_STRIDE_LEN,
                )

                if len(centers) == 0:
                    continue

                labels = assign_fatigue_labels_physionet(
                    n_windows=len(centers),
                    window_centers_sec=centers,
                    tags_sec=tags_sec,
                    fatigue_ratio=fatigue_ratio,
                )

                all_X_imu.append(X_imu)
                all_X_physio.append(X_physio)
                all_y.append(labels)
                # Use base person ID (not session suffix) to prevent same-person leakage
                # S11_a/S11_b → S11, S16_a/S16_b → S16
                pid_str = f"physionet_{base_id}"
                all_pids.extend([pid_str] * len(labels))

            except Exception as e:
                print(f"      ❌ {subj_id}: {e}")
                continue

    if not all_X_imu:
        raise ValueError("Не удалось загрузить ни одного субъекта PhysioNet")

    X_imu = np.vstack(all_X_imu)
    X_physio = np.vstack(all_X_physio)
    y = np.concatenate(all_y)
    pids = np.array(all_pids)
    n_total = len(y)

    print(f"\n   ✅ PhysioNet итого:")
    print(f"      Окон: {n_total}")
    print(f"      Уникальных субъектов: {len(np.unique(pids))}")
    print(f"      Баланс: {y.sum()}/{n_total} ({y.mean()*100:.1f}% fatigue)")

    return {
        'X_imu': X_imu,
        'X_physio': X_physio,
        'y': y,
        'pids': pids,
        'domain': np.array(['physionet'] * n_total),
        'has_physio': np.ones(n_total, dtype=bool),
    }


# =============================================================================
# Загрузка WSD4FEDSRM
# =============================================================================

# Маппинг папок → task id (MVIC исключены)
WSD4FEDSRM_TASKS = {
    "30-40_ internal rotation": "task1_35i",
    "30-40_ external rotation": "task4_35e",
    "40-50_ internal rotation": "task2_45i",
    "40-50_ external rotation": "task5_45e",
    "50-60_ internal rotation": "task3_55i",
    "50-60_ external rotation": "task6_55e",
}

WSD4FEDSRM_IMU_FS = 100   # Hz
WSD4FEDSRM_PPG_FS = 200   # Hz
WSD4FEDSRM_WINDOW_SEC = 10 # RPE каждые 10 сек
WSD4FEDSRM_RPE_NORMAL = 11   # RPE <= 11 → 0
WSD4FEDSRM_RPE_FATIGUE = 14  # RPE >= 14 → 1


def load_wsd4fedsrm(
    data_dir: Path,
    segment: str = "Sternum",
    target_len: int = 100,
) -> dict:
    """
    Загрузка WSD4FEDSRM: IMU (Sternum) + PPG, разметка по Borg RPE.

    PID стратегия: по человеку (wsd_001 ... wsd_034), НЕ по задаче.

    Returns:
        dict с X_imu, X_physio, y, pids, domain, has_physio
    """
    print("\n📂 Загрузка WSD4FEDSRM...")

    borg_path = data_dir / "Borg data" / "borg_data.csv"
    if not borg_path.exists():
        raise FileNotFoundError(f"Borg data не найден: {borg_path}")

    borg_df = pd.read_csv(borg_path)
    borg_df.columns = [c.strip() for c in borg_df.columns]
    borg_df["subject"] = borg_df["subject"].ffill()  # subject заполнен только на первой строке группы

    # Определяем RPE-столбцы (10_sec, 20_sec, ... но НЕ length_of_trial_(sec))
    rpe_cols = []
    for col in borg_df.columns:
        if "_sec" in col and "length" not in col.lower() and col not in ("before_task", "end_of_trial"):
            try:
                t = int(col.split("_")[0])
                rpe_cols.append((t, col))
            except ValueError:
                continue
    rpe_cols.sort(key=lambda x: x[0])

    print(f"   RPE столбцов: {len(rpe_cols)} ({rpe_cols[0][1]} ... {rpe_cols[-1][1]})")

    data_root = data_dir / "EMG, IMU, and PPG data"

    all_X_imu = []
    all_X_physio = []
    all_y = []
    all_pids = []
    skipped_trials = 0

    for folder, task_id in WSD4FEDSRM_TASKS.items():
        folder_path = data_root / folder
        if not folder_path.exists():
            print(f"   ⚠️ Папка не найдена: {folder}")
            continue

        subj_dirs = sorted(
            [d for d in folder_path.iterdir() if d.is_dir() and d.name.startswith("Subject")],
            key=lambda d: int(d.name.split()[-1]),
        )

        for subj_dir in subj_dirs:
            pid_num = int(subj_dir.name.split()[-1])
            pid_str = f"wsd_{pid_num:03d}"

            # Borg RPE для этого субъекта + задачи
            subj_key = f"subject_{pid_num}"
            borg_row = borg_df[
                (borg_df["subject"] == subj_key) &
                (borg_df["task_order"].astype(str).str.strip() == task_id)
            ]
            if borg_row.empty:
                skipped_trials += 1
                continue

            borg_row = borg_row.iloc[0]

            # IMU файлы
            acc_path = subj_dir / "IMU data" / segment / f"acc_{segment.lower()}.csv"
            gyr_path = subj_dir / "IMU data" / segment / f"gyr_{segment.lower()}.csv"

            if not acc_path.exists():
                skipped_trials += 1
                continue

            try:
                acc = pd.read_csv(acc_path).values.astype(np.float32)
                gyr = pd.read_csv(gyr_path).values.astype(np.float32) if gyr_path.exists() else np.zeros_like(acc)

                # PPG
                ppg_path = subj_dir / "PPG data" / "ppg.csv"
                ppg = pd.read_csv(ppg_path).values.astype(np.float32).ravel() if ppg_path.exists() else np.array([])

                imu_data = np.hstack([acc[:, :3], gyr[:, :3]])  # (N, 6)

                # Нарезка на 10-сек окна с RPE labels
                for t_sec, col_name in rpe_cols:
                    rpe_val = borg_row.get(col_name)
                    if pd.isna(rpe_val):
                        continue
                    rpe_val = float(rpe_val)

                    if rpe_val <= WSD4FEDSRM_RPE_NORMAL:
                        label = 0
                    elif rpe_val >= WSD4FEDSRM_RPE_FATIGUE:
                        label = 1
                    else:
                        continue  # RPE 12-13: skip

                    # Окно индексов
                    win_idx = (t_sec // WSD4FEDSRM_WINDOW_SEC) - 1
                    imu_start = win_idx * WSD4FEDSRM_WINDOW_SEC * WSD4FEDSRM_IMU_FS
                    imu_end = (win_idx + 1) * WSD4FEDSRM_WINDOW_SEC * WSD4FEDSRM_IMU_FS
                    imu_start, imu_end = int(imu_start), int(imu_end)

                    if imu_end > len(imu_data) or (imu_end - imu_start) < 50:
                        continue

                    imu_seg = imu_data[imu_start:imu_end]
                    x_imu = np.zeros((target_len, 6), dtype=np.float32)
                    for c in range(6):
                        x_imu[:, c] = resample_stride(imu_seg[:, c], target_len)

                    x_physio = np.zeros((target_len, 4), dtype=np.float32)
                    if len(ppg) > 0:
                        ppg_start = int(imu_start * WSD4FEDSRM_PPG_FS / WSD4FEDSRM_IMU_FS)
                        ppg_end = int(imu_end * WSD4FEDSRM_PPG_FS / WSD4FEDSRM_IMU_FS)
                        if ppg_end <= len(ppg) and (ppg_end - ppg_start) >= 50:
                            x_physio[:, 0] = resample_stride(ppg[ppg_start:ppg_end], target_len)

                    all_X_imu.append(x_imu)
                    all_X_physio.append(x_physio)
                    all_y.append(label)
                    all_pids.append(pid_str)

            except Exception as e:
                print(f"      ❌ Subject {pid_num} / {task_id}: {e}")
                skipped_trials += 1
                continue

    if not all_X_imu:
        raise ValueError("Не удалось загрузить ни одного окна WSD4FEDSRM")

    X_imu = np.stack(all_X_imu)
    X_physio = np.stack(all_X_physio)
    y = np.array(all_y, dtype=np.int8)
    pids = np.array(all_pids)
    n_total = len(y)

    print(f"\n   ✅ WSD4FEDSRM итого:")
    print(f"      Окон: {n_total}")
    print(f"      Пропущено trials: {skipped_trials}")
    print(f"      Уникальных субъектов (PID): {len(np.unique(pids))}")
    print(f"      Баланс: {y.sum()}/{n_total} ({y.mean()*100:.1f}% fatigue)")

    return {
        'X_imu': X_imu,
        'X_physio': X_physio,
        'y': y,
        'pids': pids,
        'domain': np.array(['wsd4fedsrm'] * n_total),
        # WSD4FEDSRM PPG ≠ Empatica BVP; EDA/TEMP/HR channels are zeros
        # Mark as no-physio to avoid misleading the physio encoder
        'has_physio': np.zeros(n_total, dtype=bool),
    }


# =============================================================================
# Сборка композиционного датасета
# =============================================================================

def build_composite_dataset(
    raw_dir: Path,
    physionet_dir: Path,
    output_path: Path,
    use_zenodo: bool = True,
    use_4tu: bool = True,
    use_physionet: bool = True,
    use_wsd4fedsrm: bool = True,
    segment_4tu: str = 'pelvis',
    segment_wsd: str = 'Forearm',
):
    """Сборка и сохранение композиционного датасета (dual-branch, v4.0)."""

    print("=" * 60)
    print("СБОРКА КОМПОЗИЦИОННОГО ДАТАСЕТА (v4.0)")
    print("=" * 60)

    datasets = []

    if use_zenodo:
        zenodo_dir = raw_dir / 'zenodo'
        if zenodo_dir.exists():
            try:
                datasets.append(load_zenodo(zenodo_dir))
            except Exception as e:
                print(f"❌ Ошибка загрузки Zenodo: {e}")
        else:
            print(f"⚠️ Директория Zenodo не найдена: {zenodo_dir}")

    if use_4tu:
        ftu_dir = raw_dir / '4tu'
        if ftu_dir.exists():
            try:
                datasets.append(load_4tu(ftu_dir, segment=segment_4tu))
            except Exception as e:
                print(f"❌ Ошибка загрузки 4TU: {e}")
        else:
            print(f"⚠️ Директория 4TU не найдена: {ftu_dir}")

    if use_physionet:
        if physionet_dir.exists():
            try:
                datasets.append(load_physionet(
                    physionet_dir,
                    protocols=["AEROBIC", "ANAEROBIC"],
                    window_sec=PHYSIONET_WINDOW_SEC,
                    stride_sec=PHYSIONET_STRIDE_SEC,
                    fatigue_ratio=PHYSIONET_FATIGUE_RATIO,
                ))
            except Exception as e:
                print(f"❌ Ошибка загрузки PhysioNet: {e}")
        else:
            print(f"⚠️ Директория PhysioNet не найдена: {physionet_dir}")

    if use_wsd4fedsrm:
        wsd_dir = raw_dir / 'WSD4FEDSRM'
        if wsd_dir.exists():
            try:
                datasets.append(load_wsd4fedsrm(wsd_dir, segment=segment_wsd))
            except Exception as e:
                print(f"❌ Ошибка загрузки WSD4FEDSRM: {e}")
        else:
            print(f"⚠️ Директория WSD4FEDSRM не найдена: {wsd_dir}")

    if not datasets:
        raise ValueError("Не удалось загрузить ни один датасет!")

    # Объединение
    print("\n📦 Объединение датасетов...")
    X_imu = np.vstack([d['X_imu'] for d in datasets])
    X_physio = np.vstack([d['X_physio'] for d in datasets])
    y = np.concatenate([d['y'] for d in datasets])
    pids = np.concatenate([d['pids'] for d in datasets])
    domains = np.concatenate([d['domain'] for d in datasets])
    has_physio = np.concatenate([d['has_physio'] for d in datasets])

    n_total = len(y)
    print(f"   Всего окон: {n_total}")
    print(f"   X_imu shape: {X_imu.shape}")
    print(f"   X_physio shape: {X_physio.shape}")

    # Нормализация IMU по субъектам
    print("\n🔧 Нормализация IMU по субъектам...")
    X_imu = normalize_per_subject(X_imu, pids)

    # Нормализация Physio (только для окон с has_physio=True)
    physio_mask = has_physio.astype(bool)
    if physio_mask.any():
        print("🔧 Нормализация Physio по субъектам...")
        X_physio_sub = X_physio[physio_mask]
        pids_sub = pids[physio_mask]
        X_physio[physio_mask] = normalize_per_subject(X_physio_sub, pids_sub)

    # Проверка на NaN/Inf
    for name, arr in [('X_imu', X_imu), ('X_physio', X_physio)]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"   ⚠️ {name}: NaN/Inf → замена на 0")
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Сохранение
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X_imu=X_imu.astype(np.float32),
        X_physio=X_physio.astype(np.float32),
        y=y.astype(np.int8),
        pids=pids,
        domains=domains,
        has_physio=has_physio,
        imu_channels=np.array(IMU_CHANNELS),
        physio_channels=np.array(PHYSIO_CHANNELS),
        target_stride_len=TARGET_STRIDE_LEN,
    )

    print("\n" + "=" * 60)
    print("✅ ДАТАСЕТ ГОТОВ")
    print("=" * 60)
    print(f"   Файл: {output_path}")
    print(f"   Размер: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"   Окон: {n_total}")
    print(f"   X_imu: {X_imu.shape}")
    print(f"   X_physio: {X_physio.shape}")
    print(f"   Субъектов: {len(np.unique(pids))}")
    print(f"   has_physio: {has_physio.sum()}/{n_total}")
    print(f"   Label=0: {(y == 0).sum()}")
    print(f"   Label=1: {(y == 1).sum()}")
    print(f"   Баланс: {y.mean()*100:.1f}% fatigue")

    print("\n📊 По доменам:")
    for domain in np.unique(domains):
        mask = domains == domain
        print(f"   {domain}: {mask.sum()} окон, "
              f"{y[mask].sum()}/{mask.sum()} ({y[mask].mean()*100:.1f}% fatigue), "
              f"субъектов: {len(np.unique(pids[mask]))}")

    return {
        'X_imu': X_imu,
        'X_physio': X_physio,
        'y': y,
        'pids': pids,
        'domains': domains,
        'has_physio': has_physio,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Сборка композиционного датасета из Zenodo, 4TU, PhysioNet и WSD4FEDSRM"
    )
    parser.add_argument(
        '--raw-dir',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'raw',
        help="Путь к сырым данным (zenodo/, 4tu/)"
    )
    parser.add_argument(
        '--physionet-dir',
        type=Path,
        default=(
            PROJECT_ROOT / 'data' / 'raw'
            / 'wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1'
            / 'Wearable_Dataset'
        ),
        help="Путь к PhysioNet Wearable_Dataset/"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'data' / 'processed' / 'composite_full.npz',
        help="Путь для сохранения результата"
    )
    parser.add_argument('--no-zenodo', action='store_true', help="Не использовать Zenodo")
    parser.add_argument('--no-4tu', action='store_true', help="Не использовать 4TU")
    parser.add_argument('--no-physionet', action='store_true', help="Не использовать PhysioNet")
    parser.add_argument('--no-wsd4fedsrm', action='store_true', help="Не использовать WSD4FEDSRM")
    parser.add_argument(
        '--segment',
        type=str,
        default='pelvis',
        choices=['pelvis', 'sternum', 'lfoot', 'rfoot', 'lll', 'rll', 'lul', 'rul'],
        help="Сегмент тела для 4TU (default: pelvis)"
    )
    parser.add_argument(
        '--wsd-segment',
        type=str,
        default='Forearm',
        choices=['Forearm', 'Hand', 'Sternum', 'Shoulder', 'Pelvis', 'Upper arm'],
        help="Сегмент тела для WSD4FEDSRM (default: Forearm)"
    )

    args = parser.parse_args()

    build_composite_dataset(
        raw_dir=args.raw_dir,
        physionet_dir=args.physionet_dir,
        output_path=args.output,
        use_zenodo=not args.no_zenodo,
        use_4tu=not args.no_4tu,
        use_physionet=not args.no_physionet,
        use_wsd4fedsrm=not args.no_wsd4fedsrm,
        segment_4tu=args.segment,
        segment_wsd=args.wsd_segment,
    )


if __name__ == '__main__':
    main()
