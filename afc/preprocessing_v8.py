"""
preprocessing_v8.py — Data preprocessing utilities for v8.1
(Exercise-only + Stress Profile Embedding).

Exports:
  Constants:   PROFILE_FEATURES, HRV_FEATURES, EDA_FEATURES, ACC_FEATURES, SIGNAL_FEATURES
  Loading:     resample_stride, load_empatica_csv, load_tags, extract_physionet_windows,
               assign_stress_labels_phase_based, normalize_per_subject
  Profile:     build_stress_profiles, extract_physio_reactivity
  NK2 feats:   extract_hrv_features_from_hr, extract_eda_features, extract_acc_features,
               extract_signal_features
  Data build:  build_profiles_all, create_subject_split
  Augment:     augment_sample, IMU_TRANSFORMS, PHYSIO_TRANSFORMS + individual functions
  Dataset:     FatigueDataset, apply_smote_dual_branch
"""

import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import resample as sp_resample

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

try:
    import neurokit2 as nk
except ImportError:
    nk = None
    print("Warning: neurokit2 not installed. NK2 feature extraction will fall back to zeros.")

# ─────────────────────────────────────────────────────────────
# Constants / feature name lists
# ─────────────────────────────────────────────────────────────

TARGET_STRIDE_LEN = 100
STRESS_EXCLUDE = {"S02", "f07"}
STRESS_EXCLUDE_PROFILE = {"S02"}

PROFILE_FEATURES = [
    "sl_baseline", "sl_peak", "sl_reactivity", "sl_mean_tasks",
    "hr_baseline_mean", "hr_tasks_mean", "hr_reactivity",
    "eda_baseline_mean", "eda_tasks_mean", "eda_reactivity",
    "age", "bmi", "gender",
]

HRV_FEATURES = [
    "max_ibi", "min_ibi", "mean_ibi", "hr_mean_ibi",
    "pnn20", "pnn50", "rmssd", "sdnn",
    "total_power", "ratio", "VLF_power", "VLF_peak",
    "LF_power", "LF_peak", "LH_n",
    "HF_power", "HF_peak", "HF_n", "VHF_power", "VHF_peak",
]

EDA_FEATURES = [
    "mean_raw_eda", "std_raw_eda",
    "mean_tonic_eda", "std_tonic_eda",
    "mean_phasic_eda", "std_phasic_eda",
    "tonic_ratio_down", "tonic_ratio_up",
    "peaks_density", "scr_mean_amp", "scr_mean_height",
    "scr_mean_risetime", "scr_mean_recoverytime",
]

ACC_FEATURES = [
    "x_mean", "x_std", "y_mean", "y_std", "z_mean", "z_std",
    "acc_mean", "acc_std", "acc_ratio_down", "acc_ratio_up",
]

SIGNAL_FEATURES = HRV_FEATURES + EDA_FEATURES + ACC_FEATURES

# ─────────────────────────────────────────────────────────────
# Empatica E4 file loading utilities
# ─────────────────────────────────────────────────────────────

def resample_stride(stride: np.ndarray, target_len: int) -> np.ndarray:
    original_len = len(stride)
    if original_len == target_len:
        return stride
    x_old = np.linspace(0, 1, original_len)
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, stride, kind="linear", fill_value="extrapolate")
    return f(x_new)


def load_empatica_csv(filepath: Path):
    """Return (start_time, fs, data_array) from an Empatica E4 CSV."""
    with open(filepath, "r") as f:
        header_line = f.readline().strip()
        fs_line = f.readline().strip()
    start_str = header_line.split(",")[0].strip()
    try:
        start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        start_time = datetime.utcfromtimestamp(float(start_str))
    fs = float(fs_line.split(",")[0].strip())
    data = pd.read_csv(filepath, header=None, skiprows=2).values.astype(np.float32)
    return start_time, fs, data


def load_tags(filepath: Path):
    """Return list of datetime objects from an Empatica tags.csv."""
    tags = []
    with open(filepath, "r") as f:
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


def extract_physionet_windows(
    acc_data, bvp_data, eda_data, temp_data, hr_data,
    fs_acc, fs_bvp, fs_eda, fs_temp, fs_hr,
    window_sec=5.0, stride_sec=5.0, target_len=100,
):
    durations = []
    for d, fs in [(acc_data, fs_acc), (bvp_data, fs_bvp), (eda_data, fs_eda),
                  (temp_data, fs_temp), (hr_data, fs_hr)]:
        if len(d) > 0:
            durations.append(len(d) / fs)
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
        t0, t1 = window_starts[w], window_starts[w] + window_sec
        i0, i1 = int(t0 * fs_acc), min(int(t1 * fs_acc), len(acc_data))
        if i1 > i0:
            seg = acc_data[i0:i1]
            if seg.ndim == 1:
                seg = seg.reshape(-1, 1)
            for c in range(min(seg.shape[1], 3)):
                X_imu[w, :, c] = resample_stride(seg[:, c], target_len)
        i0, i1 = int(t0 * fs_bvp), min(int(t1 * fs_bvp), len(bvp_data))
        if i1 > i0:
            X_physio[w, :, 0] = resample_stride(bvp_data[i0:i1].flatten(), target_len)
        i0, i1 = int(t0 * fs_eda), min(int(t1 * fs_eda), len(eda_data))
        if i1 > i0:
            X_physio[w, :, 1] = resample_stride(eda_data[i0:i1].flatten(), target_len)
        i0, i1 = int(t0 * fs_temp), min(int(t1 * fs_temp), len(temp_data))
        if i1 > i0:
            X_physio[w, :, 2] = resample_stride(temp_data[i0:i1].flatten(), target_len)
        i0, i1 = int(t0 * fs_hr), min(int(t1 * fs_hr), len(hr_data))
        if i1 > i0:
            X_physio[w, :, 3] = resample_stride(hr_data[i0:i1].flatten(), target_len)

    return X_imu, X_physio, centers


def assign_stress_labels_phase_based(n_windows, window_centers_sec, tags_sec):
    """Phase-based labeling for STRESS protocol (odd segments = stress)."""
    labels = np.zeros(n_windows, dtype=np.int8)
    if len(tags_sec) < 2:
        mid = n_windows // 2
        labels[mid:] = 1
        return labels
    for i in range(n_windows):
        t = window_centers_sec[i]
        seg_idx = 0
        for j in range(len(tags_sec)):
            if t >= tags_sec[j]:
                seg_idx = j
        if seg_idx % 2 == 1:
            labels[i] = 1
    return labels


def normalize_per_subject(X, pids):
    X_norm = X.copy().astype(np.float32)
    for pid in np.unique(pids):
        mask = pids == pid
        subj_data = X_norm[mask]
        for ch in range(subj_data.shape[2]):
            ch_data = subj_data[:, :, ch].flatten()
            mean, std = ch_data.mean(), ch_data.std() + 1e-8
            X_norm[mask, :, ch] = (subj_data[:, :, ch] - mean) / std
    return X_norm

# ─────────────────────────────────────────────────────────────
# Stress profile extraction
# ─────────────────────────────────────────────────────────────

def extract_physio_reactivity(subj_dir: Path):
    """Extract HR and EDA reactivity from a STRESS session directory.

    Returns dict with keys: hr_baseline_mean, hr_tasks_mean, hr_reactivity,
    eda_baseline_mean, eda_tasks_mean, eda_reactivity. Returns None if data
    is insufficient.
    """
    result = {}

    hr_path = subj_dir / "HR.csv"
    if hr_path.exists():
        _, fs_hr, hr_data = load_empatica_csv(hr_path)
        hr_vals = hr_data.flatten()
        if np.isnan(hr_vals).any():
            median_hr = np.nanmedian(hr_vals)
            hr_vals = np.where(np.isnan(hr_vals), median_hr, hr_vals)
    else:
        hr_vals = None
        fs_hr = 1.0

    eda_path = subj_dir / "EDA.csv"
    if eda_path.exists():
        _, fs_eda, eda_data = load_empatica_csv(eda_path)
        eda_vals = eda_data.flatten()
    else:
        eda_vals = None
        fs_eda = 4.0

    tags_path = subj_dir / "tags.csv"
    if not tags_path.exists():
        return None

    acc_path = subj_dir / "ACC.csv"
    if acc_path.exists():
        start_acc, _, _ = load_empatica_csv(acc_path)
    else:
        return None

    tags = load_tags(tags_path)
    if len(tags) < 2:
        return None

    tags_sec = np.array([(t - start_acc).total_seconds() for t in tags])
    tags_sec = tags_sec[tags_sec >= 0]
    if len(tags_sec) < 2:
        return None

    baseline_end = tags_sec[0] if len(tags_sec) > 0 else 60.0
    baseline_start = max(0, baseline_end - 120)
    task_segments = []
    for i in range(len(tags_sec) - 1):
        if i % 2 == 0:
            task_segments.append((tags_sec[i], tags_sec[i + 1]))

    def mean_in_range(signal, fs, t0, t1):
        i0 = max(0, int(t0 * fs))
        i1 = min(len(signal), int(t1 * fs))
        if i1 <= i0:
            return np.nan
        return np.nanmean(signal[i0:i1])

    if hr_vals is not None and len(hr_vals) > 0:
        hr_bl = mean_in_range(hr_vals, fs_hr, baseline_start, baseline_end)
        hr_task_vals = [mean_in_range(hr_vals, fs_hr, t0, t1) for t0, t1 in task_segments]
        hr_tasks = np.nanmean([v for v in hr_task_vals if np.isfinite(v)]) if hr_task_vals else np.nan
        result["hr_baseline_mean"] = hr_bl
        result["hr_tasks_mean"] = hr_tasks
        result["hr_reactivity"] = (
            hr_tasks - hr_bl if np.isfinite(hr_bl) and np.isfinite(hr_tasks) else np.nan
        )
    else:
        result.update(hr_baseline_mean=np.nan, hr_tasks_mean=np.nan, hr_reactivity=np.nan)

    if eda_vals is not None and len(eda_vals) > 0:
        eda_bl = mean_in_range(eda_vals, fs_eda, baseline_start, baseline_end)
        eda_task_vals = [mean_in_range(eda_vals, fs_eda, t0, t1) for t0, t1 in task_segments]
        eda_tasks = np.nanmean([v for v in eda_task_vals if np.isfinite(v)]) if eda_task_vals else np.nan
        result["eda_baseline_mean"] = eda_bl
        result["eda_tasks_mean"] = eda_tasks
        result["eda_reactivity"] = (
            eda_tasks - eda_bl if np.isfinite(eda_bl) and np.isfinite(eda_tasks) else np.nan
        )
    else:
        result.update(eda_baseline_mean=np.nan, eda_tasks_mean=np.nan, eda_reactivity=np.nan)

    return result


def build_stress_profiles(physionet_root: Path, wsd_root: Path, exercise_pids) -> dict:
    """Build z-scored stress profiles for all exercise subjects.

    Args:
        physionet_root: path to Wearable_Dataset/ (contains STRESS/ subdir)
        wsd_root:       path to the WSD dataset root (contains Stress_Level_v1.csv etc.)
        exercise_pids:  iterable of pid strings like 'physionet_S01'

    Returns:
        stress_profiles: dict mapping pid → np.array(len(PROFILE_FEATURES),) z-scored
    """
    stress_dir = physionet_root / "STRESS"

    # ── Step 1: Self-reported stress ──
    sl_v1 = pd.read_csv(wsd_root / "Stress_Level_v1.csv", index_col=0)
    sl_v2 = pd.read_csv(wsd_root / "Stress_Level_v2.csv", index_col=0)

    rest_cols = {"Baseline", "First Rest", "Second Rest"}
    task_cols_v1 = [c for c in sl_v1.columns if c not in rest_cols]
    task_cols_v2 = [c for c in sl_v2.columns if c not in rest_cols]

    self_report = {}
    for sid in sl_v1.index:
        row = sl_v1.loc[sid]
        baseline = row["Baseline"]
        task_vals = row[task_cols_v1].values
        peak = np.nanmax(task_vals)
        self_report[sid] = {
            "sl_baseline": baseline,
            "sl_peak": peak,
            "sl_reactivity": peak - baseline,
            "sl_mean_tasks": np.nanmean(task_vals),
        }
    for sid in sl_v2.index:
        row = sl_v2.loc[sid]
        baseline = row["Baseline"]
        task_vals = row[task_cols_v2].values
        peak = np.nanmax(task_vals)
        self_report[sid] = {
            "sl_baseline": baseline,
            "sl_peak": peak,
            "sl_reactivity": peak - baseline,
            "sl_mean_tasks": np.nanmean(task_vals),
        }
    print(f"   Self-reported stress: {len(self_report)} subjects")

    # ── Step 2: Physiological reactivity ──
    physio_react = {}
    for subj_dir in sorted(stress_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        subj_id = subj_dir.name
        base_id = subj_id.split("_")[0]
        if base_id in STRESS_EXCLUDE_PROFILE:
            continue
        if base_id == "f14":
            if subj_id == "f14_a":
                continue
            base_id = "f14"
        react = extract_physio_reactivity(subj_dir)
        if react is not None:
            physio_react[base_id] = react
    print(f"   Physiological reactivity: {len(physio_react)} subjects")

    # ── Step 3: Demographics ──
    subj_info = pd.read_csv(wsd_root / "subject-info.csv", index_col=0)
    subj_info.index = subj_info.index.str.strip()
    demographics = {}
    for sid in subj_info.index:
        if not isinstance(sid, str) or sid.startswith("Reference") or sid == "":
            continue
        row = subj_info.loc[sid]
        gender = 0 if str(row.get("Gender", "")).strip().lower() == "m" else 1
        try:
            age = float(row["Age"])
        except (ValueError, TypeError):
            age = np.nan
        try:
            height = float(str(row["Height (cm)"]).replace(",", "."))
            weight = float(str(row["Weight (kg)"]).replace(",", "."))
            bmi = weight / (height / 100) ** 2
        except (ValueError, TypeError, ZeroDivisionError):
            bmi = np.nan
        demographics[sid] = {"age": age, "bmi": bmi, "gender": gender}
    print(f"   Demographics: {len(demographics)} subjects")

    # ── Step 4: Assemble & z-score ──
    exercise_sids = set()
    for pid in exercise_pids:
        sid = str(pid).replace("physionet_", "")
        exercise_sids.add(sid)
    print(f"   Exercise subjects: {len(exercise_sids)}")

    profiles_raw = {}
    for sid in sorted(exercise_sids):
        profile = np.full(len(PROFILE_FEATURES), np.nan, dtype=np.float64)
        if sid in self_report:
            for i, feat in enumerate(PROFILE_FEATURES[:4]):
                profile[i] = self_report[sid].get(feat, np.nan)
        if sid in physio_react:
            for i, feat in enumerate(PROFILE_FEATURES[4:10]):
                profile[i + 4] = physio_react[sid].get(feat, np.nan)
        if sid in demographics:
            profile[10] = demographics[sid]["age"]
            profile[11] = demographics[sid]["bmi"]
            profile[12] = demographics[sid]["gender"]
        profiles_raw[sid] = profile

    profiles_matrix = np.array([profiles_raw[sid] for sid in sorted(profiles_raw.keys())])
    for j in range(profiles_matrix.shape[1]):
        col = profiles_matrix[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            profiles_matrix[nan_mask, j] = median_val
            print(f"   Imputed {PROFILE_FEATURES[j]}: {nan_mask.sum()} NaN → median={median_val:.2f}")

    profile_means = profiles_matrix.mean(axis=0)
    profile_stds = profiles_matrix.std(axis=0) + 1e-8
    profiles_matrix_z = (profiles_matrix - profile_means) / profile_stds

    stress_profiles = {}
    for i, sid in enumerate(sorted(profiles_raw.keys())):
        pid = f"physionet_{sid}"
        stress_profiles[pid] = profiles_matrix_z[i].astype(np.float32)

    print(f"✅ Stress profiles ready: {len(stress_profiles)} subjects, {len(PROFILE_FEATURES)} features")
    return stress_profiles

# ─────────────────────────────────────────────────────────────
# NeuroKit2 feature extraction
# ─────────────────────────────────────────────────────────────

def _safe_stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    return float(np.nanmean(x)), float(np.nanstd(x))


def _mean_where(mask, values):
    vals = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask).astype(bool)
    if vals.size == 0 or m.size == 0:
        return 0.0
    m = m[:len(vals)]
    valid = m & np.isfinite(vals)
    if not np.any(valid):
        return 0.0
    return float(np.nanmean(vals[valid]))


def extract_hrv_features_from_hr(hr_signal, sampling_rate=20):
    hr = np.asarray(hr_signal, dtype=np.float64)
    hr = hr[np.isfinite(hr)]
    hr = hr[hr > 1.0]
    if hr.size < 8:
        return np.zeros(len(HRV_FEATURES), dtype=np.float32)

    ibi_ms = 60000.0 / np.clip(hr, 1.0, None)
    ibi_ms = ibi_ms[np.isfinite(ibi_ms)]
    ibi_ms = ibi_ms[(ibi_ms > 300.0) & (ibi_ms < 2000.0)]
    if ibi_ms.size < 5:
        return np.zeros(len(HRV_FEATURES), dtype=np.float32)

    if nk is not None:
        try:
            peaks = nk.intervals_to_peaks(ibi_ms, sampling_rate=1000)
            hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
            hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False, normalize=True)
            feature_map = {
                "max_ibi":      float(hrv_time.get("HRV_MaxNN", np.nan).iloc[0]),
                "min_ibi":      float(hrv_time.get("HRV_MinNN", np.nan).iloc[0]),
                "mean_ibi":     float(hrv_time.get("HRV_MeanNN", np.nan).iloc[0]),
                "hr_mean_ibi":  float(hrv_time.get("HRV_MeanHR", np.nan).iloc[0]),
                "pnn20":        float(hrv_time.get("HRV_pNN20", np.nan).iloc[0]),
                "pnn50":        float(hrv_time.get("HRV_pNN50", np.nan).iloc[0]),
                "rmssd":        float(hrv_time.get("HRV_RMSSD", np.nan).iloc[0]),
                "sdnn":         float(hrv_time.get("HRV_SDNN", np.nan).iloc[0]),
                "total_power":  float(hrv_freq.get("HRV_TP", np.nan).iloc[0]),
                "ratio":        float(hrv_freq.get("HRV_LFHF", np.nan).iloc[0]),
                "VLF_power":    float(hrv_freq.get("HRV_VLF", np.nan).iloc[0]),
                "VLF_peak":     float(hrv_freq.get("HRV_VLF_Peak", np.nan).iloc[0]),
                "LF_power":     float(hrv_freq.get("HRV_LF", np.nan).iloc[0]),
                "LF_peak":      float(hrv_freq.get("HRV_LF_Peak", np.nan).iloc[0]),
                "LH_n":         float(hrv_freq.get("HRV_LFn", np.nan).iloc[0]),
                "HF_power":     float(hrv_freq.get("HRV_HF", np.nan).iloc[0]),
                "HF_peak":      float(hrv_freq.get("HRV_HF_Peak", np.nan).iloc[0]),
                "HF_n":         float(hrv_freq.get("HRV_HFn", np.nan).iloc[0]),
                "VHF_power":    float(hrv_freq.get("HRV_VHF", np.nan).iloc[0]),
                "VHF_peak":     float(hrv_freq.get("HRV_VHF_Peak", np.nan).iloc[0]),
            }
            arr = np.array([feature_map[k] for k in HRV_FEATURES], dtype=np.float32)
            return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass

    # Fallback (no nk or exception)
    dibi = np.diff(ibi_ms)
    rmssd = np.sqrt(np.mean(dibi ** 2)) if dibi.size else 0.0
    sdnn = np.std(ibi_ms) if ibi_ms.size else 0.0
    pnn20 = np.mean(np.abs(dibi) > 20.0) if dibi.size else 0.0
    pnn50 = np.mean(np.abs(dibi) > 50.0) if dibi.size else 0.0
    fallback = np.array([
        np.max(ibi_ms), np.min(ibi_ms), np.mean(ibi_ms), np.mean(hr),
        pnn20, pnn50, rmssd, sdnn,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    return np.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)


def extract_eda_features(eda_signal, sampling_rate=20):
    eda = np.asarray(eda_signal, dtype=np.float64)
    if eda.size == 0 or not np.isfinite(eda).any():
        return np.zeros(len(EDA_FEATURES), dtype=np.float32)
    eda = np.nan_to_num(eda, nan=float(
        np.nanmedian(eda[np.isfinite(eda)]) if np.isfinite(eda).any() else 0.0
    ))

    tonic = eda.copy()
    phasic = np.zeros_like(eda)
    peaks = np.zeros_like(eda)
    scr_amp = scr_height = scr_risetime = scr_recovery = np.zeros_like(eda)

    if nk is not None:
        try:
            signals, _ = nk.eda_process(eda, sampling_rate=sampling_rate)
            tonic = signals["EDA_Tonic"].values.astype(np.float64)
            phasic = signals["EDA_Phasic"].values.astype(np.float64)
            peaks = signals.get("SCR_Peaks", pd.Series(np.zeros_like(tonic))).values
            scr_amp = signals.get("SCR_Amplitude", pd.Series(np.zeros_like(tonic))).values
            scr_height = signals.get("SCR_Height", pd.Series(np.zeros_like(tonic))).values
            scr_risetime = signals.get("SCR_RiseTime", pd.Series(np.zeros_like(tonic))).values
            scr_recovery = signals.get("SCR_RecoveryTime", pd.Series(np.zeros_like(tonic))).values
        except Exception:
            pass

    mean_raw, std_raw = _safe_stats(eda)
    mean_tonic, std_tonic = _safe_stats(tonic)
    mean_phasic, std_phasic = _safe_stats(phasic)
    dt = np.diff(tonic) if tonic.size > 1 else np.array([0.0])
    tonic_ratio_down = float(np.mean(dt < 0)) if dt.size else 0.0
    tonic_ratio_up = float(np.mean(dt > 0)) if dt.size else 0.0
    duration_sec = max(eda.size / float(sampling_rate), 1e-6)
    peaks_density = float(np.nansum(peaks > 0) / duration_sec)

    out = np.array([
        mean_raw, std_raw, mean_tonic, std_tonic, mean_phasic, std_phasic,
        tonic_ratio_down, tonic_ratio_up,
        peaks_density,
        _mean_where(peaks > 0, scr_amp),
        _mean_where(peaks > 0, scr_height),
        _mean_where(peaks > 0, scr_risetime),
        _mean_where(peaks > 0, scr_recovery),
    ], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract_acc_features(acc_xyz):
    acc_xyz = np.asarray(acc_xyz, dtype=np.float64)
    if acc_xyz.ndim != 2 or acc_xyz.shape[0] == 0 or acc_xyz.shape[1] < 3:
        return np.zeros(len(ACC_FEATURES), dtype=np.float32)
    x, y, z = acc_xyz[:, 0], acc_xyz[:, 1], acc_xyz[:, 2]
    mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    dm = np.diff(mag) if mag.size > 1 else np.array([0.0])
    out = np.array([
        np.mean(x), np.std(x), np.mean(y), np.std(y), np.mean(z), np.std(z),
        np.mean(mag), np.std(mag),
        np.mean(dm < 0) if dm.size else 0.0,
        np.mean(dm > 0) if dm.size else 0.0,
    ], dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def extract_signal_features(x_imu_window, x_physio_window, sampling_rate=20):
    hrv = extract_hrv_features_from_hr(x_physio_window[:, 3], sampling_rate=sampling_rate)
    eda = extract_eda_features(x_physio_window[:, 1], sampling_rate=sampling_rate)
    acc = extract_acc_features(x_imu_window[:, :3])
    return np.concatenate([hrv, eda, acc]).astype(np.float32)


def build_profiles_all(
    X_imu_base, X_physio_base, y_base, subjects_base, has_physio_base,
    stress_profiles: dict, sampling_rate: int = 20,
):
    """Build the full profile matrix: [static stress profile (13) + NK2 features (43)].

    Returns a dict with keys:
        X_imu, X_physio, y_all, subjects, has_physio  — cleaned copies of inputs
        profiles_all        — np.array(N, 56), final profile matrix
        profiles_static     — np.array(N, 13)
        signal_features_z   — np.array(N, 43)
        sf_mean, sf_std     — normalization stats for signal features
        n_mapped, n_missing — mapping stats
    """
    X_imu = X_imu_base.copy()
    X_physio = X_physio_base.copy()
    y_all = y_base.copy()
    subjects = subjects_base.copy()
    has_physio = has_physio_base.copy()

    nan_imu = np.isnan(X_imu).sum()
    nan_physio = np.isnan(X_physio).sum()
    if nan_imu > 0:
        X_imu = np.nan_to_num(X_imu, nan=0.0)
    if nan_physio > 0:
        X_physio = np.nan_to_num(X_physio, nan=0.0)

    # Static stress profiles (subject-level → each window)
    profile_static_dim = len(PROFILE_FEATURES)
    profiles_static = np.zeros((len(y_all), profile_static_dim), dtype=np.float32)
    n_mapped, n_missing = 0, 0
    for i, pid in enumerate(subjects):
        if pid in stress_profiles:
            profiles_static[i] = stress_profiles[pid]
            n_mapped += 1
        else:
            n_missing += 1

    # NK2 window-level signal features
    signal_features = np.zeros((len(y_all), len(SIGNAL_FEATURES)), dtype=np.float32)
    for i in tqdm(range(len(y_all)), desc="NK2 feature extraction"):
        signal_features[i] = extract_signal_features(X_imu[i], X_physio[i], sampling_rate)

    signal_features = np.nan_to_num(signal_features, nan=0.0, posinf=0.0, neginf=0.0)
    sf_mean = signal_features.mean(axis=0)
    sf_std = signal_features.std(axis=0) + 1e-8
    signal_features_z = (signal_features - sf_mean) / sf_std

    profiles_all = np.concatenate([profiles_static, signal_features_z], axis=1).astype(np.float32)

    print(f"✅ Profiles assembled: static={profiles_static.shape}, signal={signal_features_z.shape}, final={profiles_all.shape}")
    print(f"   Mapped: {n_mapped}, missing: {n_missing} | NaN fixed: IMU={nan_imu}, Physio={nan_physio}")

    return {
        "X_imu": X_imu,
        "X_physio": X_physio,
        "y_all": y_all,
        "subjects": subjects,
        "has_physio": has_physio,
        "profiles_all": profiles_all,
        "profiles_static": profiles_static,
        "signal_features_z": signal_features_z,
        "sf_mean": sf_mean,
        "sf_std": sf_std,
        "n_mapped": n_mapped,
        "n_missing": n_missing,
    }

# ─────────────────────────────────────────────────────────────
# Subject-level split
# ─────────────────────────────────────────────────────────────

def create_subject_split(subjects, y, test_size=0.2, val_size=0.2, seed=42):
    unique_sids = np.unique(subjects)
    sid_tv, sid_test = train_test_split(unique_sids, test_size=test_size, random_state=seed)
    val_rel = val_size / (1.0 - test_size)
    sid_train, sid_val = train_test_split(sid_tv, test_size=val_rel, random_state=seed)
    tr_idx = np.where(np.isin(subjects, sid_train))[0]
    va_idx = np.where(np.isin(subjects, sid_val))[0]
    te_idx = np.where(np.isin(subjects, sid_test))[0]
    for name, idx in [("Train", tr_idx), ("Val", va_idx), ("Test", te_idx)]:
        if len(np.unique(y[idx])) < 2:
            raise ValueError(f"{name} split has only one class")
    return tr_idx, va_idx, te_idx

# ─────────────────────────────────────────────────────────────
# Augmentation
# ─────────────────────────────────────────────────────────────

def add_gaussian_noise(x, sigma=0.05):
    return x + np.random.normal(0, sigma, x.shape).astype(x.dtype)


def time_warp(x, sigma=0.2, knots=4):
    T, C = x.shape
    tt = np.linspace(0, T - 1, knots + 2)
    warp = np.concatenate([[0], np.random.normal(0, sigma * T, knots), [0]])
    warp_fn = CubicSpline(tt, warp)
    t_orig = np.arange(T)
    t_warped = np.clip(t_orig + warp_fn(t_orig), 0, T - 1)
    return np.stack(
        [CubicSpline(t_orig, x[:, c])(t_warped) for c in range(C)], axis=-1
    ).astype(x.dtype)


def channel_dropout(x, p=0.2):
    x = x.copy()
    for c in range(x.shape[1]):
        if np.random.rand() < p:
            x[:, c] = 0.0
    return x


def magnitude_scale(x, lo=0.7, hi=1.3):
    return x * np.random.uniform(lo, hi, (1, x.shape[1])).astype(x.dtype)


def window_slice(x, crop_lo=0.6, crop_hi=0.9):
    T, C = x.shape
    crop = np.random.uniform(crop_lo, crop_hi)
    L = max(int(T * crop), 10)
    s = np.random.randint(0, T - L + 1)
    return sp_resample(x[s:s + L], T, axis=0).astype(x.dtype)


def time_reverse(x):
    return x[::-1].copy()


def imu_rotate(x, max_deg=20.0):
    def _rot():
        theta = np.deg2rad(np.random.uniform(-max_deg, max_deg))
        n = np.random.randn(3)
        n /= (np.linalg.norm(n) + 1e-8)
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    x = x.copy()
    x[:, :3] = x[:, :3] @ _rot().T
    if x.shape[1] >= 6:
        x[:, 3:6] = x[:, 3:6] @ _rot().T
    return x.astype(np.float32)


def permutation_segments(x, n_segments=4):
    T, C = x.shape
    seg_len = T // n_segments
    if seg_len < 2:
        return x
    segments = [x[i * seg_len:(i + 1) * seg_len] for i in range(n_segments)]
    remainder = x[n_segments * seg_len:]
    np.random.shuffle(segments)
    result = np.concatenate(segments + ([remainder] if len(remainder) else []), axis=0)
    return result.astype(x.dtype)


def frequency_mask(x, max_mask_ratio=0.15):
    T, C = x.shape
    x_out = x.copy()
    for c in range(C):
        freq = np.fft.rfft(x[:, c])
        n_freq = len(freq)
        mask_len = max(1, int(n_freq * max_mask_ratio))
        start = np.random.randint(0, max(1, n_freq - mask_len))
        freq[start:start + mask_len] = 0
        x_out[:, c] = np.fft.irfft(freq, n=T)
    return x_out.astype(x.dtype)


def compute_sample_difficulty(x_imu):
    per_channel_std = np.std(x_imu, axis=0)
    return float(np.mean(per_channel_std))


IMU_TRANSFORMS = [
    ("noise",     lambda x: add_gaussian_noise(x, sigma=np.random.uniform(0.03, 0.10))),
    ("time_warp", lambda x: time_warp(x, sigma=np.random.uniform(0.15, 0.35), knots=4)),
    ("scale",     lambda x: magnitude_scale(x, 0.7, 1.3)),
    ("crop",      lambda x: window_slice(x, 0.6, 0.9)),
    ("reverse",   lambda x: time_reverse(x)),
    ("rotate",    lambda x: imu_rotate(x, max_deg=20.0)),
    ("permute",   lambda x: permutation_segments(x, n_segments=np.random.randint(3, 6))),
    ("freq_mask", lambda x: frequency_mask(x, max_mask_ratio=0.15)),
]

PHYSIO_TRANSFORMS = [
    ("noise",     lambda x: add_gaussian_noise(x, sigma=np.random.uniform(0.02, 0.06))),
    ("scale",     lambda x: magnitude_scale(x, 0.85, 1.15)),
    ("time_warp", lambda x: time_warp(x, sigma=0.1, knots=3)),
]


def augment_sample(x_imu, x_physio):
    """Strong augmentation: 3-5 techniques for IMU, 1-2 for Physio."""
    difficulty = compute_sample_difficulty(x_imu)
    border_score = 1.0 - abs(difficulty - 0.8) / max(difficulty + 0.3, 1e-6)
    border_score = np.clip(border_score, 0, 1)
    n_imu_aug = 3 + int(2 * border_score)
    n_physio_aug = 1 + int(border_score >= 0.5)

    imu_choices = np.random.choice(
        len(IMU_TRANSFORMS), size=min(n_imu_aug, len(IMU_TRANSFORMS)), replace=False
    )
    for idx in imu_choices:
        _, fn = IMU_TRANSFORMS[idx]
        x_imu = fn(x_imu)
    x_imu = channel_dropout(x_imu, 0.2)

    physio_choices = np.random.choice(
        len(PHYSIO_TRANSFORMS), size=min(n_physio_aug, len(PHYSIO_TRANSFORMS)), replace=False
    )
    for idx in physio_choices:
        _, fn = PHYSIO_TRANSFORMS[idx]
        x_physio = fn(x_physio)

    return x_imu, x_physio

# ─────────────────────────────────────────────────────────────
# SMOTE
# ─────────────────────────────────────────────────────────────

def apply_smote_dual_branch(
    X_imu, X_physio, y, has_physio_arr, profiles_arr=None, k_neighbors=5, random_state=42
):
    """SMOTE in the combined IMU+Physio+Profile space."""
    N, T, C_imu = X_imu.shape
    C_physio = X_physio.shape[2]

    flat_imu = X_imu.reshape(N, -1)
    flat_physio = X_physio.reshape(N, -1)
    parts = [flat_imu, flat_physio]
    if profiles_arr is not None:
        parts.append(profiles_arr.reshape(N, -1))
    X_flat = np.hstack(parts)

    class_counts = np.bincount(y.astype(int))
    min_class_count = class_counts.min()
    k = min(k_neighbors, min_class_count - 1)
    if k < 1:
        print("   ⚠️ SMOTE: недостаточно примеров для k_neighbors, пропуск")
        return X_imu, X_physio, y, has_physio_arr, profiles_arr

    smote = SMOTE(k_neighbors=k, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y.astype(int))

    N_new = len(y_resampled)
    offset = 0
    imu_flat = X_resampled[:, offset:offset + T * C_imu]; offset += T * C_imu
    physio_flat = X_resampled[:, offset:offset + T * C_physio]; offset += T * C_physio
    X_imu_new = imu_flat.reshape(N_new, T, C_imu).astype(np.float32)
    X_physio_new = physio_flat.reshape(N_new, T, C_physio).astype(np.float32)

    if profiles_arr is not None:
        P = profiles_arr.shape[1]
        profiles_new = X_resampled[:, offset:offset + P].astype(np.float32)
    else:
        profiles_new = None

    hp_new = np.zeros(N_new, dtype=bool)
    hp_new[:N] = has_physio_arr
    for cls in [0, 1]:
        cls_mask_orig = y.astype(int) == cls
        if cls_mask_orig.any():
            physio_ratio = has_physio_arr[cls_mask_orig].mean()
            cls_mask_new = (y_resampled == cls) & (np.arange(N_new) >= N)
            hp_new[cls_mask_new] = physio_ratio > 0.5

    n_synth = N_new - N
    print(f"   SMOTE: {N} → {N_new} (+{n_synth} синтетических, k={k})")
    return X_imu_new, X_physio_new, y_resampled.astype(np.int64), hp_new, profiles_new

# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────

class FatigueDataset(Dataset):
    """Returns (x_imu, x_physio, has_physio, profile, y) per sample — 5-tuple."""

    def __init__(self, X_imu, X_physio, y, has_physio, profiles=None, augment=False, profile_dim=None):
        self.X_imu = X_imu.astype(np.float32)
        self.X_physio = X_physio.astype(np.float32)
        self.y = y.astype(np.float32)
        self.has_physio = has_physio.astype(np.float32)
        if profiles is not None:
            self.profiles = profiles.astype(np.float32)
        else:
            dim = int(profile_dim if profile_dim is not None else len(PROFILE_FEATURES))
            self.profiles = np.zeros((len(y), dim), dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        xi, xp = self.X_imu[idx].copy(), self.X_physio[idx].copy()
        if self.augment:
            xi, xp = augment_sample(xi, xp)
        return (
            torch.FloatTensor(xi),
            torch.FloatTensor(xp),
            torch.tensor(self.has_physio[idx], dtype=torch.float32),
            torch.FloatTensor(self.profiles[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )
