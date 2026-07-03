#!/usr/bin/env python3
"""
Build a compact personalization demo slice for mobile scenarios:
- normal
- aerobic exercise
- anaerobic exercise

Outputs:
1) NPZ slice with tensors and metadata
2) JSON payloads compatible with ml_service /ml/v1/predict
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_dataset import (
    PHYSIONET_FATIGUE_RATIO,
    PHYSIONET_STRIDE_SEC,
    PHYSIONET_WINDOW_SEC,
    load_physionet,
    normalize_per_subject,
)


def to_base64_f32(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return base64.b64encode(a.tobytes()).decode("ascii")


def evenly_spaced_indices(n: int, k: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.int64)
    if n <= k:
        return np.arange(n, dtype=np.int64)
    return np.linspace(0, n - 1, num=k, dtype=np.int64)


def pick_subject(aerobic: dict, anaerobic: dict, min_per_scenario: int) -> str:
    a_pids = np.unique(aerobic["pids"])
    an_pids = np.unique(anaerobic["pids"])
    candidates = sorted(set(a_pids).intersection(set(an_pids)))

    best_pid = ""
    best_score = -1

    for pid in candidates:
        a_mask = aerobic["pids"] == pid
        an_mask = anaerobic["pids"] == pid
        a_y = aerobic["y"][a_mask]
        an_y = anaerobic["y"][an_mask]

        n_normal = int((a_y == 0).sum())
        n_aero = int((a_y == 1).sum())
        n_anaero = int((an_y == 1).sum())

        if min(n_normal, n_aero, n_anaero) < min_per_scenario:
            continue

        score = n_normal + n_aero + n_anaero
        if score > best_score:
            best_score = score
            best_pid = str(pid)

    if not best_pid:
        raise RuntimeError(
            "No single subject has enough windows for all three scenarios. "
            "Try lowering --samples-per-scenario."
        )

    return best_pid


def select_scenario_rows(
    pid: str,
    aerobic: dict,
    anaerobic: dict,
    samples_per_scenario: int,
) -> dict[str, dict[str, np.ndarray]]:
    a_mask = aerobic["pids"] == pid
    an_mask = anaerobic["pids"] == pid

    a_X_imu = aerobic["X_imu"][a_mask]
    a_X_physio = aerobic["X_physio"][a_mask]
    a_y = aerobic["y"][a_mask]

    an_X_imu = anaerobic["X_imu"][an_mask]
    an_X_physio = anaerobic["X_physio"][an_mask]
    an_y = anaerobic["y"][an_mask]

    idx_normal = np.where(a_y == 0)[0]
    idx_aero = np.where(a_y == 1)[0]
    idx_anaero = np.where(an_y == 1)[0]

    sel_normal = idx_normal[evenly_spaced_indices(len(idx_normal), samples_per_scenario)]
    sel_aero = idx_aero[evenly_spaced_indices(len(idx_aero), samples_per_scenario)]
    sel_anaero = idx_anaero[evenly_spaced_indices(len(idx_anaero), samples_per_scenario)]

    return {
        "normal": {
            "X_imu": a_X_imu[sel_normal],
            "X_physio": a_X_physio[sel_normal],
            "y": a_y[sel_normal],
            "protocol": np.array(["AEROBIC"] * len(sel_normal)),
        },
        "aerobic_exercise": {
            "X_imu": a_X_imu[sel_aero],
            "X_physio": a_X_physio[sel_aero],
            "y": a_y[sel_aero],
            "protocol": np.array(["AEROBIC"] * len(sel_aero)),
        },
        "anaerobic_exercise": {
            "X_imu": an_X_imu[sel_anaero],
            "X_physio": an_X_physio[sel_anaero],
            "y": an_y[sel_anaero],
            "protocol": np.array(["ANAEROBIC"] * len(sel_anaero)),
        },
    }


def build_outputs(
    project_root: Path,
    out_npz: Path,
    out_json: Path,
    subject_id: str,
    scenarios: dict[str, dict[str, np.ndarray]],
    normalization: str,
) -> None:
    X_imu_all = []
    X_physio_all = []
    y_all = []
    pids_all = []
    has_physio_all = []
    scenario_all = []
    protocol_all = []
    payloads = []

    for scenario_name, block in scenarios.items():
        x_imu = block["X_imu"].astype(np.float32)
        x_physio = block["X_physio"].astype(np.float32)
        y = block["y"].astype(np.int8)
        protocol = block["protocol"].astype("U16")

        n = len(y)
        X_imu_all.append(x_imu)
        X_physio_all.append(x_physio)
        y_all.append(y)
        pids_all.append(np.array([subject_id] * n))
        has_physio_all.append(np.ones(n, dtype=bool))
        scenario_all.append(np.array([scenario_name] * n, dtype="U24"))
        protocol_all.append(protocol)

        for i in range(n):
            payloads.append(
                {
                    "scenario": scenario_name,
                    "protocol": str(protocol[i]),
                    "expectedLabel": int(y[i]),
                    "request": {
                        "imu": {
                            "shape": [100, 6],
                            "data": to_base64_f32(x_imu[i]),
                        },
                        "physio": {
                            "shape": [100, 4],
                            "data": to_base64_f32(x_physio[i]),
                        },
                        "hasPhysio": True,
                        "subjectId": subject_id,
                        "normalization": normalization,
                        "requestId": f"{scenario_name}-{i:03d}",
                    },
                }
            )

    X_imu_cat = np.concatenate(X_imu_all, axis=0)
    X_physio_cat = np.concatenate(X_physio_all, axis=0)
    y_cat = np.concatenate(y_all, axis=0)
    pids_cat = np.concatenate(pids_all, axis=0)
    has_physio_cat = np.concatenate(has_physio_all, axis=0)
    scenario_cat = np.concatenate(scenario_all, axis=0)
    protocol_cat = np.concatenate(protocol_all, axis=0)

    np.savez_compressed(
        out_npz,
        X_imu=X_imu_cat,
        X_physio=X_physio_cat,
        y=y_cat,
        pids=pids_cat,
        has_physio=has_physio_cat,
        scenario=scenario_cat,
        protocol=protocol_cat,
        source=np.array(["physionet"] * len(y_cat)),
    )

    summary = {
        "projectRoot": str(project_root),
        "subjectId": subject_id,
        "items": int(len(y_cat)),
        "normalization": normalization,
        "scenarios": {
            name: {
                "count": int(len(block["y"])),
                "positiveFatigue": int(block["y"].sum()),
                "negativeFatigue": int((block["y"] == 0).sum()),
            }
            for name, block in scenarios.items()
        },
        "payloads": payloads,
    }

    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build mobile personalization demo slice")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path",
    )
    parser.add_argument(
        "--physionet-dir",
        type=Path,
        default=Path(
            "data/raw/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1/Wearable_Dataset"
        ),
        help="Path to Wearable_Dataset directory (relative to project root by default)",
    )
    parser.add_argument(
        "--samples-per-scenario",
        type=int,
        default=25,
        help="How many windows to keep for each scenario",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="global-v8",
        help="Normalization tag to embed into payloads",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    physionet_dir = args.physionet_dir
    if not physionet_dir.is_absolute():
        physionet_dir = (project_root / physionet_dir).resolve()

    out_dir = project_root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "mobile_demo_personalization_slice.npz"
    out_json = out_dir / "mobile_demo_personalization_payloads.json"

    print("Loading PhysioNet AEROBIC...")
    aerobic = load_physionet(
        data_dir=physionet_dir,
        protocols=["AEROBIC"],
        window_sec=PHYSIONET_WINDOW_SEC,
        stride_sec=PHYSIONET_STRIDE_SEC,
        fatigue_ratio=PHYSIONET_FATIGUE_RATIO,
    )

    print("Loading PhysioNet ANAEROBIC...")
    anaerobic = load_physionet(
        data_dir=physionet_dir,
        protocols=["ANAEROBIC"],
        window_sec=PHYSIONET_WINDOW_SEC,
        stride_sec=PHYSIONET_STRIDE_SEC,
        fatigue_ratio=PHYSIONET_FATIGUE_RATIO,
    )

    # Keep consistency with training setup: per-subject z-score normalization
    aerobic["X_imu"] = normalize_per_subject(aerobic["X_imu"], aerobic["pids"])
    aerobic["X_physio"] = normalize_per_subject(aerobic["X_physio"], aerobic["pids"])
    anaerobic["X_imu"] = normalize_per_subject(anaerobic["X_imu"], anaerobic["pids"])
    anaerobic["X_physio"] = normalize_per_subject(anaerobic["X_physio"], anaerobic["pids"])

    pid = pick_subject(aerobic, anaerobic, min_per_scenario=args.samples_per_scenario)
    print(f"Selected subject for personalization demo: {pid}")

    scenarios = select_scenario_rows(
        pid=pid,
        aerobic=aerobic,
        anaerobic=anaerobic,
        samples_per_scenario=args.samples_per_scenario,
    )

    build_outputs(
        project_root=project_root,
        out_npz=out_npz,
        out_json=out_json,
        subject_id=pid,
        scenarios=scenarios,
        normalization=args.normalization,
    )

    print("Done.")
    print(f"NPZ:  {out_npz}")
    print(f"JSON: {out_json}")


if __name__ == "__main__":
    main()
