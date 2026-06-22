from __future__ import annotations

import base64
import hashlib
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from .schemas import PredictRequest, PredictResponse
from .settings import AppSettings


@dataclass(slots=True)
class RuntimeStatus:
    model_path: Path
    model_path_exists: bool
    using_stub: bool


class StubInferenceEngine:
    def __init__(self, app_settings: AppSettings):
        self._settings = app_settings
        self._status = RuntimeStatus(
            model_path=app_settings.model_path,
            model_path_exists=app_settings.model_path.exists(),
            using_stub=app_settings.runtime.use_stub_inference,
        )

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    def predict(self, request: PredictRequest) -> PredictResponse:
        started = time.perf_counter()

        imu_bytes = base64.b64decode(request.imu.data)
        physio_bytes = base64.b64decode(request.physio.data)
        profile_bytes = base64.b64decode(request.stressProfile.data) if request.stressProfile else b""

        score = self._score_bytes(imu_bytes, physio_bytes, profile_bytes, request.hasPhysio)
        latency_ms = max(1, int((time.perf_counter() - started) * 1000))

        return PredictResponse(
            fatigueDegree=score,
            category=self._category(score),
            threshold=self._settings.model.threshold,
            modelVersion=self._settings.model.version,
            latencyMs=latency_ms,
            requestId=request.requestId,
        )

    def _score_bytes(self, imu_bytes: bytes, physio_bytes: bytes, profile_bytes: bytes, has_physio: bool) -> float:
        digest = hashlib.sha256(imu_bytes + physio_bytes + profile_bytes + (b"1" if has_physio else b"0")).digest()
        raw_value = struct.unpack("<I", digest[:4])[0]
        score = raw_value / 4294967295.0
        return round(score, 3)

    def _category(self, value: float) -> str:
        thresholds = self._settings.api.default_category_thresholds
        if value < thresholds.normal:
            return "NORMAL"
        if value < thresholds.mild:
            return "MILD"
        if value < thresholds.moderate:
            return "MODERATE"
        return "SEVERE"