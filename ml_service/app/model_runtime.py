from __future__ import annotations

import base64
import hashlib
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn as nn

from .schemas import PredictRequest, PredictResponse
from .settings import AppSettings


@dataclass(slots=True)
class RuntimeStatus:
    model_path: Path
    model_path_exists: bool
    using_stub: bool


class InferenceEngine(Protocol):
    @property
    def status(self) -> RuntimeStatus:
        ...

    def predict(self, request: PredictRequest) -> PredictResponse:
        ...


class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.score = nn.Sequential(
            nn.Conv1d(channels, n_heads, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(n_heads, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        att = torch.softmax(self.score(x), dim=-1)  # (B,1,T)
        x_h = x.view(b, self.n_heads, self.head_dim, t)
        att_h = att.expand(-1, self.n_heads, -1).unsqueeze(2)  # (B,H,1,T)
        out = (x_h * att_h).sum(dim=-1)
        return out.reshape(b, c)


class IMUEncoderWithAttention(nn.Module):
    def __init__(self, in_channels: int = 6, out_channels: int = 16, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
        self.attention = MultiHeadTemporalAttention(out_channels, n_heads=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.attention(x)


class PhysioEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 8, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).squeeze(-1)


class FatigueWristNetV81(nn.Module):
    def __init__(self, profile_dim: int = 13):
        super().__init__()
        self.profile_dim = profile_dim
        self.imu_encoder = IMUEncoderWithAttention(6, 16, 0.3)
        self.physio_encoder = PhysioEncoder(4, 8, 0.3)
        feat_dim = 16 + 8 + profile_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 18),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(18, 1),
        )

    def forward(
        self,
        x_imu: torch.Tensor,
        x_physio: torch.Tensor,
        has_physio: torch.Tensor | None = None,
        profile: torch.Tensor | None = None,
    ) -> torch.Tensor:
        imu_feat = self.imu_encoder(x_imu)
        physio_feat = self.physio_encoder(x_physio)
        if has_physio is not None:
            physio_feat = physio_feat * has_physio.unsqueeze(-1)
        if profile is None:
            profile = torch.zeros((x_imu.size(0), self.profile_dim), device=x_imu.device)
        x = torch.cat([imu_feat, physio_feat, profile], dim=1)
        return self.classifier(x).squeeze(-1)


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


class RealInferenceEngine:
    def __init__(self, app_settings: AppSettings):
        self._settings = app_settings
        self._device = torch.device(app_settings.runtime.torch_device)
        self._model = FatigueWristNetV81(profile_dim=app_settings.model.stress_profile_size).to(self._device)

        ckpt = torch.load(app_settings.model_path, map_location=self._device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self._model.load_state_dict(state_dict)
        self._model.eval()

        self._status = RuntimeStatus(
            model_path=app_settings.model_path,
            model_path_exists=app_settings.model_path.exists(),
            using_stub=False,
        )

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    def _decode_f32(self, data_b64: str, shape: list[int]) -> np.ndarray:
        raw = base64.b64decode(data_b64)
        arr = np.frombuffer(raw, dtype=np.float32)
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"Invalid tensor payload: expected {expected} float32 values, got {arr.size}")
        return arr.reshape(shape)

    def predict(self, request: PredictRequest) -> PredictResponse:
        started = time.perf_counter()

        x_imu = self._decode_f32(request.imu.data, request.imu.shape)
        x_physio = self._decode_f32(request.physio.data, request.physio.shape)

        profile_size = self._settings.model.stress_profile_size
        if request.stressProfile is not None:
            profile = self._decode_f32(request.stressProfile.data, request.stressProfile.shape)
            profile = profile.reshape(-1)
            if profile.size != profile_size:
                raise ValueError(
                    f"stressProfile size must be {profile_size}, got {profile.size}. "
                    "Update settings.model.stress_profile_size if needed."
                )
        else:
            profile = np.zeros((profile_size,), dtype=np.float32)

        xi = torch.from_numpy(x_imu).unsqueeze(0).to(self._device)
        xp = torch.from_numpy(x_physio).unsqueeze(0).to(self._device)
        hp = torch.tensor([1.0 if request.hasPhysio else 0.0], dtype=torch.float32, device=self._device)
        prof = torch.from_numpy(profile.astype(np.float32)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(xi, xp, hp, prof)
            score = float(torch.sigmoid(logits).item())

        latency_ms = max(1, int((time.perf_counter() - started) * 1000))
        return PredictResponse(
            fatigueDegree=round(score, 3),
            category=self._category(score),
            threshold=self._settings.model.threshold,
            modelVersion=self._settings.model.version,
            latencyMs=latency_ms,
            requestId=request.requestId,
        )

    def _category(self, value: float) -> str:
        thresholds = self._settings.api.default_category_thresholds
        if value < thresholds.normal:
            return "NORMAL"
        if value < thresholds.mild:
            return "MILD"
        if value < thresholds.moderate:
            return "MODERATE"
        return "SEVERE"


def create_inference_engine(app_settings: AppSettings) -> InferenceEngine:
    if app_settings.runtime.use_stub_inference:
        return StubInferenceEngine(app_settings)
    return RealInferenceEngine(app_settings)