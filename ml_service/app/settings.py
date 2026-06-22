from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ServiceSettings(BaseModel):
    name: str = "AthleteFatigueTracker ML Service"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class ModelSettings(BaseModel):
    version: str = "fatiguewristnet-v8.1"
    path: str = "../res/results_v8_stress/results_test/best_model_v8_best.pth"
    normalization: str = "global-v8"
    threshold: float = 0.5
    stress_profile_size: int = 12


class RuntimeSettings(BaseModel):
    use_stub_inference: bool = True
    torch_device: str = "cpu"


class ApiThresholds(BaseModel):
    normal: float = 0.2
    mild: float = 0.45
    moderate: float = 0.75


class ApiSettings(BaseModel):
    default_category_thresholds: ApiThresholds = Field(default_factory=ApiThresholds)


class AppSettings(BaseModel):
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)
    settings_file: Path

    @property
    def model_path(self) -> Path:
        configured = Path(self.model.path)
        if configured.is_absolute():
            return configured
        return (self.settings_file.parent / configured).resolve()


def load_settings(settings_file: str | Path | None = None) -> AppSettings:
    current_file = Path(settings_file or Path(__file__).resolve().parent.parent / "settings.yaml").resolve()
    with current_file.open("r", encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    payload["settings_file"] = current_file
    return AppSettings.model_validate(payload)


settings = load_settings()