from __future__ import annotations

import base64
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TensorPayload(BaseModel):
    shape: list[int]
    data: str

    @field_validator("shape")
    @classmethod
    def validate_shape_len(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("shape must not be empty")
        if any(item <= 0 for item in value):
            raise ValueError("shape values must be positive")
        return value

    @field_validator("data")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError("data must be valid base64") from exc
        return value


class PredictRequest(BaseModel):
    imu: TensorPayload
    physio: TensorPayload
    hasPhysio: bool = Field(alias="hasPhysio")
    stressProfile: TensorPayload | None = Field(default=None, alias="stressProfile")
    subjectId: str | None = Field(default=None, alias="subjectId")
    normalization: str | None = None
    requestId: str | None = Field(default=None, alias="requestId")

    @model_validator(mode="after")
    def validate_shapes(self) -> "PredictRequest":
        if self.imu.shape != [100, 6]:
            raise ValueError(f"imu.shape must be [100, 6], got {self.imu.shape}")
        if self.physio.shape != [100, 4]:
            raise ValueError(f"physio.shape must be [100, 4], got {self.physio.shape}")
        if self.stressProfile is not None and len(self.stressProfile.shape) != 1:
            raise ValueError("stressProfile.shape must be 1-dimensional")
        return self


class PredictResponse(BaseModel):
    fatigueDegree: float = Field(alias="fatigueDegree")
    category: Literal["NORMAL", "MILD", "MODERATE", "SEVERE"]
    threshold: float
    modelVersion: str = Field(alias="modelVersion")
    latencyMs: int = Field(alias="latencyMs")
    requestId: str | None = Field(default=None, alias="requestId")


class PredictBatchRequest(BaseModel):
    items: list[PredictRequest]


class PredictBatchResponse(BaseModel):
    items: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    modelVersion: str = Field(alias="modelVersion")
    stubInference: bool = Field(alias="stubInference")
    modelPathExists: bool = Field(alias="modelPathExists")