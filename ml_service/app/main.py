from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .model_runtime import StubInferenceEngine
from .schemas import HealthResponse, PredictBatchRequest, PredictBatchResponse, PredictRequest, PredictResponse
from .settings import settings

app = FastAPI(title=settings.service.name, version=settings.model.version)
engine = StubInferenceEngine(settings)


def _problem(status: int, title: str, detail: str, instance: str, extra: dict | None = None) -> JSONResponse:
    payload = {
        "type": f"https://errors.afc.local/{title.lower().replace(' ', '-')}",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": instance,
    }
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=status, content=payload)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else {"msg": "invalid request"}
    return _problem(
        status=400,
        title="Invalid window payload",
        detail=str(first_error.get("msg", "invalid request")),
        instance="/ml/v1/predict",
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(_request, exc: ValidationError):
    first_error = exc.errors()[0] if exc.errors() else {"msg": "invalid request"}
    return _problem(
        status=400,
        title="Invalid window payload",
        detail=str(first_error.get("msg", "invalid request")),
        instance="/ml/v1/predict",
    )


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok",
        modelVersion=settings.model.version,
        stubInference=engine.status.using_stub,
        modelPathExists=engine.status.model_path_exists,
    )


@app.get("/readyz", response_model=HealthResponse)
def readyz() -> HealthResponse:
    status = "ready" if engine.status.model_path_exists or engine.status.using_stub else "not_ready"
    return HealthResponse(
        status=status,
        modelVersion=settings.model.version,
        stubInference=engine.status.using_stub,
        modelPathExists=engine.status.model_path_exists,
    )


@app.post("/ml/v1/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    normalization = payload.normalization or settings.model.normalization
    if normalization != settings.model.normalization:
        raise HTTPException(
            status_code=400,
            detail=f"normalization must be {settings.model.normalization}, got {normalization}",
        )
    return engine.predict(payload)


@app.post("/ml/v1/predict-batch", response_model=PredictBatchResponse)
def predict_batch(payload: PredictBatchRequest) -> PredictBatchResponse:
    if not payload.items:
        raise HTTPException(status_code=400, detail="items must not be empty")
    return PredictBatchResponse(items=[engine.predict(item) for item in payload.items])
