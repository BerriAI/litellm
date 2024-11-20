import json
from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

from fastapi import APIRouter, HTTPException, Request

from .data import get_params, get_provider_keys, provider_data, raw_data

T = TypeVar("T")


@dataclass
class RagaApiResponse(Generic[T]):
    success: bool
    data: T
    message: Optional[str]


router = APIRouter(prefix="/raga/internal")


@router.get("/providers")
async def get_providers() -> RagaApiResponse:
    return RagaApiResponse(True, {"providers": list(provider_data.keys())}, None)


@router.post("/providers/keys")
async def get_keys_by_provider(request: Request) -> RagaApiResponse:
    body = json.loads((await request.body()).decode())
    provider_name = body.get("provider")
    if provider_name is None:
        raise HTTPException(status_code=400, detail="provider is required")

    if provider_name not in provider_data:
        raise HTTPException(status_code=400, detail="provider not supported")

    return RagaApiResponse(True, {"keys": provider_data.get(provider_name, {}).get("keys")}, None)


@router.post("/providers/keys/native")
async def get_native_keys_by_provider(request: Request) -> RagaApiResponse:
    body = json.loads((await request.body()).decode())
    provider_name = body.get("provider")
    if provider_name is None:
        raise HTTPException(status_code=400, detail="provider is required")

    return RagaApiResponse(True, {"keys": get_provider_keys(provider_name)}, None)


@router.post("/providers/models")
async def get_model_by_provider(request: Request) -> RagaApiResponse:
    body = json.loads((await request.body()).decode())
    provider_name = body.get("provider")
    if provider_name is None:
        raise HTTPException(status_code=400, detail="provider is required")

    if provider_name not in provider_data:
        raise HTTPException(status_code=400, detail="provider not supported")

    return RagaApiResponse(True, {"models": list(provider_data.get(provider_name, {}).get("models").keys())}, None)


@router.post("/providers/models/params")
async def get_model_params_by_model(request: Request) -> RagaApiResponse:
    body = json.loads((await request.body()).decode())
    provider_name = body.get("provider")
    if provider_name is None:
        raise HTTPException(status_code=400, detail="provider is required")

    if provider_name not in provider_data:
        raise HTTPException(status_code=400, detail="provider not supported")

    model_name = body.get("model")
    if model_name is None:
        raise HTTPException(status_code=400, detail="model is required")

    model_dict = provider_data.get(provider_name, {}).get("models")
    if provider_name != "azure" and model_name not in model_dict:
        raise HTTPException(status_code=400, detail="model is not valid")

    params = provider_data.get(provider_name, {}).get("models").get(model_name)
    if params is None:
        params = get_params(provider_name, model_name)
    return RagaApiResponse(True, {"params": params}, None)


@router.post("/providers/models/params/values")
async def get_model_params_values(request: Request) -> RagaApiResponse:
    body = json.loads((await request.body()).decode())
    provider_name = body.get("provider")
    if provider_name is None:
        raise HTTPException(status_code=400, detail="provider is required")

    if provider_name not in provider_data:
        raise HTTPException(status_code=400, detail="provider not supported")

    model_name = body.get("model")
    if model_name is None:
        raise HTTPException(status_code=400, detail="model is required")
    params_values = raw_data.get(model_name, {})
    return RagaApiResponse(True, {"paramsValues": params_values}, None)
