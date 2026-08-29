from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ProviderWireRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    path: str
    body: dict[str, object]


class OcrFixtureRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    sdk_kwargs: dict[str, object]
    provider_request: ProviderWireRequest


class OcrFixtureResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status_code: int
    headers: dict[str, str]
    body: dict[str, object]


class OcrFixture(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: OcrFixtureRequest
    response: OcrFixtureResponse
