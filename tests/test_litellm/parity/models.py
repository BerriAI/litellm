from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExceptionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    class_name: str
    status_code: int | None
    message: str


class SDKOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    response_type: str
    response_json: dict[str, object]


class ParityTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    outputs: tuple[SDKOutput, ...]
    exception: ExceptionReport | None


class NativeEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    rust_enabled: bool
    native_callable_loaded: bool
    native_handled_case: bool


class SDKReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace: ParityTrace
    native: NativeEvidence


class CapturedRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    path: str
    authorization: str | None
    content_type: str | None
    parity_case: str | None
    body: dict[str, object]
    user_agent: str | None


class Execution(BaseModel):
    model_config = ConfigDict(frozen=True)

    report: SDKReport
    requests: tuple[CapturedRequest, ...]


class ReplayResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status_code: int
    headers: dict[str, str]
    body: dict[str, object]
