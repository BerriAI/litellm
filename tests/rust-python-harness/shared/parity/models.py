from __future__ import annotations

import base64
from typing import Annotated, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter


class CapturedRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    path: str
    headers: tuple[tuple[str, str], ...]
    body: JsonValue
    user_agent: str | None


class SDKSuccess(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    response: JsonValue


class SDKError(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["error"] = "error"
    exception_type: str
    message: str
    status_code: int | None
    code: str | None
    error_type: str | None
    param: str | None
    model: str | None
    llm_provider: str | None


class SDKJsonChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["json"] = "json"
    value: JsonValue


class SDKBytesChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["bytes"] = "bytes"
    data_b64: str

    def data_bytes(self) -> bytes:
        return base64.b64decode(self.data_b64, validate=True)


SDKChunk = Annotated[SDKJsonChunk | SDKBytesChunk, Field(discriminator="kind")]


class SDKStreamCompleted(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["completed"] = "completed"


class SDKStreamFailed(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["failed"] = "failed"
    error: SDKError


SDKStreamTerminal = Annotated[SDKStreamCompleted | SDKStreamFailed, Field(discriminator="kind")]


class SDKStreamReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["stream"] = "stream"
    chunks: tuple[SDKChunk, ...]
    terminal: SDKStreamTerminal


SDKReport = Annotated[SDKSuccess | SDKError | SDKStreamReport, Field(discriminator="status")]
JSON_VALUE_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


def sdk_chunk(value: object) -> SDKChunk:
    if isinstance(value, bytes):
        return SDKBytesChunk(data_b64=base64.b64encode(value).decode("ascii"))
    if isinstance(value, BaseModel):
        return SDKJsonChunk(value=JSON_VALUE_ADAPTER.validate_python(value.model_dump(mode="json")))
    return SDKJsonChunk(value=JSON_VALUE_ADAPTER.validate_python(value))


def _string_attribute(error: Exception, name: str) -> str | None:
    value: Final = cast(object | None, getattr(error, name, None))
    return None if value is None else str(value)


def sdk_error_report(error: Exception) -> SDKError:
    message, _, _ = str(error).partition("\nTraceback (most recent call last):")
    raw_status_code: Final = cast(object | None, getattr(error, "status_code", None))
    status_code: Final = raw_status_code if isinstance(raw_status_code, int) else None
    return SDKError(
        exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
        message=message.rstrip(),
        status_code=status_code,
        code=_string_attribute(error, "code"),
        error_type=_string_attribute(error, "type"),
        param=_string_attribute(error, "param"),
        model=_string_attribute(error, "model"),
        llm_provider=_string_attribute(error, "llm_provider"),
    )


class Execution(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: tuple[CapturedRequest, ...]
    report: SDKReport


class SDKCommand(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_file: str
    route: str


class WorkerSuccess(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["ok"] = "ok"
    report: SDKReport


class WorkerFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["error"] = "error"
    error: str


WorkerResult = Annotated[WorkerSuccess | WorkerFailure, Field(discriminator="status")]
