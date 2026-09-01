from __future__ import annotations

from typing import Annotated, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue


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


SDKReport = Annotated[SDKSuccess | SDKError, Field(discriminator="status")]


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
