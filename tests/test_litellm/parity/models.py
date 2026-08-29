from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from litellm.llms.base_llm.ocr.transformation import OCRResponse


class CapturedRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    path: str
    headers: tuple[tuple[str, str], ...]
    body: JsonValue
    user_agent: str | None


class SDKReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    response: OCRResponse


class Execution(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: CapturedRequest
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
