from __future__ import annotations

from pydantic import BaseModel, ConfigDict, JsonValue

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
