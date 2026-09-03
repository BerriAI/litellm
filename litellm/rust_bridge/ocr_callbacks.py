from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import BaseModel, JsonValue
from typing_extensions import ReadOnly, TypedDict


class PreCallArguments(TypedDict):
    complete_input_dict: ReadOnly[Mapping[str, JsonValue]]
    api_base: ReadOnly[str]
    headers: ReadOnly[Mapping[str, str]]


class OcrLogging(Protocol):
    def pre_call(self, *, input: str, api_key: str | None, additional_args: PreCallArguments) -> None: ...
    def post_call(self, *, original_response: str, input: str, api_key: str | None) -> None: ...


class OcrRequestSnapshot(BaseModel):
    data: Mapping[str, JsonValue]


class OcrPreCall(BaseModel):
    model: str
    request: OcrRequestSnapshot
    api_base: str
    headers: Mapping[str, str]


class OcrPostCall(BaseModel):
    original_response: str


@dataclass(frozen=True, slots=True)
class OcrLoggingAdapter:
    logging_obj: OcrLogging
    api_key: str | None

    def pre_call(self, payload: object) -> None:
        event: Final = OcrPreCall.model_validate(payload)
        additional_args: Final[PreCallArguments] = {
            "complete_input_dict": event.request.data,
            "api_base": event.api_base,
            "headers": event.headers,
        }
        self.logging_obj.pre_call(
            input="OCR document processing",
            api_key=self.api_key,
            additional_args=additional_args,
        )

    def post_call(self, payload: object) -> None:
        event: Final = OcrPostCall.model_validate(payload)
        self.logging_obj.post_call(
            original_response=event.original_response,
            input="OCR document processing",
            api_key=self.api_key,
        )
