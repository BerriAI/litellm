import datetime
from asyncio import Future
from collections.abc import Coroutine, Mapping
from typing import Never, final

import httpx

from litellm.llms.base_llm.ocr.transformation import OCRResponse

class RustBridgeDeclined(Exception): ...
class RustUpstreamError(Exception): ...

def ocr(
    model: str,
    document: Mapping[str, object],
    api_key: str | None = ...,
    api_base: str | None = ...,
    timeout: float | httpx.Timeout | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: dict[str, object] | None = ...,
    **kwargs: object,
) -> OCRResponse: ...
def aocr(
    model: str,
    document: Mapping[str, object],
    api_key: str | None = ...,
    api_base: str | None = ...,
    timeout: float | httpx.Timeout | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: dict[str, object] | None = ...,
    **kwargs: object,
) -> Coroutine[object, object, OCRResponse]: ...
def transcription(
    model: str,
    audio: object,
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: object = ...,
    optional_params: object = ...,
    timeout_seconds: float | None = ...,
) -> dict[str, object]: ...
def atranscription(
    model: str,
    audio: object,
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: object = ...,
    optional_params: object = ...,
    timeout_seconds: float | None = ...,
) -> Future[dict[str, object]]: ...
def messages(
    model: str,
    body: object,
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: object = ...,
    timeout_seconds: float | None = ...,
) -> dict[str, object]: ...
def amessages(
    model: str,
    body: object,
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: object = ...,
    timeout_seconds: float | None = ...,
) -> Future[dict[str, object]]: ...
def chat_completions(
    model: str,
    messages: object,
    optional_params: object = ...,
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: object = ...,
    timeout_seconds: float | None = ...,
) -> dict[str, object]: ...
def achat_completions(
    model: str,
    messages: object,
    optional_params: object = ...,
    api_key: str | None = ...,
    api_base: str | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: object = ...,
    timeout_seconds: float | None = ...,
) -> Future[dict[str, object]]: ...
def chat_completions_decline(
    model: str,
    messages: object,
    optional_params: object = ...,
    custom_llm_provider: str | None = ...,
) -> str | None: ...

@final
class ResponsesWebSocketConnection:
    def __new__(cls, _uninstantiable: Never, /) -> Never: ...
    @classmethod
    def connect(
        cls, url: str, headers: object = ..., timeout_seconds: float | None = ...
    ) -> Future[ResponsesWebSocketConnection]: ...
    def send_text(self, text: str) -> Future[None]: ...
    def recv_text(self) -> Future[str | None]: ...
    def close(self) -> Future[None]: ...

@final
class TokenCounter:
    def __new__(cls, tokenizer_json: str) -> TokenCounter: ...
    @staticmethod
    def from_cl100k_ranks(rank_file: str) -> TokenCounter: ...
    @staticmethod
    def from_o200k_ranks(rank_file: str) -> TokenCounter: ...
    def acount_request(self, body: bytes) -> Future[dict[str, object]]: ...

def gil_stats() -> dict[str, int]: ...
def _debug_setup(
    call_type: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    start: datetime.datetime,
    asynchronous: bool,
) -> tuple[object, dict[str, object]]: ...

__all__ = [
    "ResponsesWebSocketConnection",
    "RustBridgeDeclined",
    "RustUpstreamError",
    "TokenCounter",
    "_debug_setup",
    "achat_completions",
    "amessages",
    "aocr",
    "atranscription",
    "chat_completions",
    "chat_completions_decline",
    "gil_stats",
    "messages",
    "ocr",
    "transcription",
]
