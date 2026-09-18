from asyncio import Future
from collections.abc import Coroutine, Mapping, Sequence
from typing import Never, final

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest

class RustBridgeDeclined(Exception): ...
class RustUpstreamError(Exception): ...

def ocr(
    request: LiteLLMOcrRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> OCRResponse: ...
def aocr(
    request: LiteLLMOcrRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> Coroutine[object, object, OCRResponse]: ...
def transcription(
    model: str,
    audio: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    optional_params: Mapping[str, object] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, object]: ...
def atranscription(
    model: str,
    audio: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    optional_params: Mapping[str, object] | None = None,
    timeout_seconds: float | None = None,
) -> Future[dict[str, object]]: ...
@final
class ResponsesWebSocketConnection:
    def __new__(cls, _uninstantiable: Never, /) -> Never: ...
    @classmethod
    def connect(
        cls,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
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

__all__ = [
    "ResponsesWebSocketConnection",
    "RustBridgeDeclined",
    "RustUpstreamError",
    "TokenCounter",
    "aocr",
    "atranscription",
    "gil_stats",
    "ocr",
    "transcription",
]
