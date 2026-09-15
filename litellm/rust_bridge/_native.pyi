from asyncio import Future
from collections.abc import Coroutine, Mapping, Sequence
from typing import final

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.ocr import LiteLLMOcrRequest

class RustBridgeDeclined(Exception): ...
class RustUpstreamError(Exception): ...

def ocr(
    model: str,
    document: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    optional_params: Mapping[str, object] | None = None,
    input_sources: Sequence[object] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, object]: ...
def aocr(
    model: str,
    document: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    optional_params: Mapping[str, object] | None = None,
    input_sources: Sequence[object] | None = None,
    timeout_seconds: float | None = None,
) -> Future[dict[str, object]]: ...

_OCR_MAX_FILE_BYTES: int

def _ocr_upload_document(
    file_content: bytes,
    file_name: str | None = None,
    content_type: str | None = None,
) -> dict[str, str]: ...
def _ocr_file_document(document: Mapping[str, object]) -> dict[str, str]: ...
def _ocr_mime_type(file_name: str) -> str: ...
def _ocr_lifecycle(
    request: LiteLLMOcrRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    asynchronous: bool,
) -> OCRResponse | Coroutine[object, object, OCRResponse]: ...
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
def messages(
    model: str,
    body: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, object]: ...
def amessages(
    model: str,
    body: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    timeout_seconds: float | None = None,
) -> Future[dict[str, object]]: ...
def chat_completions_decline(
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object] | None = None,
    custom_llm_provider: str | None = None,
) -> str | None: ...
def chat_completions(
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, object]: ...
def achat_completions(
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: Mapping[str, object] | None = None,
    timeout_seconds: float | None = None,
) -> Future[dict[str, object]]: ...

@final
class ResponsesWebSocketConnection:
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
    "_OCR_MAX_FILE_BYTES",
    "ResponsesWebSocketConnection",
    "RustBridgeDeclined",
    "RustUpstreamError",
    "TokenCounter",
    "_ocr_file_document",
    "_ocr_lifecycle",
    "_ocr_mime_type",
    "_ocr_upload_document",
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
