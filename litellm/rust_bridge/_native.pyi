from asyncio import Future
from collections.abc import Coroutine
from typing import Literal, overload

from typing_extensions import Never

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.ocr import LiteLLMOcrRequest

class RustBridgeDeclined(Exception): ...
class RustUpstreamError(Exception): ...

@overload
def _ocr_lifecycle(
    request: LiteLLMOcrRequest, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: Literal[False]
) -> OCRResponse: ...
@overload
def _ocr_lifecycle(
    request: LiteLLMOcrRequest, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: Literal[True]
) -> Coroutine[object, object, OCRResponse]: ...
def _messages_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def _chat_completions_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def _transcription_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def _embeddings_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def _rerank_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def _image_generation_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def _image_edit_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def _speech_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def _moderation_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def _responses_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool
) -> Never: ...
def ocr(
    model: str,
    document: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: object = None,
    optional_params: object = None,
    input_sources: object = None,
    timeout_seconds: float | None = None,
) -> dict[str, object]: ...
def aocr(
    model: str,
    document: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: object = None,
    optional_params: object = None,
    input_sources: object = None,
    timeout_seconds: float | None = None,
) -> Future[dict[str, object]]: ...
def transcription(
    model: str,
    audio: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: object = None,
    optional_params: object = None,
    timeout_seconds: float | None = None,
) -> dict[str, object]: ...
def atranscription(
    model: str,
    audio: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: object = None,
    optional_params: object = None,
    timeout_seconds: float | None = None,
) -> Future[dict[str, object]]: ...
def messages(
    model: str,
    body: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: object = None,
    timeout_seconds: float | None = None,
) -> dict[str, object]: ...
def amessages(
    model: str,
    body: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: object = None,
    timeout_seconds: float | None = None,
) -> Future[dict[str, object]]: ...
def chat_completions(
    model: str,
    messages: object,
    optional_params: object = None,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: object = None,
    timeout_seconds: float | None = None,
) -> dict[str, object]: ...
def achat_completions(
    model: str,
    messages: object,
    optional_params: object = None,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: object = None,
    timeout_seconds: float | None = None,
) -> Future[dict[str, object]]: ...
def chat_completions_decline(
    model: str, messages: object, optional_params: object = None, custom_llm_provider: str | None = None
) -> str | None: ...

_OCR_MAX_FILE_BYTES: int

def _ocr_file_document(document: object) -> dict[str, object]: ...
def _ocr_mime_type(file_name: str) -> str: ...
def _ocr_upload_document(
    file_content: bytes, file_name: str | None = None, content_type: str | None = None
) -> dict[str, object]: ...

class ResponsesWebSocketConnection:
    @classmethod
    def connect(
        cls, url: str, headers: object = None, timeout_seconds: float | None = None
    ) -> Future[ResponsesWebSocketConnection]: ...
    def send_text(self, text: str) -> Future[None]: ...
    def recv_text(self) -> Future[str | None]: ...
    def close(self) -> Future[None]: ...

class TokenCounter:
    def __init__(self, tokenizer_json: str) -> None: ...
    @staticmethod
    def from_cl100k_ranks(rank_file: str) -> TokenCounter: ...
    @staticmethod
    def from_o200k_ranks(rank_file: str) -> TokenCounter: ...
    def acount_request(self, body: bytes) -> Future[dict[str, object]]: ...

def gil_stats() -> dict[str, int]: ...
