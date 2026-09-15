from asyncio import Future
from collections.abc import Callable, Coroutine
from typing import Literal, final, overload

from typing_extensions import Never

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.ocr import LiteLLMOcrRequest

class RustBridgeDeclined(Exception): ...
class RustBridgeUnavailable(Exception): ...
class RustHostCallbackError(Exception): ...
class RustUpstreamError(Exception): ...

@overload
def _ocr_lifecycle(
    request: LiteLLMOcrRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    asynchronous: Literal[False],
    host: object,
) -> OCRResponse: ...
@overload
def _ocr_lifecycle(
    request: LiteLLMOcrRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    asynchronous: Literal[True],
    host: object,
) -> Coroutine[object, object, OCRResponse]: ...
def _messages_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
) -> Never: ...
def _chat_completions_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
) -> Never: ...
def _transcription_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
) -> Never: ...
def _embeddings_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
) -> Never: ...
def _rerank_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
) -> Never: ...
def _image_generation_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
) -> Never: ...
def _image_edit_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
) -> Never: ...
def _speech_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
) -> Never: ...
def _moderation_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
) -> Never: ...
def _responses_lifecycle(
    request: object, args: tuple[object, ...], kwargs: dict[str, object], asynchronous: bool, host: object
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
    has_agentic_hook: bool | None = None,
    on_request: Callable[[], None] | None = None,
) -> dict[str, object]: ...
def amessages(
    model: str,
    body: object,
    api_key: str | None = None,
    api_base: str | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: object = None,
    timeout_seconds: float | None = None,
    has_agentic_hook: bool | None = None,
    on_request: Callable[[], None] | None = None,
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
    host_facts: object = None,
    on_request: Callable[[], None] | None = None,
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
    host_facts: object = None,
    on_request: Callable[[], None] | None = None,
) -> Future[dict[str, object]]: ...

_OCR_MAX_FILE_BYTES: int

def _ocr_file_document(document: object) -> dict[str, object]: ...
def _ocr_mime_type(file_name: str) -> str: ...
def _ocr_upload_document(
    file_content: bytes, file_name: str | None = None, content_type: str | None = None
) -> dict[str, object]: ...

@final
class ResponsesWebSocketConnection:
    @classmethod
    def connect(
        cls,
        url: str,
        headers: object = None,
        timeout_seconds: float | None = None,
        custom_llm_provider: str | None = None,
    ) -> Future[ResponsesWebSocketConnection]: ...
    def send_text(self, text: str) -> Future[None]: ...
    def recv_text(self) -> Future[str | None]: ...
    def close(self) -> Future[None]: ...

def count_input_tokens(
    body: bytes,
    kind: str | None,
    encoding: str,
    disabled: bool,
    legacy_accounting: bool,
    resource_loader: Callable[[str], str],
) -> Future[dict[str, object]]: ...
def gil_stats() -> dict[str, int]: ...

__all__ = [
    "_OCR_MAX_FILE_BYTES",
    "ResponsesWebSocketConnection",
    "RustBridgeDeclined",
    "RustBridgeUnavailable",
    "RustHostCallbackError",
    "RustUpstreamError",
    "_chat_completions_lifecycle",
    "_embeddings_lifecycle",
    "_image_edit_lifecycle",
    "_image_generation_lifecycle",
    "_messages_lifecycle",
    "_moderation_lifecycle",
    "_ocr_file_document",
    "_ocr_lifecycle",
    "_ocr_mime_type",
    "_ocr_upload_document",
    "_rerank_lifecycle",
    "_responses_lifecycle",
    "_speech_lifecycle",
    "_transcription_lifecycle",
    "achat_completions",
    "amessages",
    "aocr",
    "atranscription",
    "chat_completions",
    "count_input_tokens",
    "gil_stats",
    "messages",
    "ocr",
    "transcription",
]
