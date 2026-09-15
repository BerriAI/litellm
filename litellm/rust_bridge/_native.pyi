from asyncio import Future
from collections.abc import Callable, Coroutine
from typing import Literal, Protocol, TypedDict, final

from typing_extensions import Never, NotRequired, Required

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.lifecycle import Await, Complete
from litellm.rust_bridge.ocr import LiteLLMOcrRequest

class _CommonLifecycleRequest(TypedDict):
    model: Required[str]
    api_key: NotRequired[str | None]
    api_base: NotRequired[str | None]
    custom_llm_provider: NotRequired[str | None]
    extra_headers: NotRequired[object]
    timeout: NotRequired[object]

class _MessagesLifecycleRequest(_CommonLifecycleRequest):
    body: Required[object]
    has_agentic_hook: NotRequired[bool | None]

class _ChatCompletionsLifecycleRequest(_CommonLifecycleRequest):
    messages: Required[object]
    optional_params: NotRequired[object]
    host_facts: NotRequired[object]

class _TranscriptionLifecycleRequest(_CommonLifecycleRequest):
    audio: Required[object]
    optional_params: NotRequired[object]

class _CompletedLifecycleHost(Protocol):
    def invoke(
        self,
        operation: Literal[
            "post_process",
            "cache_response",
            "cached_response",
            "project",
            "before_request",
            "after_response",
            "response",
            "map_failure",
        ],
        payload: object,
        request: object,
        kwargs: dict[str, object],
        logger: object,
    ) -> Await | Complete: ...

class RustBridgeDeclined(Exception): ...
class RustBridgeUnavailable(Exception): ...
class RustHostCallbackError(Exception): ...
class RustUpstreamError(Exception): ...

def ocr(
    request: LiteLLMOcrRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    host: object,
) -> OCRResponse: ...
def aocr(
    request: LiteLLMOcrRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    host: object,
) -> Coroutine[object, object, OCRResponse]: ...
def transcription(
    request: _TranscriptionLifecycleRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    host: _CompletedLifecycleHost,
) -> object: ...
def atranscription(
    request: _TranscriptionLifecycleRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    host: _CompletedLifecycleHost,
) -> Coroutine[object, object, object]: ...
def embedding(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def aembedding(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def rerank(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def arerank(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def image_generation(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def aimage_generation(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def image_edit(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def aimage_edit(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def speech(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def aspeech(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def moderation(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def amoderation(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def responses(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def aresponses(request: object, args: tuple[object, ...], kwargs: dict[str, object], host: object) -> Never: ...
def messages(
    request: _MessagesLifecycleRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    host: _CompletedLifecycleHost,
) -> object: ...
def amessages(
    request: _MessagesLifecycleRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    host: _CompletedLifecycleHost,
) -> Coroutine[object, object, object]: ...
def chat_completions(
    request: _ChatCompletionsLifecycleRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    host: _CompletedLifecycleHost,
) -> object: ...
def achat_completions(
    request: _ChatCompletionsLifecycleRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    host: _CompletedLifecycleHost,
) -> Coroutine[object, object, object]: ...

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
    "_ocr_file_document",
    "_ocr_mime_type",
    "_ocr_upload_document",
    "achat_completions",
    "aembedding",
    "aimage_edit",
    "aimage_generation",
    "amessages",
    "amoderation",
    "aocr",
    "arerank",
    "aresponses",
    "aspeech",
    "atranscription",
    "chat_completions",
    "count_input_tokens",
    "embedding",
    "gil_stats",
    "image_edit",
    "image_generation",
    "messages",
    "moderation",
    "ocr",
    "rerank",
    "responses",
    "speech",
    "transcription",
]
