from asyncio import Future
from collections.abc import AsyncIterator, Coroutine, Iterator, Mapping, Sequence
from typing import Never, final

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.messages.entrypoints import LiteLLMMessagesRequest
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

class RustBridgeDeclined(Exception): ...
class RustUpstreamError(Exception): ...
class ForkedAfterNativeRuntimeStarted(RuntimeError): ...
class ProcessReservedForForking(RuntimeError): ...

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
def messages(
    request: LiteLLMMessagesRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> AnthropicMessagesResponse | Iterator[bytes]: ...
def amessages(
    request: LiteLLMMessagesRequest,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> Coroutine[object, object, AnthropicMessagesResponse | AsyncIterator[bytes]]: ...
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
class _CacheTestBinding:
    @property
    def kind(self) -> str: ...
    def lookup(
        self,
        request: object,
        *,
        callback_kwargs: Mapping[str, object] | Sequence[object] | None = None,
    ) -> object: ...
    def store(
        self,
        request: object,
        response: object,
        *,
        callback_kwargs: Mapping[str, object] | None = None,
    ) -> None: ...
    def lookup_batch(
        self,
        requests: Sequence[object],
        *,
        callback_kwargs: Sequence[object] | None = None,
    ) -> object: ...
    def async_lookup(
        self,
        request: object,
        *,
        callback_kwargs: Mapping[str, object] | None = None,
    ) -> Future[object]: ...
    def async_store(
        self,
        request: object,
        response: object,
        *,
        callback_kwargs: Mapping[str, object] | None = None,
    ) -> Future[None]: ...
    def async_lookup_batch(
        self,
        requests: Sequence[object],
        *,
        callback_kwargs: Sequence[object] | None = None,
    ) -> Future[object]: ...
    def async_store_batch(
        self,
        requests: Sequence[object],
        responses: Sequence[object],
        *,
        callback_result: object = None,
        callback_kwargs: Mapping[str, object] | None = None,
    ) -> Future[object]: ...
    def async_flush(self) -> Future[None]: ...
    def ping(self) -> Future[object]: ...

@final
class _CacheTestHandle:
    def __new__(cls, _uninstantiable: Never, /) -> Never: ...
    @staticmethod
    def memory(
        *,
        capacity: int = 200,
        ttl_seconds: float = 600.0,
        max_entry_bytes: int = 1048576,
    ) -> _CacheTestHandle: ...
    @staticmethod
    def redis(
        url: str,
        *,
        ttl_seconds: float = 60.0,
        namespace: str | None = None,
        startup_nodes: Sequence[tuple[str, int]] | None = None,
    ) -> _CacheTestHandle: ...
    @staticmethod
    def disk(directory: str) -> _CacheTestHandle: ...
    @staticmethod
    def qdrant_semantic(
        url: str,
        *,
        collection_name: str,
        similarity_threshold: float,
        vector_size: int,
        embedding_model: str = "text-embedding-3-small",
        api_key: str | None = None,
        embedding_api_key: str | None = None,
        embedding_api_base: str | None = None,
        embedding_timeout_seconds: float | None = None,
        quantization: str = "binary",
    ) -> _CacheTestHandle: ...
    @staticmethod
    def azure_blob(account_url: str, container: str) -> _CacheTestHandle: ...
    @staticmethod
    def redis_semantic(backend: object) -> _CacheTestHandle: ...
    @staticmethod
    def valkey_semantic(
        url: str,
        similarity_threshold: float,
        index_name: str,
        embedder: object,
    ) -> _CacheTestHandle: ...
    @staticmethod
    def gcs(
        bucket_name: str,
        *,
        gcs_path: str | None = None,
        path_service_account: str | None = None,
        endpoint: str | None = None,
        token: str | None = None,
    ) -> _CacheTestHandle: ...
    @staticmethod
    def s3(
        bucket: str,
        *,
        region: str,
        endpoint_url: str | None = None,
        key_prefix: str = "",
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
    ) -> _CacheTestHandle: ...
    @property
    def backend(self) -> str: ...
    def _bind_facade(self, facade: object) -> None: ...

@final
class _CacheTestResolver:
    def __new__(cls, namespace: object) -> _CacheTestResolver: ...
    def resolve(self) -> _CacheTestBinding: ...

@final
class TokenCounter:
    def __new__(cls, tokenizer_json: str) -> TokenCounter: ...
    @staticmethod
    def from_cl100k_ranks(rank_file: str) -> TokenCounter: ...
    @staticmethod
    def from_o200k_ranks(rank_file: str) -> TokenCounter: ...
    @staticmethod
    def from_tiktoken(encoding: str) -> TokenCounter: ...
    def acount_request(self, body: bytes) -> Future[dict[str, object]]: ...

def gil_stats() -> dict[str, int]: ...
def process_state_started() -> bool: ...
def reserve_process_for_forking() -> None: ...

__all__ = [
    "ForkedAfterNativeRuntimeStarted",
    "ProcessReservedForForking",
    "ResponsesWebSocketConnection",
    "RustBridgeDeclined",
    "RustUpstreamError",
    "TokenCounter",
    "achat_completions",
    "amessages",
    "aocr",
    "atranscription",
    "chat_completions",
    "chat_completions_decline",
    "gil_stats",
    "messages",
    "ocr",
    "process_state_started",
    "reserve_process_for_forking",
    "transcription",
]
