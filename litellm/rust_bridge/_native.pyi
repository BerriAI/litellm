from asyncio import Future
from collections.abc import AsyncIterator, Coroutine, Iterator, Mapping, Sequence
from typing import Never, final

import httpx
from pydantic import JsonValue

from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.chat_completions.entrypoints import LiteLLMChatCompletionsRequest
from litellm.rust_bridge.embeddings.entrypoints import LiteLLMEmbeddingRequest
from litellm.rust_bridge.messages.entrypoints import LiteLLMMessagesRequest
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest
from litellm.rust_bridge.responses.entrypoints import LiteLLMResponsesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import EmbeddingResponse, ModelResponse

class RustBridgeDeclined(Exception): ...
class RustUpstreamError(Exception): ...
class ForkedAfterNativeRuntimeStarted(RuntimeError): ...
class ProcessReservedForForking(RuntimeError): ...

@final
class NativeDiagnosticProcessor:
    def __new__(cls, minimum_custom_key_length: int) -> NativeDiagnosticProcessor: ...
    def redact_text(self, text: str) -> str: ...
    def redact_structured_text(self, key: str | None, text: str) -> str: ...
    def redact_client_message(self, text: str) -> str: ...
    def process_diagnostic(
        self,
        message: str,
        exception: str | None,
        stack: str | None,
        leaves: Sequence[tuple[str | None, str]],
        policy: tuple[bool, int, int],
    ) -> tuple[str, str | None, str | None, list[str], bool]: ...
    def scrub_access_arguments(self, arguments: Sequence[str]) -> list[str]: ...

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
def ocr_health_check_document(model: str, custom_llm_provider: str | None) -> dict[str, object]: ...
def ocr_passthrough_response(
    model: str,
    api_base: str,
    endpoint: str,
    status_code: int,
    body: bytes,
) -> dict[str, object] | None: ...
def embedding(
    request: LiteLLMEmbeddingRequest,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> EmbeddingResponse: ...
def aembedding(
    request: LiteLLMEmbeddingRequest,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> Coroutine[object, object, EmbeddingResponse]: ...
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
def completion(
    request: LiteLLMChatCompletionsRequest,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> ModelResponse: ...
def acompletion(
    request: LiteLLMChatCompletionsRequest,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> Coroutine[object, object, ModelResponse]: ...
def responses(
    request: LiteLLMResponsesRequest,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> ResponsesAPIResponse: ...
def aresponses(
    request: LiteLLMResponsesRequest,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> Coroutine[object, object, ResponsesAPIResponse]: ...
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
class _ResponseCacheRuntime:
    @staticmethod
    def from_cache(cache: object) -> _ResponseCacheRuntime: ...
    @staticmethod
    def from_selected(cache: object) -> _ResponseCacheRuntime: ...
    @property
    def kind(self) -> str: ...
    def lookup(
        self,
        request: object,
        *,
        callback_kwargs: Mapping[str, object] | Sequence[object] | None = None,
    ) -> object: ...
    def lookup_semantic(self, request: object) -> tuple[object, float | None]: ...
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
    def async_lookup_semantic(self, request: object) -> Future[tuple[object, float | None]]: ...
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
class _CacheResolver:
    def __new__(cls, namespace: object) -> _CacheResolver: ...
    def resolve(self) -> _ResponseCacheRuntime: ...

@final
class _CacheTestResolver:
    def __new__(cls, namespace: object) -> _CacheTestResolver: ...
    def resolve(self) -> _ResponseCacheRuntime: ...

@final
class TokenCounter:
    @staticmethod
    def from_tokenizer(tokenizer: Tokenizer, fast: bool = False) -> TokenCounter: ...
    def acount_request(self, body: bytes) -> Future[dict[str, object]]: ...

@final
class Tokenizer:
    @staticmethod
    def from_tiktoken(encoding: str) -> Tokenizer: ...
    @staticmethod
    def from_json(tokenizer_json: str) -> Tokenizer: ...
    @staticmethod
    def from_pretrained(
        identifier: str,
        revision: str = "main",
        token: str | None = None,
    ) -> Tokenizer: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...
    def count(self, text: str, fast: bool = False) -> int: ...
    # tiktoken encodings
    def encode_special(self, text: str, allowed: Sequence[str]) -> list[int]: ...
    def encode_with_unstable(self, text: str, allowed: Sequence[str]) -> tuple[list[int], list[list[int]]]: ...
    def encode_single_token(self, piece: bytes) -> int: ...
    def special_tokens(self) -> dict[str, int]: ...
    def max_token_value(self) -> int: ...
    def is_special_token(self, token: int) -> bool: ...
    def token_byte_values(self) -> list[bytes]: ...
    def decode_bytes(self, ids: Sequence[int]) -> bytes: ...
    # Hugging Face tokenizers
    def to_json(self, pretty: bool = False) -> str: ...
    def token_to_id(self, token: str) -> int | None: ...
    def id_to_token(self, id: int) -> str | None: ...
    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]: ...
    def get_vocab_size(self, with_added_tokens: bool = True) -> int: ...
    def added_tokens_decoder(self) -> list[tuple[int, tuple[str, bool, bool, bool, bool, bool]]]: ...
    def padding(self) -> dict[str, object] | None: ...
    def truncation(self) -> dict[str, object] | None: ...
    def num_special_tokens_to_add(self, is_pair: bool) -> int: ...
    def encode_special_tokens(self) -> bool: ...
    def encode_huggingface(
        self,
        sequence: str | Sequence[str],
        pair: str | Sequence[str] | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        fast: bool = False,
    ) -> HuggingFaceEncoding: ...
    def encode_batch_huggingface(
        self,
        inputs: Sequence[tuple[str | Sequence[str], str | Sequence[str] | None]],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        fast: bool = False,
    ) -> list[HuggingFaceEncoding]: ...
    @property
    def name(self) -> str: ...

@final
class HuggingFaceEncoding:
    def __new__(cls, json: str | None = None) -> HuggingFaceEncoding: ...
    @staticmethod
    def merge(encodings: Sequence[HuggingFaceEncoding], growing_offsets: bool = True) -> HuggingFaceEncoding: ...
    def __len__(self) -> int: ...
    def __reduce__(self) -> tuple[type[HuggingFaceEncoding], tuple[str]]: ...
    def word_to_tokens(self, word_index: int, sequence_index: int = 0) -> tuple[int, int] | None: ...
    def word_to_chars(self, word_index: int, sequence_index: int = 0) -> tuple[int, int] | None: ...
    def token_to_sequence(self, token_index: int) -> int | None: ...
    def token_to_chars(self, token_index: int) -> tuple[int, int] | None: ...
    def token_to_word(self, token_index: int) -> int | None: ...
    def char_to_token(self, char_pos: int, sequence_index: int = 0) -> int | None: ...
    def char_to_word(self, char_pos: int, sequence_index: int = 0) -> int | None: ...
    def set_sequence_id(self, sequence_id: int) -> None: ...
    def pad(
        self,
        length: int,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
    ) -> None: ...
    def truncate(self, max_length: int, stride: int = 0, direction: str = "right") -> None: ...
    @property
    def ids(self) -> list[int]: ...
    @property
    def tokens(self) -> list[str]: ...
    @property
    def offsets(self) -> list[tuple[int, int]]: ...
    @property
    def type_ids(self) -> list[int]: ...
    @property
    def attention_mask(self) -> list[int]: ...
    @property
    def special_tokens_mask(self) -> list[int]: ...
    @property
    def word_ids(self) -> list[int | None]: ...
    @property
    def sequence_ids(self) -> list[int | None]: ...
    @property
    def overflowing(self) -> list[HuggingFaceEncoding]: ...
    @property
    def n_sequences(self) -> int: ...

def gil_stats() -> dict[str, int]: ...
def process_state_started() -> bool: ...
def reserve_process_for_forking() -> None: ...

__all__ = [
    "ForkedAfterNativeRuntimeStarted",
    "HuggingFaceEncoding",
    "NativeDiagnosticProcessor",
    "ProcessReservedForForking",
    "ResponsesWebSocketConnection",
    "RustBridgeDeclined",
    "RustUpstreamError",
    "TokenCounter",
    "Tokenizer",
    "achat_completions",
    "acompletion",
    "aembedding",
    "amessages",
    "aocr",
    "aresponses",
    "atranscription",
    "chat_completions",
    "chat_completions_decline",
    "completion",
    "embedding",
    "gil_stats",
    "messages",
    "ocr",
    "ocr_health_check_document",
    "ocr_passthrough_response",
    "process_state_started",
    "reserve_process_for_forking",
    "responses",
    "transcription",
]

@final
class _SecretManagerRuntime:
    @staticmethod
    def from_config(
        system: str,
        environment: Mapping[str, str],
        settings: Mapping[str, object] | None = None,
        enterprise_enabled: bool = False,
    ) -> _SecretManagerRuntime: ...
    @staticmethod
    def from_client(client: object) -> _SecretManagerRuntime | None: ...
    @property
    def system(self) -> str: ...
    def read_secret(self, name: str, settings: Mapping[str, object] | None = None) -> JsonValue: ...
    def read_secret_async(self, name: str, settings: Mapping[str, object] | None = None) -> Future[JsonValue]: ...
    def async_write_secret(
        self, secret_name: str, secret_value: str, description: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None, tags: object = None,
    ) -> Future[dict[str, JsonValue]]: ...
    def async_delete_secret(
        self, secret_name: str, recovery_window_in_days: int | None = None,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Future[dict[str, JsonValue]]: ...
    def async_rotate_secret(
        self, current_secret_name: str, new_secret_name: str, new_secret_value: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Future[dict[str, JsonValue]]: ...
    def sync_read_secret(
        self, secret_name: str, optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None, primary_secret_name: str | None = None,
    ) -> JsonValue: ...
    def async_read_secret(
        self, secret_name: str, optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None, primary_secret_name: str | None = None,
    ) -> Future[JsonValue]: ...
