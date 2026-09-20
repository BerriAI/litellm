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
class TokenCounter:
    def __new__(cls, tokenizer_json: str) -> TokenCounter: ...
    @staticmethod
    def from_cl100k_ranks(rank_file: str) -> TokenCounter: ...
    @staticmethod
    def from_o200k_ranks(rank_file: str) -> TokenCounter: ...
    @staticmethod
    def from_tiktoken(encoding: str) -> TokenCounter: ...
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
    def count(self, text: str) -> int: ...
    def encode_special(self, text: str, allowed: Sequence[str]) -> list[int]: ...
    def special_tokens(self) -> set[str]: ...
    def decode_bytes(self, ids: Sequence[int]) -> bytes: ...
    def to_json(self, pretty: bool = False) -> str: ...
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
    def __len__(self) -> int: ...
    def __reduce__(self) -> tuple[type[HuggingFaceEncoding], tuple[str]]: ...
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

def tiktoken_encoding_for_model(model: str) -> str | None: ...
def gil_stats() -> dict[str, int]: ...
def process_state_started() -> bool: ...
def reserve_process_for_forking() -> None: ...

__all__ = [
    "ForkedAfterNativeRuntimeStarted",
    "HuggingFaceEncoding",
    "ProcessReservedForForking",
    "ResponsesWebSocketConnection",
    "RustBridgeDeclined",
    "RustUpstreamError",
    "TokenCounter",
    "Tokenizer",
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
    "tiktoken_encoding_for_model",
    "transcription",
]
