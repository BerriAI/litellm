"""
``PythonHost``: what the router core reaches outside itself, served from the litellm package.

The core never touches ``litellm.*`` directly. Inference calls (``invoke``), backoff sleeps,
callback registration and the process-wide cache all go through the host, and the process settings
come from ``dynamic()`` (``litellm/router_config.py``). A Rust core gets the same operations.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypeAlias,
    cast,  # noqa: TID251  # untyped litellm globals: the live config view and the callback lists
)

from typing_extensions import overload

import litellm
from litellm.router_config import RouterDynamicConfig, live_dynamic_config

if TYPE_CHECKING:
    from pydantic import BaseModel

    from litellm.caching.caching import Cache
    from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
    from litellm.text_completions.streaming_iterator import TextCompletionStreamWrapper
    from litellm.types.llms.openai import HttpxBinaryResponseContent, OpenAIFileObject
    from litellm.types.rerank import RerankResponse
    from litellm.types.utils import (
        AdapterCompletionStreamWrapper,
        EmbeddingResponse,
        ImageResponse,
        LiteLLMBatch,
        ModelInfo,
        ModelResponse,
        TextCompletionResponse,
        TranscriptionResponse,
    )

CallbackList: TypeAlias = Literal[
    "success_callback", "_async_success_callback", "failure_callback", "_async_failure_callback", "input_callback"
]

Endpoint: TypeAlias = Callable[..., object]


class PythonHost:
    __slots__ = ()

    # The endpoints whose results the router core reads, typed as litellm declares them; any other
    # op gets the untyped Endpoint.
    @overload
    def invoke(self, op: Literal["acompletion"]) -> Callable[..., Awaitable["ModelResponse | CustomStreamWrapper"]]: ...
    @overload
    def invoke(
        self, op: Literal["atext_completion"]
    ) -> Callable[..., Awaitable["TextCompletionResponse | TextCompletionStreamWrapper"]]: ...
    @overload
    def invoke(
        self, op: Literal["aadapter_completion"]
    ) -> Callable[..., Awaitable["BaseModel | AdapterCompletionStreamWrapper | None"]]: ...
    @overload
    def invoke(self, op: Literal["aembedding"]) -> Callable[..., Awaitable["EmbeddingResponse"]]: ...
    @overload
    def invoke(self, op: Literal["aimage_generation"]) -> Callable[..., Awaitable["ImageResponse"]]: ...
    @overload
    def invoke(self, op: Literal["atranscription"]) -> Callable[..., Awaitable["TranscriptionResponse"]]: ...
    @overload
    def invoke(self, op: Literal["aspeech"]) -> Callable[..., Awaitable["HttpxBinaryResponseContent"]]: ...
    @overload
    def invoke(self, op: Literal["arerank"]) -> Callable[..., Awaitable["RerankResponse"]]: ...
    @overload
    def invoke(self, op: Literal["acreate_file"]) -> Callable[..., Awaitable["OpenAIFileObject"]]: ...
    @overload
    def invoke(self, op: Literal["acreate_batch"]) -> Callable[..., Awaitable["LiteLLMBatch"]]: ...
    @overload
    def invoke(self, op: Literal["aretrieve_batch"]) -> Callable[..., Awaitable["LiteLLMBatch"]]: ...
    @overload
    def invoke(self, op: Literal["acancel_batch"]) -> Callable[..., Awaitable["LiteLLMBatch"]]: ...
    @overload
    def invoke(self, op: Literal["alist_batches"]) -> Callable[..., Awaitable[object]]: ...
    @overload
    def invoke(self, op: Literal["get_llm_provider"]) -> Callable[..., "tuple[str, str, str | None, str | None]"]: ...
    @overload
    def invoke(self, op: Literal["get_model_info"]) -> Callable[..., "ModelInfo"]: ...
    @overload
    def invoke(self, op: Literal["get_supported_openai_params"]) -> Callable[..., "Sequence[str] | None"]: ...
    @overload
    def invoke(self, op: Literal["token_counter"]) -> Callable[..., int]: ...
    @overload
    def invoke(self, op: Literal["calculate_retry_after"]) -> Callable[..., float]: ...
    @overload
    def invoke(self, op: str) -> Endpoint: ...
    def invoke(self, op: str) -> Endpoint:
        """The litellm function named ``op`` (``acompletion``, ``get_model_info``, ...), looked up at call time."""
        endpoint: Final[object] = getattr(litellm, op, None)
        if endpoint is None:
            raise AttributeError(f"litellm has no endpoint {op!r}")
        if not callable(endpoint):
            raise TypeError(f"litellm.{op} is not an endpoint")
        return endpoint

    def sleep(self, seconds: float) -> Awaitable[None]:
        return asyncio.sleep(seconds)

    def dynamic(self) -> RouterDynamicConfig:
        """The process settings, read live on each access (``current_dynamic_config()`` takes a snapshot)."""
        return cast(RouterDynamicConfig, live_dynamic_config)  # cast-ok: the live view reads the same fields

    @property
    def callbacks(self) -> "LoggingCallbackManager":
        return litellm.logging_callback_manager

    def callback_list(self, name: CallbackList) -> list[object]:  # mutable-ok: the process callback list itself
        value: Final = getattr(litellm, name)
        return cast(list[object], value)  # cast-ok: the process callback list is an untyped litellm global

    def set_callback_list(self, name: CallbackList, value: list[object]) -> None:  # mutable-ok: assigned as given
        setattr(litellm, name, value)

    def global_cache(self) -> "Cache | None":
        return litellm.cache

    def set_global_cache(self, cache: "Cache | None") -> None:
        litellm.cache = cache

    def suppress_debug_info(self) -> None:
        litellm.suppress_debug_info = True


python_host: Final = PythonHost()
