"""
``PythonHost``: what the router core reaches outside itself, served from the litellm package.

The core never touches ``litellm.*`` directly. Inference calls (``invoke``), backoff sleeps,
callback registration and the process-wide cache all go through the host, and the process settings
come from ``dynamic()`` (``litellm/router_config.py``). A Rust core gets the same operations.
"""

import asyncio
import importlib
import time
import types
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypeAlias,
    cast,  # noqa: TID251  # untyped litellm globals: the live config view and the callback lists
)

from typing_extensions import overload

import litellm
from litellm.router_config import RouterConfig, RouterDynamicConfig, live_dynamic_config

if TYPE_CHECKING:
    from pydantic import BaseModel

    from litellm.caching.caching import Cache, RedisCache
    from litellm.caching.dual_cache import DualCache
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
        endpoint: Final[object] = getattr(litellm, op, None) if "." not in op else _dotted(op)
        if endpoint is None:
            raise AttributeError(f"litellm has no endpoint {op!r}")
        if not callable(endpoint):
            raise TypeError(f"litellm.{op} is not an endpoint")
        return endpoint

    def sleep(self, seconds: float) -> Awaitable[None]:
        return asyncio.sleep(seconds)

    def sleep_sync(self, seconds: float) -> None:
        """The backoff of a sync endpoint's retry."""
        time.sleep(seconds)

    def dynamic(self) -> RouterDynamicConfig:
        """The process settings, read live on each access (``current_dynamic_config()`` takes a snapshot)."""
        return cast(RouterDynamicConfig, live_dynamic_config)  # cast-ok: the live view reads the same fields

    @property
    def callbacks(self) -> "LoggingCallbackManager":
        return litellm.logging_callback_manager

    def callback_list(self, name: CallbackList) -> list[object]:  # mutable-ok: the process callback list itself
        value: Final[object] = getattr(litellm, name)  # pyright: ignore[reportAny]  # an untyped litellm global
        return cast(list[object], value)  # cast-ok: the process callback list is an untyped litellm global

    def set_callback_list(self, name: CallbackList, value: list[object]) -> None:  # mutable-ok: assigned as given
        setattr(litellm, name, value)

    def global_cache(self) -> "Cache | None":
        return litellm.cache

    def set_global_cache(self, cache: "Cache | None") -> None:
        litellm.cache = cache

    def suppress_debug_info(self) -> None:
        litellm.suppress_debug_info = True

    def routing_cache(self, config: RouterConfig) -> "DualCache":
        """The store a router keeps its routing state in (cooldowns, fail counts, usage): in memory,
        plus Redis when the router's arguments name one. The host builds and owns it; a Rust core
        reads and writes it through cache steps."""
        from litellm.caching.caching import DualCache, InMemoryCache

        return DualCache(redis_cache=redis_cache_for(config), in_memory_cache=InMemoryCache())


def redis_cache_for(config: RouterConfig) -> "RedisCache | None":
    """The Redis cache the router's ``redis_*`` and ``cache_kwargs`` arguments describe, or None."""
    import json

    from litellm.caching.caching import RedisCache, RedisClusterCache
    from litellm.utils import get_secret

    url: Final = config.get("redis_url")
    host: Final = config.get("redis_host")
    port: Final = config.get("redis_port")
    if url is None and (host is None or port is None):
        return None
    settings: Final[dict[str, object]] = {}  # mutable-ok: built here, then passed as keywords
    if url is not None:
        settings["url"] = url
    if host is not None:
        settings["host"] = host
    if port is not None:
        settings["port"] = str(port)
    if config.get("redis_password") is not None:
        settings["password"] = config.get("redis_password")
    if config.get("redis_db") is not None:
        settings["db"] = str(config.get("redis_db"))
    extra: Final = config.get("cache_kwargs")
    if isinstance(extra, dict):
        settings.update({str(key): value for key, value in extra.items()})  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # the cache_kwargs argument is untyped
    nodes: object = settings.get("startup_nodes")  # rebind-ok: the env cluster nodes when the arguments name none
    if not nodes:
        env_nodes: Final[object] = get_secret("REDIS_CLUSTER_NODES")  # pyright: ignore[reportUnknownVariableType]  # get_secret is untyped
        if isinstance(env_nodes, str):
            parsed: Final[object] = json.loads(env_nodes)  # pyright: ignore[reportAny]  # plain JSON
            nodes = parsed  # rebind-ok: see above
    if nodes:
        return RedisClusterCache(**{**settings, "startup_nodes": nodes})  # pyright: ignore[reportArgumentType]  # RedisCache's keyword arguments
    return RedisCache(**settings)  # pyright: ignore[reportArgumentType]  # RedisCache's keyword arguments


python_host: Final = PythonHost()


def _dotted(op: str) -> object:
    """``op`` named by its path under ``litellm`` (``vector_stores.main.asearch``), read at call time.

    Each segment is a module attribute, a submodule, or a key of a mapping such as a module's
    table of generated endpoints.
    """
    node: object = litellm  # rebind-ok: walks the path one segment at a time
    path: str = "litellm"  # rebind-ok: the module path walked so far
    for segment in op.split("."):
        path = f"{path}.{segment}"
        if isinstance(node, Mapping):
            node = cast(Mapping[str, object], node).get(segment)  # cast-ok: endpoint tables are keyed by name
        elif isinstance(node, types.ModuleType) and (submodule := _submodule(path)) is not None:
            node = submodule
        else:
            node = cast(object, getattr(node, segment, None))  # cast-ok: any attribute
        if node is None:
            return None
    return node


def _submodule(path: str) -> types.ModuleType | None:
    try:
        return importlib.import_module(path)
    except ModuleNotFoundError:
        return None
