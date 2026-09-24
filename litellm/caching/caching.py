# +-----------------------------------------------+
# |                                               |
# |           Give Feedback / Get Help            |
# | https://github.com/BerriAI/litellm/issues/new |
# |                                               |
# +-----------------------------------------------+
#
#  Thank you users! We ❤️ you! - Krrish & Ishaan

import time
from collections.abc import Iterator, Sequence
from enum import Enum
from typing import Final

from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.constants import CACHED_STREAMING_CHUNK_DELAY
from litellm.rust_bridge._native import Cache
from litellm.types.caching import *

from .azure_blob_cache import AzureBlobCache
from .base_cache import BaseCache
from .disk_cache import DiskCache
from .dual_cache import DualCache
from .gcs_cache import GCSCache
from .in_memory_cache import InMemoryCache
from .qdrant_semantic_cache import QdrantSemanticCache
from .redis_cache import RedisCache, log_redis_failure
from .redis_cluster_cache import RedisClusterCache
from .redis_semantic_cache import RedisSemanticCache
from .s3_cache import S3Cache

__all__ = (
    "AzureBlobCache",
    "BaseCache",
    "Cache",
    "CacheMode",
    "DiskCache",
    "DualCache",
    "GCSCache",
    "InMemoryCache",
    "LiteLLMCacheType",
    "QdrantSemanticCache",
    "RedisCache",
    "RedisClusterCache",
    "RedisSemanticCache",
    "S3Cache",
    "disable_cache",
    "enable_cache",
    "log_redis_failure",
    "print_verbose",
    "update_cache",
)


def print_verbose(print_statement):
    try:
        verbose_logger.debug(print_statement)
        if litellm.set_verbose:
            print(print_statement)  # noqa: T201
    except Exception:
        pass


class CacheMode(str, Enum):
    default_on = "default_on"
    default_off = "default_off"


class _ReplayDelta(TypedDict):
    role: ReadOnly[str]
    content: ReadOnly[str]


class _ReplayChoice(TypedDict):
    delta: ReadOnly[_ReplayDelta]


class _ReplayChunk(TypedDict):
    choices: ReadOnly[Sequence[_ReplayChoice]]


def _generate_streaming_content(content: str) -> Iterator[_ReplayChunk]:
    """`Cache.generate_streaming_content`: a cached completion replayed as assistant chunks."""
    chunk_size: Final = 5
    for i in range(0, len(content), chunk_size):
        delta: Final[_ReplayDelta] = {"role": "assistant", "content": content[i : i + chunk_size]}
        choice: Final[_ReplayChoice] = {"delta": delta}
        chunk: Final[_ReplayChunk] = {"choices": (choice,)}
        yield chunk
        time.sleep(CACHED_STREAMING_CHUNK_DELAY)


def enable_cache(
    type: LiteLLMCacheType | None = LiteLLMCacheType.LOCAL,
    host: str | None = None,
    port: str | None = None,
    password: str | None = None,
    supported_call_types: list[CachingSupportedCallTypes] | None = list(DEFAULT_CACHING_SUPPORTED_CALL_TYPES),
    **kwargs,
):
    """
    Enable cache with the specified configuration.

    Args:
        type (Optional[Literal["local", "redis", "s3", "disk"]]): The type of cache to enable. Defaults to "local".
        host (Optional[str]): The host address of the cache server. Defaults to None.
        port (Optional[str]): The port number of the cache server. Defaults to None.
        password (Optional[str]): The password for the cache server. Defaults to None.
        supported_call_types (Optional[List[Literal["completion", "acompletion", "embedding", "aembedding"]]]):
            The supported call types for the cache. Defaults to ["completion", "acompletion", "embedding", "aembedding"].
        **kwargs: Additional keyword arguments.

    Returns:
        None

    Raises:
        None
    """
    print_verbose("LiteLLM: Enabling Cache")
    if "cache" not in litellm.input_callback:
        litellm.input_callback.append("cache")
    if "cache" not in litellm.success_callback:
        litellm.logging_callback_manager.add_litellm_success_callback("cache")
    if "cache" not in litellm._async_success_callback:
        litellm.logging_callback_manager.add_litellm_async_success_callback("cache")

    if litellm.cache is None:
        litellm.cache = Cache(
            type=type,
            host=host,
            port=port,
            password=password,
            supported_call_types=supported_call_types,
            **kwargs,
        )
    print_verbose(f"LiteLLM: Cache enabled, litellm.cache={litellm.cache}")
    print_verbose(f"LiteLLM Cache: {vars(litellm.cache)}")


def update_cache(
    type: LiteLLMCacheType | None = LiteLLMCacheType.LOCAL,
    host: str | None = None,
    port: str | None = None,
    password: str | None = None,
    supported_call_types: list[CachingSupportedCallTypes] | None = list(DEFAULT_CACHING_SUPPORTED_CALL_TYPES),
    **kwargs,
):
    """
    Update the cache for LiteLLM.

    Args:
        type (Optional[Literal["local", "redis", "s3", "disk"]]): The type of cache. Defaults to "local".
        host (Optional[str]): The host of the cache. Defaults to None.
        port (Optional[str]): The port of the cache. Defaults to None.
        password (Optional[str]): The password for the cache. Defaults to None.
        supported_call_types (Optional[List[Literal["completion", "acompletion", "embedding", "aembedding"]]]):
            The supported call types for the cache. Defaults to ["completion", "acompletion", "embedding", "aembedding"].
        **kwargs: Additional keyword arguments for the cache.

    Returns:
        None

    """
    print_verbose("LiteLLM: Updating Cache")
    litellm.cache = Cache(
        type=type,
        host=host,
        port=port,
        password=password,
        supported_call_types=supported_call_types,
        **kwargs,
    )
    print_verbose(f"LiteLLM: Cache Updated, litellm.cache={litellm.cache}")
    print_verbose(f"LiteLLM Cache: {vars(litellm.cache)}")


def disable_cache():
    """
    Disable the cache used by LiteLLM.

    This function disables the cache used by the LiteLLM module. It removes the cache-related callbacks from the input_callback, success_callback, and _async_success_callback lists. It also sets the litellm.cache attribute to None.

    Parameters:
    None

    Returns:
    None
    """
    from contextlib import suppress

    print_verbose("LiteLLM: Disabling Cache")
    with suppress(ValueError):
        litellm.input_callback.remove("cache")
        litellm.success_callback.remove("cache")
        litellm._async_success_callback.remove("cache")

    litellm.cache = None
    print_verbose(f"LiteLLM: Cache disabled, litellm.cache={litellm.cache}")
