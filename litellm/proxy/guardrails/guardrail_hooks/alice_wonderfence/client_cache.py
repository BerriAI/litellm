"""WonderFence SDK loader + per-api_key LRU client cache."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from wonderfence_sdk.client import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
        WonderFenceV2Client as _WonderFenceV2Client,
    )


@dataclass(frozen=True)
class ClientBuildSpec:
    """How to construct a WonderFenceV2Client on a cache miss."""

    client_class: Callable[..., object]
    api_timeout: float
    api_base: str | None
    platform: str | None
    connection_pool_limit: int | None


def load_sdk() -> tuple[Any, Any]:
    """Lazy-import WonderFence SDK classes (``WonderFenceV2Client``, ``AnalysisContext``).

    Deferred to instance construction (not module load) because wonderfence_sdk
    is an optional dependency: importing it at module top would break litellm
    installs that don't use this guardrail. Callers cache the returned classes
    on the instance so per-call hot paths don't re-trigger the import machinery.
    """
    try:
        from wonderfence_sdk.client import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
            WonderFenceV2Client,
        )
        from wonderfence_sdk.models import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
            AnalysisContext,
        )
    except ImportError as e:
        raise ImportError("Alice WonderFence SDK not installed. Install with: pip install wonderfence-sdk") from e
    return WonderFenceV2Client, AnalysisContext


def get_or_create_client(
    api_key: str,
    cache: "OrderedDict[str, _WonderFenceV2Client]",
    cache_maxsize: int,
    spec: ClientBuildSpec,
) -> "_WonderFenceV2Client":
    """LRU client lookup keyed by ``api_key``; construct on miss."""
    if api_key in cache:
        cache.move_to_end(api_key)
        return cache[api_key]

    client_kwargs: dict = {
        "api_key": api_key,
        "api_timeout": round(spec.api_timeout),
    }
    if spec.api_base:
        client_kwargs["base_url"] = spec.api_base
    if spec.platform:
        client_kwargs["platform"] = spec.platform
    if spec.connection_pool_limit is not None:
        client_kwargs["connection_pool_limit"] = spec.connection_pool_limit

    client = spec.client_class(**client_kwargs)
    cache[api_key] = client

    if len(cache) > cache_maxsize:
        # Drop reference only — never close. An evicted client may still be
        # held by in-flight apply_guardrail coroutines; closing it would
        # break their pooled HTTP connections. GC handles cleanup.
        cache.popitem(last=False)

    return client
