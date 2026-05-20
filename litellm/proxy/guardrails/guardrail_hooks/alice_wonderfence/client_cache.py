"""WonderFence SDK loader + per-api_key LRU client cache."""

from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from wonderfence_sdk.client import (  # type: ignore[import-untyped]
        WonderFenceV2Client as _WonderFenceV2Client,
    )


def load_sdk() -> Tuple[Any, Any]:
    """Lazy-import WonderFence SDK classes (``WonderFenceV2Client``, ``AnalysisContext``).

    Deferred to instance construction (not module load) because wonderfence_sdk
    is an optional dependency: importing it at module top would break litellm
    installs that don't use this guardrail. Callers cache the returned classes
    on the instance so per-call hot paths don't re-trigger the import machinery.
    """
    try:
        from wonderfence_sdk.client import (  # type: ignore[import-untyped]
            WonderFenceV2Client,
        )
        from wonderfence_sdk.models import (  # type: ignore[import-untyped]
            AnalysisContext,
        )
    except ImportError as e:
        raise ImportError(
            "Alice WonderFence SDK not installed. Install with: pip install wonderfence-sdk"
        ) from e
    return WonderFenceV2Client, AnalysisContext


def get_or_create_client(
    api_key: str,
    cache: "OrderedDict[str, _WonderFenceV2Client]",
    cache_maxsize: int,
    client_class: Any,
    api_timeout: float,
    api_base: Optional[str],
    platform: Optional[str],
    connection_pool_limit: Optional[int],
) -> "_WonderFenceV2Client":
    """LRU client lookup keyed by ``api_key``; construct on miss."""
    if api_key in cache:
        cache.move_to_end(api_key)
        return cache[api_key]

    client_kwargs: dict = {
        "api_key": api_key,
        "api_timeout": round(api_timeout),
    }
    if api_base:
        client_kwargs["base_url"] = api_base
    if platform:
        client_kwargs["platform"] = platform
    if connection_pool_limit is not None:
        client_kwargs["connection_pool_limit"] = connection_pool_limit

    client = client_class(**client_kwargs)
    cache[api_key] = client

    if len(cache) > cache_maxsize:
        # Drop reference only — never close. An evicted client may still be
        # held by in-flight apply_guardrail coroutines; closing it would
        # break their pooled HTTP connections. GC handles cleanup.
        cache.popitem(last=False)

    return client
