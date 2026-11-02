"""
A deployment's ``max_parallel_requests`` slot on the Rust router core (``_deployment_slot`` and
``_get_client(client_type="max_parallel_requests")``): the attempt holds one of the deployment's
slots around its provider call, and an open chat stream keeps it until the stream is exhausted,
closed, or falls back. A caller arriving while every slot is taken gets a 429, which the core
records as that attempt's failure.

The limit object is the one the Python router caches (``MaxParallelRequestsLimit``), kept in the
core's routing cache under the same key, so its in-flight count is shared by every attempt.
"""

import contextlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm._internal_context import with_service_target
from litellm.router_rust_shaping import ROUTER_CLIENTS_TARGET, as_dict

if TYPE_CHECKING:
    from litellm.router_rust_core import RustCore
    from litellm.router_rust_endpoints import Plan

# The attempt helpers that run inside `_deployment_slot` (`_acompletion` holds the same slot
# itself). Sync helpers, rerank, moderation and the search helper hold none.
_SLOTTED_KINDS: Final = frozenset(
    {
        "chat",
        "text",
        "embedding",
        "image",
        "transcription",
        "speech",
        "adapter",
        "generic",
        "batch",
        "cancel_batch",
        "vector_store_create",
    }
)


@with_service_target(ROUTER_CLIENTS_TARGET)
def max_parallel_requests_limit(
    core: "RustCore", deployment: Mapping[str, object], kwargs: Mapping[str, object]
) -> object:
    """`_get_client(client_type="max_parallel_requests")`: the cached limit, built on first use
    (`InitalizeCachedClient.set_max_parallel_requests_client`)."""
    from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs
    from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit
    from litellm.utils import calculate_max_parallel_requests

    model_id: Final = as_dict(deployment.get("model_info")).get("id")
    key: Final = f"{model_id}_max_parallel_requests_client"
    span: Final = get_parent_otel_span_from_kwargs(dict(kwargs))
    cache: Final = core._cache  # pyright: ignore[reportPrivateUsage]  # the core's routing cache
    cached: Final[object] = cache.get_cache(key=key, local_only=True, parent_otel_span=span)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportArgumentType]  # the cache holds untyped values
    if cached is not None:
        return cached  # pyright: ignore[reportUnknownVariableType]  # the cache holds untyped values
    params: Final = as_dict(deployment.get("litellm_params"))
    calculated: Final = calculate_max_parallel_requests(
        rpm=params.get("rpm"),  # pyright: ignore[reportArgumentType]  # the configured value, as the Python router reads it
        max_parallel_requests=params.get("max_parallel_requests"),  # pyright: ignore[reportArgumentType]  # same
        tpm=params.get("tpm"),  # pyright: ignore[reportArgumentType]  # same
        default_max_parallel_requests=core._endpoints.default_max_parallel_requests,  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # the setting as given, as the Python router passes it
    )
    if calculated:
        cache.set_cache(  # pyright: ignore[reportUnknownMemberType]  # the routing cache
            key=key,
            value=MaxParallelRequestsLimit(
                max_parallel_requests=calculated,
                model_id=str(model_id),
                model_group=str(deployment.get("model_name", "")),
            ),
            local_only=True,
        )
    return cache.get_cache(key=key, local_only=True, parent_otel_span=span)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportArgumentType]  # see above


def enter_deployment_slot(
    core: "RustCore", plan: "Plan", deployment: Mapping[str, object], kwargs: Mapping[str, object]
) -> contextlib.ExitStack:
    """The attempt's slot, entered (a full deployment raises its 429 here); an empty stack when
    the plan's helper holds none or the deployment has no limit."""
    from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit

    slot: Final = contextlib.ExitStack()
    if plan.sync or plan.kind not in _SLOTTED_KINDS:
        return slot
    limit: Final = max_parallel_requests_limit(core, deployment, kwargs)
    if isinstance(limit, MaxParallelRequestsLimit):
        slot.enter_context(limit)
    return slot


def deployment_slot(
    core: "RustCore", deployment: Mapping[str, object], kwargs: Mapping[str, object]
) -> contextlib.ExitStack:
    """`_deployment_slot` outside a plan's attempt (a file created on every deployment)."""
    from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit

    slot: Final = contextlib.ExitStack()
    limit: Final = max_parallel_requests_limit(core, deployment, kwargs)
    if isinstance(limit, MaxParallelRequestsLimit):
        slot.enter_context(limit)
    return slot


__all__ = ["deployment_slot", "enter_deployment_slot", "max_parallel_requests_limit"]
