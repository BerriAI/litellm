"""
A deployment's ``max_parallel_requests`` slot on the Rust router core (``_deployment_slot`` and
``_get_client(client_type="max_parallel_requests")``): the attempt holds one of the deployment's
slots around its provider call, an open chat stream keeps it until the stream is exhausted, closed,
or falls back, and a streamed speech response keeps it until its audio ends. A caller arriving while
every slot is taken gets a 429, which the core records as that attempt's failure.

The limit object is the one the Python router caches (``MaxParallelRequestsLimit``), kept in the
core's routing cache under the same key, so its in-flight count is shared by every attempt.
"""

import contextlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Final

import httpx

from litellm._internal_context import with_service_target
from litellm.litellm_core_utils.on_end_byte_stream import OnEndAsyncByteStream, StreamEnd
from litellm.router_rust.invoke.shaping import ROUTER_CLIENTS_TARGET, as_dict

if TYPE_CHECKING:
    from litellm.router_rust.invoke.endpoints import EndpointCore, Plan

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
    core: "EndpointCore", deployment: Mapping[str, object], kwargs: Mapping[str, object]
) -> object:
    """`_get_client(client_type="max_parallel_requests")`: the cached limit, built on first use
    (`InitalizeCachedClient.set_max_parallel_requests_client`)."""
    from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs
    from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit
    from litellm.utils import calculate_max_parallel_requests

    model_id: Final = as_dict(deployment.get("model_info")).get("id")
    key: Final = f"{model_id}_max_parallel_requests_client"
    span: Final = get_parent_otel_span_from_kwargs(dict(kwargs))
    cache: Final = core.routing_cache
    cached: Final[object] = cache.get_cache(key=key, local_only=True, parent_otel_span=span)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportArgumentType]  # the cache holds untyped values
    if cached is not None:
        return cached  # pyright: ignore[reportUnknownVariableType]  # the cache holds untyped values
    params: Final = as_dict(deployment.get("litellm_params"))
    calculated: Final = calculate_max_parallel_requests(
        rpm=params.get("rpm"),  # pyright: ignore[reportArgumentType]  # the configured value, as the Python router reads it
        max_parallel_requests=params.get("max_parallel_requests"),  # pyright: ignore[reportArgumentType]  # same
        tpm=params.get("tpm"),  # pyright: ignore[reportArgumentType]  # same
        default_max_parallel_requests=core.endpoints.default_max_parallel_requests,  # pyright: ignore[reportArgumentType]  # the setting as given, as the Python router passes it
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
    core: "EndpointCore", plan: "Plan", deployment: Mapping[str, object], kwargs: Mapping[str, object]
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
    core: "EndpointCore", deployment: Mapping[str, object], kwargs: Mapping[str, object]
) -> contextlib.ExitStack:
    """`_deployment_slot` outside a plan's attempt (a file created on every deployment)."""
    from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit

    slot: Final = contextlib.ExitStack()
    limit: Final = max_parallel_requests_limit(core, deployment, kwargs)
    if isinstance(limit, MaxParallelRequestsLimit):
        slot.enter_context(limit)
    return slot


def hold_slot_until_closed(
    response: object, attempt_slot: contextlib.ExitStack, on_end: Callable[[StreamEnd], None]
) -> bool:
    """Whether ``response`` took the attempt's slot. A body already in hand leaves the slot with the
    attempt. A body still arriving from the provider takes it, or the deployment's
    max_parallel_requests would only bound how many responses can start: it is held until the body
    is read to the end, the read fails, or the response is closed, and ``on_end`` runs then."""
    raw: Final = getattr(response, "response", None)
    if (
        not isinstance(raw, httpx.Response)
        or raw.is_stream_consumed
        or not isinstance(raw.stream, httpx.AsyncByteStream)
    ):
        return False
    held: Final = attempt_slot.pop_all()

    async def release(end: StreamEnd) -> None:
        held.close()
        on_end(end)

    raw.stream = OnEndAsyncByteStream(raw.stream, map_error=lambda error: error, on_end=release)
    return True


__all__ = ["deployment_slot", "enter_deployment_slot", "hold_slot_until_closed", "max_parallel_requests_limit"]
