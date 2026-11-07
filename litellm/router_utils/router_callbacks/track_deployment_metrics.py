"""
Helper functions to get/set num success and num failures per deployment


set_deployment_failures_for_current_minute
set_deployment_successes_for_current_minute

get_deployment_failures_for_current_minute
get_deployment_successes_for_current_minute
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm.constants import ROUTER_BUDGET_ATTRIBUTED_SPEND_CALL_DETAIL_KEY, ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY
from litellm.router_deprecated import renamed_router_param

if TYPE_CHECKING:
    from litellm.router_api import RoutingContext

_METADATA_CHANNELS: Final = ("litellm_metadata", "metadata")


def find_deployment_metadata(kwargs: Mapping[str, object]) -> dict[str, object] | None:
    buckets: Final = (kwargs.get(channel) for channel in _METADATA_CHANNELS)
    return next((bucket for bucket in buckets if isinstance(bucket, dict) and "model_info" in bucket), None)


def get_counted_usage_tokens(litellm_params: Mapping[str, object]) -> int | None:
    buckets: Final = (litellm_params.get(channel) for channel in _METADATA_CHANNELS)
    counted: Final = next(
        (
            bucket[ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY]
            for bucket in buckets
            if isinstance(bucket, dict) and ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY in bucket
        ),
        None,
    )
    return counted if isinstance(counted, int) and not isinstance(counted, bool) else None


def attributed_spend_marker(litellm_call_id: object, spend: float) -> Mapping[str, object]:
    """The marker the Router stamps on a request's logging object when it has charged part of the request's spend
    to a deployment's budget ahead of the success callback (a cascade's cheap leg). Bound to the request's call id."""
    return {"litellm_call_id": litellm_call_id if isinstance(litellm_call_id, str) else None, "spend": spend}


def get_attributed_spend(callback_kwargs: Mapping[str, object], litellm_call_id: object) -> float | None:
    """Spend of THIS request (by call id) already charged to a deployment's and its provider's budgets by the
    Router, read off the success callback's kwargs (the request's logging object's own details, which a caller's
    reused metadata dict cannot reach), so the callback charges the row's deployment only the rest. None when
    nothing was attributed for this call id."""
    marker: Final[object] = callback_kwargs.get(ROUTER_BUDGET_ATTRIBUTED_SPEND_CALL_DETAIL_KEY)
    if not isinstance(marker, Mapping) or not isinstance(litellm_call_id, str):
        return None
    if marker.get("litellm_call_id") != litellm_call_id:  # pyright: ignore[reportUnknownMemberType]  # as above
        return None
    spend: Final[object] = marker.get("spend")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # as above
    return float(spend) if isinstance(spend, (int, float)) and not isinstance(spend, bool) else None


@renamed_router_param("litellm_router_instance")
def increment_deployment_successes_for_current_minute(
    ctx: "RoutingContext",
    deployment_id: str,
) -> str:
    """
    In-Memory: Increments the number of successes for the current minute for a deployment_id
    """
    key: Final = f"{deployment_id}:successes"
    ctx.health().store.increment_cache(
        local_only=True,
        key=key,
        value=1,
        ttl=60,
    )
    return key


@renamed_router_param("litellm_router_instance")
def increment_deployment_failures_for_current_minute(
    ctx: "RoutingContext",
    deployment_id: str,
):
    """
    In-Memory: Increments the number of failures for the current minute for a deployment_id
    """
    key: Final = f"{deployment_id}:fails"
    ctx.health().store.increment_cache(
        local_only=True,
        key=key,
        value=1,
        ttl=60,
    )


@renamed_router_param("litellm_router_instance")
def get_deployment_successes_for_current_minute(
    ctx: "RoutingContext",
    deployment_id: str,
) -> int:
    """
    Returns the number of successes for the current minute for a deployment_id

    Returns 0 if no value found
    """
    key: Final = f"{deployment_id}:successes"
    return (
        ctx.health().store.get_cache(
            local_only=True,
            key=key,
        )
        or 0
    )


@renamed_router_param("litellm_router_instance")
def get_deployment_failures_for_current_minute(
    ctx: "RoutingContext",
    deployment_id: str,
) -> int:
    """
    Returns the number of fails for the current minute for a deployment_id

    Returns 0 if no value found
    """
    key: Final = f"{deployment_id}:fails"
    return (
        ctx.health().store.get_cache(
            local_only=True,
            key=key,
        )
        or 0
    )
