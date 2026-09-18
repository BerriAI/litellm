"""
Helper functions to get/set num success and num failures per deployment


set_deployment_failures_for_current_minute
set_deployment_successes_for_current_minute

get_deployment_failures_for_current_minute
get_deployment_successes_for_current_minute
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from litellm.constants import ROUTER_USAGE_COUNTED_TOKENS_METADATA_KEY

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any

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


def increment_deployment_successes_for_current_minute(
    litellm_router_instance: LitellmRouter,
    deployment_id: str,
) -> str:
    """
    In-Memory: Increments the number of successes for the current minute for a deployment_id
    """
    key: Final = f"{deployment_id}:successes"
    litellm_router_instance.cache.increment_cache(
        local_only=True,
        key=key,
        value=1,
        ttl=60,
    )
    return key


def increment_deployment_failures_for_current_minute(
    litellm_router_instance: LitellmRouter,
    deployment_id: str,
):
    """
    In-Memory: Increments the number of failures for the current minute for a deployment_id
    """
    key: Final = f"{deployment_id}:fails"
    litellm_router_instance.cache.increment_cache(
        local_only=True,
        key=key,
        value=1,
        ttl=60,
    )


def get_deployment_successes_for_current_minute(
    litellm_router_instance: LitellmRouter,
    deployment_id: str,
) -> int:
    """
    Returns the number of successes for the current minute for a deployment_id

    Returns 0 if no value found
    """
    key: Final = f"{deployment_id}:successes"
    return (
        litellm_router_instance.cache.get_cache(
            local_only=True,
            key=key,
        )
        or 0
    )


def get_deployment_failures_for_current_minute(
    litellm_router_instance: LitellmRouter,
    deployment_id: str,
) -> int:
    """
    Returns the number of fails for the current minute for a deployment_id

    Returns 0 if no value found
    """
    key: Final = f"{deployment_id}:fails"
    return (
        litellm_router_instance.cache.get_cache(
            local_only=True,
            key=key,
        )
        or 0
    )
