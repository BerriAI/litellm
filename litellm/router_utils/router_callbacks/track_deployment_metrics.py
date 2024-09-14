"""
Helper functions to get/set num success and num failures per deployment 


set_deployment_failures_for_current_minute
set_deployment_successes_for_current_minute

get_deployment_failures_for_current_minute
get_deployment_successes_for_current_minute
"""

from typing import TYPE_CHECKING, Any, Callable, Optional

from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def set_deployment_failures_for_current_minute(
    litellm_router_instance: LitellmRouter,
    deployment_id: str,
    num_fails: int,
):
    """
    Adds a deployment to cooldown when %fails/%successes is greater than ALLOWED_FAILURE_RATE_PER_MINUTE
    """
    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    key = f"{current_minute}:{deployment_id}:fails"
    litellm_router_instance.cache.set_cache(
        local_only=True,
        key=key,
        value=num_fails,
        ttl_seconds=120,
    )


def get_deployment_failures_for_current_minute(
    litellm_router_instance: LitellmRouter,
    deployment_id: str,
) -> int:
    """
    Returns the number of fails for the current minute for a deployment_id

    Returns 0 if no value found
    """
    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    key = f"{current_minute}:{deployment_id}:fails"
    return (
        litellm_router_instance.cache.get_cache(
            local_only=True,
            key=key,
        )
        or 0
    )


def set_deployment_successes_for_current_minute(
    litellm_router_instance: LitellmRouter,
    deployment_id: str,
    num_successes: int,
):
    """
    Sets the number of successes for the current minute for a deployment_id
    """
    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    key = f"{current_minute}:{deployment_id}:successes"
    litellm_router_instance.cache.set_cache(
        local_only=True,
        key=key,
        value=num_successes,
        ttl_seconds=120,
    )


def get_deployment_successes_for_current_minute(
    litellm_router_instance: LitellmRouter,
    deployment_id: str,
) -> int:
    """
    Returns the number of successes for the current minute for a deployment_id

    Returns 0 if no value found
    """
    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    key = f"{current_minute}:{deployment_id}:successes"
    return (
        litellm_router_instance.cache.get_cache(
            local_only=True,
            key=key,
        )
        or 0
    )
