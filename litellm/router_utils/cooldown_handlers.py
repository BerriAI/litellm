"""
Router cooldown handlers
- _set_cooldown_deployments: puts a deployment in the cooldown list
- get_cooldown_deployments: returns the list of deployments in the cooldown list
- async_get_cooldown_deployments: ASYNC: returns the list of deployments in the cooldown list

"""

import asyncio
from typing import TYPE_CHECKING, Any, List, Optional, Union

import litellm
from litellm._logging import verbose_router_logger
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback
from litellm.utils import get_utc_datetime

from .router_callbacks.track_deployment_metrics import (
    get_deployment_failures_for_current_minute,
    get_deployment_successes_for_current_minute,
)

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any

DEFAULT_FAILURE_THRESHOLD_PERCENT = (
    0.5  # default cooldown a deployment if 50% of requests fail in a given minute
)
DEFAULT_COOLDOWN_TIME_SECONDS = 5


def _should_run_cooldown_logic(
    litellm_router_instance: LitellmRouter,
    deployment: Optional[str],
    exception_status: Union[str, int],
    original_exception: Any,
) -> bool:
    """
    Helper that decides if cooldown logic should be run
    Returns False if cooldown logic should not be run

    Does not run cooldown logic when:
    - router.disable_cooldowns is True
    - deployment is None
    - _is_cooldown_required() returns False
    - deployment is in litellm_router_instance.provider_default_deployment_ids
    - exception_status is not one that should be immediately retried (e.g. 401)
    """
    if litellm_router_instance.disable_cooldowns:
        return False

    if deployment is None:
        return False

    if not litellm_router_instance._is_cooldown_required(
        model_id=deployment,
        exception_status=exception_status,
        exception_str=str(original_exception),
    ):
        return False

    if deployment in litellm_router_instance.provider_default_deployment_ids:
        return False

    return True


def _should_cooldown_deployment(
    litellm_router_instance: LitellmRouter,
    deployment: str,
    exception_status: Union[str, int],
    original_exception: Any,
) -> bool:
    """
    Helper that decides if a deployment should be put in cooldown

    Returns True if the deployment should be put in cooldown
    Returns False if the deployment should not be put in cooldown


    Deployment is put in cooldown when:
    - v2 logic (Current):
    cooldown if:
        - got a 429 error from LLM API
        - if %fails/%(successes + fails) > ALLOWED_FAILURE_RATE_PER_MINUTE
        - got 401 Auth error, 404 NotFounder - checked by litellm._should_retry()



    - v1 logic (Legacy): if allowed fails or allowed fail policy set, coolsdown if num fails in this minute > allowed fails
    """
    if litellm_router_instance.allowed_fails_policy is None:
        num_successes_this_minute = get_deployment_successes_for_current_minute(
            litellm_router_instance=litellm_router_instance, deployment_id=deployment
        )
        num_fails_this_minute = get_deployment_failures_for_current_minute(
            litellm_router_instance=litellm_router_instance, deployment_id=deployment
        )

        total_requests_this_minute = num_successes_this_minute + num_fails_this_minute
        percent_fails = 0.0
        if total_requests_this_minute > 0:
            percent_fails = num_fails_this_minute / (
                num_successes_this_minute + num_fails_this_minute
            )
        verbose_router_logger.debug(
            "percent fails for deployment = %s, percent fails = %s, num successes = %s, num fails = %s",
            deployment,
            percent_fails,
            num_successes_this_minute,
            num_fails_this_minute,
        )
        exception_status_int = cast_exception_status_to_int(exception_status)
        if exception_status_int == 429:
            return True
        elif (
            total_requests_this_minute == 1
        ):  # if the 1st request fails it's not guaranteed that the deployment should be cooled down
            return False
        elif percent_fails > DEFAULT_FAILURE_THRESHOLD_PERCENT:
            return True

        elif (
            litellm._should_retry(
                status_code=cast_exception_status_to_int(exception_status)
            )
            is False
        ):
            return True

        return False
    else:
        return should_cooldown_based_on_allowed_fails_policy(
            litellm_router_instance=litellm_router_instance,
            deployment=deployment,
            original_exception=original_exception,
        )

    return False


def _set_cooldown_deployments(
    litellm_router_instance: LitellmRouter,
    original_exception: Any,
    exception_status: Union[str, int],
    deployment: Optional[str] = None,
    time_to_cooldown: Optional[float] = None,
):
    """
    Add a model to the list of models being cooled down for that minute, if it exceeds the allowed fails / minute

    or

    the exception is not one that should be immediately retried (e.g. 401)
    """
    if (
        _should_run_cooldown_logic(
            litellm_router_instance, deployment, exception_status, original_exception
        )
        is False
        or deployment is None
    ):
        return

    exception_status_int = cast_exception_status_to_int(exception_status)

    verbose_router_logger.debug(f"Attempting to add {deployment} to cooldown list")
    cooldown_time = litellm_router_instance.cooldown_time or 1
    if time_to_cooldown is not None:
        cooldown_time = time_to_cooldown

    if _should_cooldown_deployment(
        litellm_router_instance, deployment, exception_status, original_exception
    ):
        litellm_router_instance.cooldown_cache.add_deployment_to_cooldown(
            model_id=deployment,
            original_exception=original_exception,
            exception_status=exception_status_int,
            cooldown_time=cooldown_time,
        )

        # Trigger cooldown callback handler
        asyncio.create_task(
            router_cooldown_event_callback(
                litellm_router_instance=litellm_router_instance,
                deployment_id=deployment,
                exception_status=exception_status,
                cooldown_time=cooldown_time,
            )
        )


async def _async_get_cooldown_deployments(
    litellm_router_instance: LitellmRouter,
) -> List[str]:
    """
    Async implementation of '_get_cooldown_deployments'
    """
    model_ids = litellm_router_instance.get_model_ids()
    cooldown_models = (
        await litellm_router_instance.cooldown_cache.async_get_active_cooldowns(
            model_ids=model_ids
        )
    )

    cached_value_deployment_ids = []
    if (
        cooldown_models is not None
        and isinstance(cooldown_models, list)
        and len(cooldown_models) > 0
        and isinstance(cooldown_models[0], tuple)
    ):
        cached_value_deployment_ids = [cv[0] for cv in cooldown_models]

    verbose_router_logger.debug(f"retrieve cooldown models: {cooldown_models}")
    return cached_value_deployment_ids


async def _async_get_cooldown_deployments_with_debug_info(
    litellm_router_instance: LitellmRouter,
) -> List[tuple]:
    """
    Async implementation of '_get_cooldown_deployments'
    """
    model_ids = litellm_router_instance.get_model_ids()
    cooldown_models = (
        await litellm_router_instance.cooldown_cache.async_get_active_cooldowns(
            model_ids=model_ids
        )
    )

    verbose_router_logger.debug(f"retrieve cooldown models: {cooldown_models}")
    return cooldown_models


def _get_cooldown_deployments(litellm_router_instance: LitellmRouter) -> List[str]:
    """
    Get the list of models being cooled down for this minute
    """
    # get the current cooldown list for that minute

    # ----------------------
    # Return cooldown models
    # ----------------------
    model_ids = litellm_router_instance.get_model_ids()
    cooldown_models = litellm_router_instance.cooldown_cache.get_active_cooldowns(
        model_ids=model_ids
    )

    cached_value_deployment_ids = []
    if (
        cooldown_models is not None
        and isinstance(cooldown_models, list)
        and len(cooldown_models) > 0
        and isinstance(cooldown_models[0], tuple)
    ):
        cached_value_deployment_ids = [cv[0] for cv in cooldown_models]

    return cached_value_deployment_ids


def should_cooldown_based_on_allowed_fails_policy(
    litellm_router_instance: LitellmRouter,
    deployment: str,
    original_exception: Any,
) -> bool:
    """
    Check if fails are within the allowed limit and update the number of fails.

    Returns:
    - True if fails exceed the allowed limit (should cooldown)
    - False if fails are within the allowed limit (should not cooldown)
    """
    allowed_fails = (
        litellm_router_instance.get_allowed_fails_from_policy(
            exception=original_exception,
        )
        or litellm_router_instance.allowed_fails
    )
    cooldown_time = (
        litellm_router_instance.cooldown_time or DEFAULT_COOLDOWN_TIME_SECONDS
    )

    current_fails = litellm_router_instance.failed_calls.get_cache(key=deployment) or 0
    updated_fails = current_fails + 1

    if updated_fails > allowed_fails:
        return True
    else:
        litellm_router_instance.failed_calls.set_cache(
            key=deployment, value=updated_fails, ttl=cooldown_time
        )

    return False


def cast_exception_status_to_int(exception_status: Union[str, int]) -> int:
    if isinstance(exception_status, str):
        try:
            exception_status = int(exception_status)
        except Exception:
            verbose_router_logger.debug(
                f"Unable to cast exception status to int {exception_status}. Defaulting to status=500."
            )
            exception_status = 500
    return exception_status
