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
from litellm.router_utils.cooldown_callbacks import router_cooldown_handler
from litellm.utils import get_utc_datetime

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


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
    if litellm_router_instance.disable_cooldowns is True:
        return

    if deployment is None:
        return

    if (
        litellm_router_instance._is_cooldown_required(
            model_id=deployment,
            exception_status=exception_status,
            exception_str=str(original_exception),
        )
        is False
    ):
        return

    if deployment in litellm_router_instance.provider_default_deployment_ids:
        return

    _allowed_fails = litellm_router_instance.get_allowed_fails_from_policy(
        exception=original_exception,
    )

    allowed_fails = (
        _allowed_fails
        if _allowed_fails is not None
        else litellm_router_instance.allowed_fails
    )

    dt = get_utc_datetime()
    current_minute = dt.strftime("%H-%M")
    # get current fails for deployment
    # update the number of failed calls
    # if it's > allowed fails
    # cooldown deployment
    current_fails = litellm_router_instance.failed_calls.get_cache(key=deployment) or 0
    updated_fails = current_fails + 1
    verbose_router_logger.debug(
        f"Attempting to add {deployment} to cooldown list. updated_fails: {updated_fails}; litellm_router_instance.allowed_fails: {allowed_fails}"
    )
    cooldown_time = litellm_router_instance.cooldown_time or 1
    if time_to_cooldown is not None:
        cooldown_time = time_to_cooldown

    if isinstance(exception_status, str):
        try:
            exception_status = int(exception_status)
        except Exception as e:
            verbose_router_logger.debug(
                "Unable to cast exception status to int {}. Defaulting to status=500.".format(
                    exception_status
                )
            )
            exception_status = 500
    _should_retry = litellm._should_retry(status_code=exception_status)

    if updated_fails > allowed_fails or _should_retry is False:
        # get the current cooldown list for that minute
        verbose_router_logger.debug(f"adding {deployment} to cooldown models")
        # update value
        litellm_router_instance.cooldown_cache.add_deployment_to_cooldown(
            model_id=deployment,
            original_exception=original_exception,
            exception_status=exception_status,
            cooldown_time=cooldown_time,
        )

        # Trigger cooldown handler
        asyncio.create_task(
            router_cooldown_handler(
                litellm_router_instance=litellm_router_instance,
                deployment_id=deployment,
                exception_status=exception_status,
                cooldown_time=cooldown_time,
            )
        )
    else:
        litellm_router_instance.failed_calls.set_cache(
            key=deployment, value=updated_fails, ttl=cooldown_time
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
