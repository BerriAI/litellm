from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


async def run_async_fallback(
    litellm_router: LitellmRouter,
    *args: Tuple[Any],
    fallback_model_group: List[str],
    original_model_group: str,
    original_exception: Exception,
    **kwargs,
) -> Any:
    """
    Iterate through the model groups and try calling that deployment.
    """
    error_from_fallbacks = original_exception
    for mg in fallback_model_group:
        if mg == original_model_group:
            continue
        try:
            # LOGGING
            kwargs = litellm_router.log_retry(kwargs=kwargs, e=original_exception)
            verbose_router_logger.info(f"Falling back to model_group = {mg}")
            kwargs["model"] = mg
            kwargs.setdefault("metadata", {}).update(
                {"model_group": mg}
            )  # update model_group used, if fallbacks are done
            response = await litellm_router.async_function_with_fallbacks(
                *args, **kwargs
            )
            verbose_router_logger.info("Successful fallback b/w models.")
            # callback for successfull_fallback_event():
            await log_success_fallback_event(
                original_model_group=original_model_group, kwargs=kwargs
            )
            return response
        except Exception as e:
            error_from_fallbacks = e
            await log_failure_fallback_event(
                original_model_group=original_model_group, kwargs=kwargs
            )
    raise error_from_fallbacks


def run_sync_fallback(
    litellm_router: LitellmRouter,
    *args: Tuple[Any],
    fallback_model_group: List[str],
    original_model_group: str,
    original_exception: Exception,
    **kwargs,
) -> Any:
    """
    Iterate through the model groups and try calling that deployment.
    """
    error_from_fallbacks = original_exception
    for mg in fallback_model_group:
        if mg == original_model_group:
            continue
        try:
            # LOGGING
            kwargs = litellm_router.log_retry(kwargs=kwargs, e=original_exception)
            verbose_router_logger.info(f"Falling back to model_group = {mg}")
            kwargs["model"] = mg
            kwargs.setdefault("metadata", {}).update(
                {"model_group": mg}
            )  # update model_group used, if fallbacks are done
            response = litellm_router.function_with_fallbacks(*args, **kwargs)
            verbose_router_logger.info("Successful fallback b/w models.")
            return response
        except Exception as e:
            error_from_fallbacks = e
    raise error_from_fallbacks


async def log_success_fallback_event(original_model_group: str, kwargs: dict):
    for _callback in litellm.callbacks:
        if isinstance(_callback, CustomLogger):
            try:
                await _callback.log_success_fallback_event(
                    original_model_group=original_model_group, kwargs=kwargs
                )
            except Exception as e:
                verbose_router_logger.error(
                    f"Error in log_success_fallback_event: {(str(e))}"
                )
                pass


async def log_failure_fallback_event(original_model_group: str, kwargs: dict):
    for _callback in litellm.callbacks:
        if isinstance(_callback, CustomLogger):
            try:
                await _callback.log_failure_fallback_event(
                    original_model_group=original_model_group, kwargs=kwargs
                )
            except Exception as e:
                verbose_router_logger.error(
                    f"Error in log_failure_fallback_event: {(str(e))}"
                )
                pass
