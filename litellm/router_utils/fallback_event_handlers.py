from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.main import verbose_logger

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
    Loops through all the fallback model groups and calls kwargs["original_function"] with the arguments and keyword arguments provided.

    If the call is successful, it logs the success and returns the response.
    If the call fails, it logs the failure and continues to the next fallback model group.
    If all fallback model groups fail, it raises the most recent exception.

    Args:
        litellm_router: The litellm router instance.
        *args: Positional arguments.
        fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
        original_model_group: The original model group. example: "gpt-3.5-turbo"
        original_exception: The original exception.
        **kwargs: Keyword arguments.

    Returns:
        The response from the successful fallback model group.
    Raises:
        The most recent exception if all fallback model groups fail.
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
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            return response
        except Exception as e:
            error_from_fallbacks = e
            await log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
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
    Synchronous version of run_async_fallback.
    Loops through all the fallback model groups and calls kwargs["original_function"] with the arguments and keyword arguments provided.

    If the call is successful, returns the response.
    If the call fails, continues to the next fallback model group.
    If all fallback model groups fail, it raises the most recent exception.

    Args:
        litellm_router: The litellm router instance.
        *args: Positional arguments.
        fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
        original_model_group: The original model group. example: "gpt-3.5-turbo"
        original_exception: The original exception.
        **kwargs: Keyword arguments.

    Returns:
        The response from the successful fallback model group.
    Raises:
        The most recent exception if all fallback model groups fail.
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


async def log_success_fallback_event(
    original_model_group: str, kwargs: dict, original_exception: Exception
):
    """
    Log a successful fallback event to all registered callbacks.

    This function iterates through all callbacks, initializing _known_custom_logger_compatible_callbacks  if needed,
    and calls the log_success_fallback_event method on CustomLogger instances.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    from litellm.litellm_core_utils.litellm_logging import (
        _init_custom_logger_compatible_class,
    )

    for _callback in litellm.callbacks:
        if isinstance(_callback, CustomLogger) or (
            _callback in litellm._known_custom_logger_compatible_callbacks
        ):
            try:
                _callback_custom_logger: Optional[CustomLogger] = None
                if _callback in litellm._known_custom_logger_compatible_callbacks:
                    _callback_custom_logger = _init_custom_logger_compatible_class(
                        logging_integration=_callback,  # type: ignore
                        llm_router=None,
                        internal_usage_cache=None,
                    )
                elif isinstance(_callback, CustomLogger):
                    _callback_custom_logger = _callback
                else:
                    verbose_router_logger.exception(
                        f"{_callback} logger not found / initialized properly"
                    )
                    continue

                if _callback_custom_logger is None:
                    verbose_router_logger.exception(
                        f"{_callback} logger not found / initialized properly, callback is None"
                    )
                    continue

                await _callback_custom_logger.log_success_fallback_event(
                    original_model_group=original_model_group,
                    kwargs=kwargs,
                    original_exception=original_exception,
                )
            except Exception as e:
                verbose_router_logger.error(
                    f"Error in log_success_fallback_event: {str(e)}"
                )


async def log_failure_fallback_event(
    original_model_group: str, kwargs: dict, original_exception: Exception
):
    """
    Log a failed fallback event to all registered callbacks.

    This function iterates through all callbacks, initializing _known_custom_logger_compatible_callbacks if needed,
    and calls the log_failure_fallback_event method on CustomLogger instances.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    from litellm.litellm_core_utils.litellm_logging import (
        _init_custom_logger_compatible_class,
    )

    for _callback in litellm.callbacks:
        if isinstance(_callback, CustomLogger) or (
            _callback in litellm._known_custom_logger_compatible_callbacks
        ):
            try:
                _callback_custom_logger: Optional[CustomLogger] = None
                if _callback in litellm._known_custom_logger_compatible_callbacks:
                    _callback_custom_logger = _init_custom_logger_compatible_class(
                        logging_integration=_callback,  # type: ignore
                        llm_router=None,
                        internal_usage_cache=None,
                    )
                elif isinstance(_callback, CustomLogger):
                    _callback_custom_logger = _callback
                else:
                    verbose_router_logger.exception(
                        f"{_callback} logger not found / initialized properly"
                    )
                    continue

                if _callback_custom_logger is None:
                    verbose_router_logger.exception(
                        f"{_callback} logger not found / initialized properly"
                    )
                    continue

                await _callback_custom_logger.log_failure_fallback_event(
                    original_model_group=original_model_group,
                    kwargs=kwargs,
                    original_exception=original_exception,
                )
            except Exception as e:
                verbose_router_logger.error(
                    f"Error in log_failure_fallback_event: {str(e)}"
                )
