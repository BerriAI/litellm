from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import LiteLLMParamsTypedDict

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any


def _check_stripped_model_group(model_group: str, fallback_key: str) -> bool:
    """
    Handles wildcard routing scenario

    where fallbacks set like:
    [{"gpt-3.5-turbo": ["claude-3-haiku"]}]

    but model_group is like:
    "openai/gpt-3.5-turbo"

    Returns:
    - True if the stripped model group == fallback_key
    """
    for provider in litellm.provider_list:
        if isinstance(provider, Enum):
            _provider = provider.value
        else:
            _provider = provider
        if model_group.startswith(f"{_provider}/"):
            stripped_model_group = model_group.replace(f"{_provider}/", "")
            if stripped_model_group == fallback_key:
                return True
    return False


def get_fallback_model_group(
    fallbacks: List[Any], model_group: str
) -> Tuple[Optional[List[str]], Optional[int]]:
    """
    Returns:
    - fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
    - generic_fallback_idx: int of the index of the generic fallback in the fallbacks list.

    Checks:
    - exact match
    - stripped model group match
    - generic fallback
    """
    generic_fallback_idx: Optional[int] = None
    stripped_model_fallback: Optional[List[str]] = None
    fallback_model_group: Optional[List[str]] = None
    ## check for specific model group-specific fallbacks
    for idx, item in enumerate(fallbacks):
        if isinstance(item, dict):
            if list(item.keys())[0] == model_group:  # check exact match
                fallback_model_group = item[model_group]
                break
            elif _check_stripped_model_group(
                model_group=model_group, fallback_key=list(item.keys())[0]
            ):  # check generic fallback
                stripped_model_fallback = item[list(item.keys())[0]]
            elif list(item.keys())[0] == "*":  # check generic fallback
                generic_fallback_idx = idx
        elif isinstance(item, str):
            fallback_model_group = [fallbacks.pop(idx)]  # returns single-item list
    ## if none, check for generic fallback
    if fallback_model_group is None:
        if stripped_model_fallback is not None:
            fallback_model_group = stripped_model_fallback
        elif generic_fallback_idx is not None:
            fallback_model_group = fallbacks[generic_fallback_idx]["*"]

    return fallback_model_group, generic_fallback_idx


async def run_async_fallback(
    *args: Tuple[Any],
    litellm_router: LitellmRouter,
    fallback_model_group: List[str],
    original_model_group: str,
    original_exception: Exception,
    max_fallbacks: int,
    fallback_depth: int,
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

    ### BASE CASE ### MAX FALLBACK DEPTH REACHED
    if fallback_depth >= max_fallbacks:
        raise original_exception

    error_from_fallbacks = original_exception

    for mg in fallback_model_group:
        if mg == original_model_group:
            continue
        try:
            # LOGGING
            kwargs = litellm_router.log_retry(kwargs=kwargs, e=original_exception)
            verbose_router_logger.info(f"Falling back to model_group = {mg}")
            if isinstance(mg, str):
                kwargs["model"] = mg
            elif isinstance(mg, dict):
                kwargs.update(mg)
            kwargs.setdefault("metadata", {}).update(
                {"model_group": kwargs.get("model", None)}
            )  # update model_group used, if fallbacks are done
            kwargs["fallback_depth"] = fallback_depth + 1
            kwargs["max_fallbacks"] = max_fallbacks
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


def _check_non_standard_fallback_format(fallbacks: Optional[List[Any]]) -> bool:
    """
    Checks if the fallbacks list is a list of strings or a list of dictionaries.

    If
    - List[str]: e.g. ["claude-3-haiku", "openai/o-1"]
    - List[Dict[<LiteLLMParamsTypedDict>, Any]]: e.g. [{"model": "claude-3-haiku", "messages": [{"role": "user", "content": "Hey, how's it going?"}]}]

    If [{"gpt-3.5-turbo": ["claude-3-haiku"]}] then standard format.
    """
    if fallbacks is None or not isinstance(fallbacks, list) or len(fallbacks) == 0:
        return False
    if all(isinstance(item, str) for item in fallbacks):
        return True
    elif all(isinstance(item, dict) for item in fallbacks):
        for key in LiteLLMParamsTypedDict.__annotations__.keys():
            if key in fallbacks[0].keys():
                return True

    return False


def run_non_standard_fallback_format(
    fallbacks: Union[List[str], List[Dict[str, Any]]], model_group: str
):
    pass
