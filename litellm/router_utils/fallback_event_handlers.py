import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.sensitive_data_masker import mask_sensitive_structure
from litellm.router_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    get_fallback_error_info,
)
from litellm.router_utils.batch_utils import _get_router_metadata_variable_name
from litellm.router_utils.cooldown_handlers import (
    _first_present,  # pyright: ignore[reportPrivateUsage] - shared internal helper, used across router_utils
    _set_cooldown_deployments,  # pyright: ignore[reportPrivateUsage] - shared helper, used across router_utils
    cast_exception_status_to_int,
    is_advisor_orchestration_failure,
)
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    increment_deployment_failures_for_current_minute,
)
from litellm.types.router import LiteLLMParamsTypedDict

if TYPE_CHECKING:
    from litellm.router import Router as _Router

    LitellmRouter = _Router
else:
    LitellmRouter = Any

# Status codes a generic API call's caller-supplied resource id can trigger on its own
# (e.g. a nonexistent file/batch/thread id), independent of the selected deployment's health.
_REQUEST_SCOPED_STATUS_CODES: Final = frozenset((404,))


def _trigger_cooldown_for_failed_deployment(
    litellm_router: LitellmRouter,
    kwargs: Mapping[str, Any],
    exception: Exception,
) -> None:
    """
    Trigger cooldown for a failed fallback deployment.

    In the fallback path the normal failure-callback cooldown is skipped because the
    Logging object sets has_logged_async_failure=True after the first failure and
    blocks all subsequent failure callbacks. This helper ensures every failed
    fallback deployment is evaluated for cooldown regardless.
    """
    try:
        if is_advisor_orchestration_failure(exception):
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: failure originated "
                "from advisor orchestration, not the selected deployment."
            )
            return

        exception_status: Final[str | int] = getattr(exception, "status_code", "")

        # Generic API calls (files, batches, threads, rerank, ...) take a caller-supplied
        # resource id, so a 404 there usually means "that id doesn't exist" rather than
        # "this deployment is unhealthy". Left unguarded, one bad id would 404 every
        # deployment in the fallback chain and cool all of them down from a single request.
        if (
            kwargs.get("original_generic_function") is not None
            and cast_exception_status_to_int(exception_status) in _REQUEST_SCOPED_STATUS_CODES
        ):
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: status %s on a generic API "
                "call is caller-attributable, not a deployment health signal.",
                exception_status,
            )
            return

        # The proxy's `x-litellm-timeout` header lets a caller set an arbitrarily short
        # timeout, which litellm.Timeout reports as status 408 regardless of the deployment's
        # actual health. Left unguarded, a caller could force a 408 on every deployment in
        # the fallback chain from a single request with a near-zero timeout.
        if kwargs.get("client_side_timeout") and cast_exception_status_to_int(exception_status) == 408:
            verbose_router_logger.debug(
                "Not triggering cooldown for fallback deployment: a caller-supplied "
                "x-litellm-timeout caused this 408, not deployment health."
            )
            return

        # Only Router._set_failed_deployment_id_on_exception()'s server-stamped id is
        # trusted here: a metadata-bucket lookup (e.g. "metadata"/"litellm_metadata")
        # can't reliably tell a caller-supplied bucket from a router-authored one
        # without knowing this call's function_name, so a client with permission to
        # set metadata could otherwise get an arbitrary deployment cooled down.
        deployment_id: Final[str | None] = getattr(exception, "failed_deployment_id", None)

        if deployment_id is None:
            verbose_router_logger.debug("Cannot trigger cooldown for fallback: no failed_deployment_id on exception")
            return

        # Priority: deployment config > response header > router default, matching
        # Router.deployment_callback_on_failure's precedence for the primary path.
        deployment_dict: Final = litellm_router.get_model_info(id=deployment_id)
        deployment_cooldown: Final = (
            _first_present(
                deployment_dict.get("model_info"), deployment_dict.get("litellm_params"), key="cooldown_time"
            )
            if deployment_dict is not None
            else None
        )
        exception_headers: Final = litellm.litellm_core_utils.exception_mapping_utils._get_response_headers(
            original_exception=exception
        )
        _get_retry_after: Final = (
            litellm.utils._get_retry_after_from_exception_header  # pyright: ignore[reportPrivateUsage] - as router.py
        )
        header_cooldown: Final = (
            _get_retry_after(response_headers=exception_headers) if exception_headers is not None else None
        )
        time_to_cooldown: Final = (
            deployment_cooldown
            if deployment_cooldown is not None and deployment_cooldown >= 0
            else (
                header_cooldown
                if header_cooldown is not None and header_cooldown >= 0
                else litellm_router.cooldown_time
            )
        )

        increment_deployment_failures_for_current_minute(
            litellm_router_instance=litellm_router,
            deployment_id=deployment_id,
        )
        _set_cooldown_deployments(
            litellm_router_instance=litellm_router,
            exception_status=exception_status,
            original_exception=exception,
            deployment=deployment_id,
            time_to_cooldown=time_to_cooldown,
        )

        verbose_router_logger.debug("Triggered cooldown for fallback deployment %s", deployment_id)
    except Exception as e:  # noqa: BLE001 - best-effort cooldown trigger must never break the fallback response itself
        verbose_router_logger.debug("Error triggering cooldown for fallback deployment: %s", e)


def fallback_attempt_key(fallback_target: object) -> str | None:
    """
    Identity of one fallback attempt, so the same attempt is never made twice per request.

    A bare model group name and a `{"model": name}` entry describe the same attempt. An
    entry carrying anything else describes a different one and keeps its own identity: a
    client-side fallback list overrides request params such as `messages`, and the router
    re-targets the group that just failed by attaching `_target_order` or
    `_excluded_deployment_ids` to select a different set of deployments inside it. The
    payload is hashed rather than kept, so a large `messages` override does not make the
    request hold a second copy of itself.

    Returns None for a shape with no usable identity, which is never skipped.
    """
    if isinstance(fallback_target, str):
        return fallback_target
    if not isinstance(fallback_target, dict):
        return None
    model: Final = fallback_target.get("model")
    if tuple(fallback_target) == ("model",) and isinstance(model, str):
        return model
    serialized: Final = json.dumps(fallback_target, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(slots=True)
class AttemptedFallbackTargets:
    """
    The fallback attempts a single request has already made.

    One instance is created on the first fallback hop and shared by reference for the rest
    of the walk, so an attempt made in one branch is not repeated in a sibling branch.
    Without it the walk enumerates paths rather than attempts: a fallback graph containing
    a cycle retries one deterministic failure once per path through the cycle, and a
    client-side fallback list is re-walked at every level of the recursion.
    """

    keys: frozenset[str] = frozenset()

    def __contains__(self, key: str) -> bool:
        return key in self.keys

    def record(self, key: str) -> None:
        self.keys = self.keys | frozenset((key,))


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


def get_fallback_model_group(fallbacks: list[Any], model_group: str) -> tuple[list[str] | None, int | None]:
    """
    Returns:
    - fallback_model_group: List[str] of fallback model groups. example: ["gpt-4", "gpt-3.5-turbo"]
    - generic_fallback_idx: int of the index of the generic fallback in the fallbacks list.

    Checks:
    - exact match
    - stripped model group match
    - generic fallback
    """
    generic_fallback_idx: int | None = None
    stripped_model_fallback: list[str] | None = None
    fallback_model_group: list[str] | None = None
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
            fallback_model_group = [item]
    ## if none, check for generic fallback
    if fallback_model_group is None:
        if stripped_model_fallback is not None:
            fallback_model_group = stripped_model_fallback
        elif generic_fallback_idx is not None:
            fallback_model_group = fallbacks[generic_fallback_idx]["*"]

    return fallback_model_group, generic_fallback_idx


PROVIDER_SCOPED_RESOURCE_KEYS: Final = ("input_file_id", "training_file")
PROVIDER_SCOPED_CREATION_FUNCTION_NAMES: Final = frozenset({"_acreate_file"})


def _get_fallback_target_model_group(fallback_entry: str | Mapping[str, object]) -> str | None:
    if isinstance(fallback_entry, str):
        return fallback_entry
    target: Final = fallback_entry.get("model")
    return target if isinstance(target, str) else None


def references_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    """
    True when the request names a file that only exists under one provider's credentials.

    Batch and fine-tuning jobs are created from a file the caller already uploaded, and
    that file lives in the account of the deployment that stored it. Handing the id to a
    different model group can only fail, and the second provider's error replaces the
    error the caller actually needs to see.
    """
    return any(kwargs.get(key) for key in PROVIDER_SCOPED_RESOURCE_KEYS)


def creates_provider_scoped_resource(kwargs: Mapping[str, object]) -> bool:
    """
    True when the request creates a resource that will live under one provider's credentials.

    A file uploaded for batches or fine-tuning is stored in the account of the deployment
    that handled it, and its id is only usable against the model group the caller named.
    Letting the upload fall back to a different model group silently stores the file with
    the wrong provider, and every later use of the returned id fails.
    """
    return getattr(kwargs.get("original_function"), "__name__", None) in PROVIDER_SCOPED_CREATION_FUNCTION_NAMES


async def run_async_fallback(
    *args: tuple[Any],
    litellm_router: LitellmRouter,
    fallback_model_group: list[str],
    original_model_group: str,
    original_exception: Exception,
    max_fallbacks: int,
    fallback_depth: int,
    include_fallback_errors: bool = False,
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
        **kwargs: Keyword arguments. `attempted_targets` carries the fallback attempts
            already made for this request, created on the first hop and shared by reference
            for the rest of the walk. A target already in it is skipped, so neither a
            fallback graph that loops back on itself nor a client-side fallback list
            re-walked at each level can repeat an attempt that has already failed. Identity
            comes from `fallback_attempt_key`, so an entry that overrides request params or
            re-targets the failed group with a different deployment selection stays distinct
            from a bare name.

    Returns:
        The response from the successful fallback model group.
    Raises:
        The most recent exception if all fallback model groups fail.
    """

    ### BASE CASE ### MAX FALLBACK DEPTH REACHED
    if fallback_depth >= max_fallbacks:
        raise original_exception

    error_from_fallbacks = original_exception
    fallback_errors = (get_fallback_error_info(original_exception),)
    metadata_variable_name: Final = _get_router_metadata_variable_name(
        function_name=getattr(kwargs.get("original_function"), "__name__", None)
    )
    same_model_group_only: Final = references_provider_scoped_resource(kwargs) or creates_provider_scoped_resource(
        kwargs
    )
    # Read out of kwargs and narrowed here rather than declared as a parameter: every caller
    # reaches this function by spreading a loosely-typed kwargs dict, so a declared parameter
    # would carry an annotation that no call site can actually be checked against.
    carried_targets: Final = kwargs.get("attempted_targets")
    attempted: Final = (
        carried_targets if isinstance(carried_targets, AttemptedFallbackTargets) else AttemptedFallbackTargets()
    )
    attempted.record(original_model_group)

    for mg in fallback_model_group:
        if mg == original_model_group:
            continue
        if same_model_group_only and _get_fallback_target_model_group(mg) != original_model_group:
            verbose_router_logger.info(
                "Skipping fallback to model_group = %s: request is pinned to model_group = %s by its uploaded file",
                mask_sensitive_structure(mg),
                original_model_group,
            )
            continue
        attempt_key = fallback_attempt_key(mg)
        if attempt_key is not None:
            if attempt_key in attempted:
                verbose_router_logger.info(
                    "Skipping fallback to model_group = %s, already attempted for this request",
                    mask_sensitive_structure(mg),
                )
                continue
            attempted.record(attempt_key)
        try:
            # LOGGING
            kwargs = litellm_router.log_retry(kwargs=kwargs, e=original_exception)
            verbose_router_logger.info("Falling back to model_group = %s", mask_sensitive_structure(mg))
            if isinstance(mg, str):
                kwargs["model"] = mg
            elif isinstance(mg, dict):
                kwargs.update(mg)
            fallback_depth = fallback_depth + 1
            kwargs[metadata_variable_name] = {
                "original_model_group": original_model_group,
                **(kwargs.get(metadata_variable_name) or {}),
                "model_group": kwargs.get("model", None),
                "attempted_fallbacks": fallback_depth,
            }
            kwargs["fallback_depth"] = fallback_depth
            kwargs["max_fallbacks"] = max_fallbacks
            kwargs["attempted_targets"] = attempted
            if include_fallback_errors:
                kwargs["include_fallback_errors"] = include_fallback_errors
            response = await litellm_router.async_function_with_fallbacks(*args, **kwargs)
            verbose_router_logger.info("Successful fallback b/w models.")
            response = add_fallback_headers_to_response(
                response=response,
                attempted_fallbacks=fallback_depth,
                fallback_errors=(list(fallback_errors) if include_fallback_errors else None),
            )
            # callback for successfull_fallback_event():
            await log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            return response
        except Exception as e:
            error_from_fallbacks = e
            fallback_errors = fallback_errors + (get_fallback_error_info(e),)
            await log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
            logging_obj = kwargs.get("litellm_logging_obj")
            if logging_obj is not None and logging_obj.model_call_details.get("has_logged_async_failure", False):
                _trigger_cooldown_for_failed_deployment(
                    litellm_router=litellm_router,
                    kwargs=kwargs,
                    exception=e,
                )
    raise error_from_fallbacks


async def log_success_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    """
    Log a successful fallback event to all registered callbacks.

    Uses LoggingCallbackManager.get_custom_loggers_for_type() to get deduplicated
    CustomLogger instances from all callback lists.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    # Get deduplicated CustomLogger instances from all callback lists
    custom_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_success_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error("Error in log_success_fallback_event: %s", e)


async def log_failure_fallback_event(original_model_group: str, kwargs: dict, original_exception: Exception):
    """
    Log a failed fallback event to all registered callbacks.

    Uses LoggingCallbackManager.get_custom_loggers_for_type() to get deduplicated
    CustomLogger instances from all callback lists.

    Args:
        original_model_group (str): The original model group before fallback.
        kwargs (dict): kwargs for the request

    Note:
        Errors during logging are caught and reported but do not interrupt the process.
    """
    # Get deduplicated CustomLogger instances from all callback lists
    custom_loggers: Final = litellm.logging_callback_manager.get_custom_loggers_for_type(CustomLogger)

    for _callback_custom_logger in custom_loggers:
        try:
            await _callback_custom_logger.log_failure_fallback_event(
                original_model_group=original_model_group,
                kwargs=kwargs,
                original_exception=original_exception,
            )
        except Exception as e:
            verbose_router_logger.error("Error in log_failure_fallback_event: %s", e)


def _check_non_standard_fallback_format(fallbacks: list[Any] | None) -> bool:
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
        for item in fallbacks:
            for key in LiteLLMParamsTypedDict.__annotations__:
                if key in item:
                    # If the value is a list, it's likely a standard fallback model group mapping
                    # (e.g. {"model": ["backup"]}) rather than a parameter override.
                    if not isinstance(item[key], list):
                        return True

    return False


def run_non_standard_fallback_format(fallbacks: list[str] | list[dict[str, Any]], model_group: str):
    pass
