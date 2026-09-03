"""
Router cooldown handlers
- _set_cooldown_deployments: puts a deployment in the cooldown list
- get_cooldown_deployments: returns the list of deployments in the cooldown list
- async_get_cooldown_deployments: ASYNC: returns the list of deployments in the cooldown list

"""

import asyncio
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm._logging import verbose_router_logger
from litellm.constants import (
    DEFAULT_COOLDOWN_TIME_SECONDS,
    DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS,
    DEFAULT_FAILURE_THRESHOLD_PERCENT,
    SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD,
)
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback

from .router_callbacks.track_deployment_metrics import (
    get_deployment_failures_for_current_minute,
    get_deployment_successes_for_current_minute,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.router import Router as _Router

    LitellmRouter = _Router
    Span = _Span | Any
else:
    LitellmRouter = Any
    Span = Any

_ADVISOR_ORCHESTRATION_FAILURE_ATTR: Final = "_litellm_advisor_orchestration_failure"


def mark_advisor_orchestration_failure(exception: BaseException) -> None:
    """Tag an exception as originating from advisor orchestration rather than the
    health of the router-selected deployment.

    Advisor orchestration failures (an advisor sub-call that targets different
    provider/credentials, or the orchestration loop exceeding max_uses) are not
    caused by the selected deployment, so they must not be attributed to (and
    cool down) that otherwise-healthy deployment. The exception object is tagged
    rather than wrapped so its type is preserved and the router's retry/fallback
    classification and the client-facing error are unchanged.
    """
    setattr(exception, _ADVISOR_ORCHESTRATION_FAILURE_ATTR, True)


def is_advisor_orchestration_failure(exception: BaseException | None) -> bool:
    """Whether ``exception`` was tagged by ``mark_advisor_orchestration_failure``."""
    return bool(getattr(exception, _ADVISOR_ORCHESTRATION_FAILURE_ATTR, False))


_EXCEPTION_POLICY_FIELDS: Final[tuple[tuple[type, str], ...]] = (
    # ContentPolicyViolationError subclasses BadRequestError, so it must be checked first.
    (litellm.ContentPolicyViolationError, "ContentPolicyViolationErrorAllowedFails"),
    (litellm.BadRequestError, "BadRequestErrorAllowedFails"),
    (litellm.AuthenticationError, "AuthenticationErrorAllowedFails"),
    (litellm.Timeout, "TimeoutErrorAllowedFails"),
    (litellm.RateLimitError, "RateLimitErrorAllowedFails"),
    (litellm.InternalServerError, "InternalServerErrorAllowedFails"),
    (litellm.ServiceUnavailableError, "ServiceUnavailableErrorAllowedFails"),
    (litellm.BadGatewayError, "BadGatewayErrorAllowedFails"),
    (litellm.NotFoundError, "NotFoundErrorAllowedFails"),
)


def _first_present(*sources: Mapping[str, Any] | None, key: str) -> int | float | None:
    """Return *key* from the first source mapping where it's set, so callers can
    support a setting living in more than one deployment config location. Sources
    are checked in order from most to least specific to that setting."""
    for source in sources:
        if source is None:
            continue
        value = source.get(key)
        if value is not None:
            return value
    return None


def _get_deployment_cooldown_policy(
    litellm_router_instance: LitellmRouter,
    deployment: str,
) -> tuple[Mapping[str, int] | None, int | None]:
    """Return (allowed_fails_policy, allowed_fails) from deployment model_info, or (None, None).

    `model_info` is the only supported location for these two fields (unlike
    `cooldown_time`, they have no pre-existing `litellm_params` precedent): `litellm_params`
    gets copied wholesale into the actual provider call kwargs (see e.g.
    Router._image_generation's `data = deployment["litellm_params"].copy()`), so a new
    field placed there would leak into the outgoing LLM request instead of staying
    router-internal.
    """
    dep: Final = litellm_router_instance.get_model_info(id=deployment)
    if dep is None:
        return None, None
    mi: Final[Mapping[str, Any]] = dep.get("model_info") or MappingProxyType({})
    raw: Final = mi.get("allowed_fails_policy")
    policy: Final[Mapping[str, int] | None] = raw if isinstance(raw, dict) else None
    allowed: Final[int | None] = mi.get("allowed_fails")
    return policy, allowed


def _resolve_allowed_fails_from_policy(
    policy: Mapping[str, int] | None,
    exception: Exception,
) -> int | None:
    """Match *exception* against *policy* and return the configured allowed-fail count, or None."""
    if policy is None:
        return None
    for exc_type, field in _EXCEPTION_POLICY_FIELDS:
        if isinstance(exception, exc_type):
            value = policy.get(field)
            if value is not None:
                return value
    return None


def _should_cooldown_based_on_deployment_policy(
    litellm_router_instance: LitellmRouter,
    deployment: str,
    original_exception: Exception,
    dep_policy: Mapping[str, int] | None,
    dep_allowed_fails: int | None,
    is_single_deployment_model_group: bool,
) -> bool:
    """Resolve deployment-level allowed-fails and delegate to the shared counting logic.

    When the deployment's policy doesn't cover *original_exception*'s type and no
    deployment-wide `allowed_fails` is set either, defer to router-level behavior
    instead of forcing an immediate cooldown.

    A generic, deployment-wide `allowed_fails` predates this feature's per-exception-type
    policy and is a much less deliberate opt-in, so on a single-deployment model group it
    still defers to the "avoid cooldowns on single deployment model groups" safety net
    (see `_should_cooldown_deployment`'s BASE CASE) rather than silently disabling it. An
    explicit, named-exception-type `allowed_fails_policy` entry is unambiguous enough to
    override that safety net, matching `_has_explicit_allowed_fails_policy_for_exception`.
    """
    allowed_fails_from_policy: Final = _resolve_allowed_fails_from_policy(dep_policy, original_exception)
    if allowed_fails_from_policy is None and dep_allowed_fails is not None and is_single_deployment_model_group:
        return False

    allowed_fails_override: Final[int | None] = (
        allowed_fails_from_policy if allowed_fails_from_policy is not None else dep_allowed_fails
    )
    cache_key_suffix: Final[str | None] = (
        type(original_exception).__name__
        if allowed_fails_from_policy is not None
        else ("generic" if dep_allowed_fails is not None else None)
    )

    dep: Final = litellm_router_instance.get_model_info(id=deployment)
    cooldown_time_override: Final = (
        _first_present(dep.get("model_info"), dep.get("litellm_params"), key="cooldown_time")
        if dep is not None
        else None
    )

    return should_cooldown_based_on_allowed_fails_policy(
        litellm_router_instance=litellm_router_instance,
        deployment=deployment,
        original_exception=original_exception,
        allowed_fails_override=allowed_fails_override,
        cooldown_time_override=cooldown_time_override,
        cache_key_suffix=cache_key_suffix,
    )


def _has_explicit_allowed_fails_policy_for_exception(
    litellm_router_instance: LitellmRouter,
    deployment: str | None,
    original_exception: Exception,
) -> bool:
    """True if this deployment has an explicit, deployment-level allowed_fails_policy
    entry matching *original_exception*'s type.

    `_is_cooldown_required` skips cooldown evaluation for most 4XX errors (BadRequestError,
    ContentPolicyViolationError) by default, since a generic client error is usually not the
    deployment's fault. A deployment-level allowed_fails_policy entry naming that exact
    exception type is this PR's own per-deployment opt-in, so it overrides that default.

    Deliberately scoped to the deployment level only, and to the named-exception-type
    policy dict rather than a plain `allowed_fails` integer: a pre-existing router-wide
    `allowed_fails_policy` (or a deployment's generic `allowed_fails`) predates this
    feature and must keep its existing behavior for 4XX types `_is_cooldown_required`
    already excludes, rather than silently start cooling down deployments whose configs
    never opted into this specific override.
    """
    if deployment is None:
        return False
    dep_policy, _ = _get_deployment_cooldown_policy(litellm_router_instance, deployment)
    return _resolve_allowed_fails_from_policy(dep_policy, original_exception) is not None


def _is_cooldown_required(
    litellm_router_instance: LitellmRouter,
    model_id: str,
    exception_status: str | int,
    exception_str: str | None = None,
) -> bool:
    """
    A function to determine if a cooldown is required based on the exception status.

    Parameters:
        model_id (str) The id of the model in the model list
        exception_status (Union[str, int]): The status of the exception.

    Returns:
        bool: True if a cooldown is required, False otherwise.
    """
    try:
        ignored_strings: Final = ["APIConnectionError"]
        if exception_str is not None:  # don't cooldown on litellm api connection errors errors
            for ignored_string in ignored_strings:
                if ignored_string in exception_str:
                    return False

        if isinstance(exception_status, str):
            if len(exception_status) == 0:
                return False
            exception_status = int(exception_status)

        if exception_status >= 400 and exception_status < 500:
            if exception_status == 429:
                # Cool down 429 Rate Limit Errors
                return True

            elif exception_status == 401:
                # Cool down 401 Auth Errors
                return True

            elif exception_status == 408 or exception_status == 404:
                return True

            else:
                # Do NOT cool down all other 4XX Errors
                return False

        else:
            # should cool down for all other errors
            return True

    except Exception:
        # Catch all - if any exceptions default to cooling down
        return True


def _should_run_cooldown_logic(
    litellm_router_instance: LitellmRouter,
    deployment: str | None,
    exception_status: str | int,
    original_exception: Any,
    time_to_cooldown: float | None = None,
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
    if deployment is None or litellm_router_instance.get_model_group(id=deployment) is None:
        verbose_router_logger.debug(
            "Should Not Run Cooldown Logic: deployment id is none or model group can't be found."
        )
        return False

    #########################################################
    # If time_to_cooldown is 0 or 0.0000000, don't run cooldown logic
    #########################################################
    if time_to_cooldown is not None and math.isclose(a=time_to_cooldown, b=0.0, abs_tol=1e-9):
        verbose_router_logger.debug("Should Not Run Cooldown Logic: time_to_cooldown is effectively 0")
        return False

    if litellm_router_instance.disable_cooldowns:
        verbose_router_logger.debug("Should Not Run Cooldown Logic: disable_cooldowns is True")
        return False

    if deployment is None:
        verbose_router_logger.debug("Should Not Run Cooldown Logic: deployment is None")
        return False

    if not _is_cooldown_required(
        litellm_router_instance=litellm_router_instance,
        model_id=deployment,
        exception_status=exception_status,
        exception_str=str(original_exception),
    ) and not _has_explicit_allowed_fails_policy_for_exception(
        litellm_router_instance=litellm_router_instance,
        deployment=deployment,
        original_exception=original_exception,
    ):
        verbose_router_logger.debug("Should Not Run Cooldown Logic: _is_cooldown_required returned False")
        return False

    if deployment in litellm_router_instance.provider_default_deployment_ids:
        verbose_router_logger.debug("Should Not Run Cooldown Logic: deployment is in provider_default_deployment_ids")
        return False

    return True


def _should_cooldown_deployment(
    litellm_router_instance: LitellmRouter,
    deployment: str,
    exception_status: str | int,
    original_exception: Any,
    requested_model_group: str | None = None,
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
    model_group: Final = litellm_router_instance.get_model_group(id=deployment)
    is_single_deployment_model_group = False
    if model_group is not None and len(model_group) == 1:
        is_single_deployment_model_group = not litellm_router_instance.routing_group_has_alternatives(
            requested_model_group
        )

    ## CHECK DEPLOYMENT-LEVEL POLICY FIRST (overrides router-level)
    dep_policy, dep_allowed_fails = _get_deployment_cooldown_policy(litellm_router_instance, deployment)
    if dep_policy is not None or dep_allowed_fails is not None:
        return _should_cooldown_based_on_deployment_policy(
            litellm_router_instance,
            deployment,
            original_exception,
            dep_policy,
            dep_allowed_fails,
            is_single_deployment_model_group,
        )

    ## BASE CASE - single deployment
    if (
        litellm_router_instance.allowed_fails_policy is None
        and _is_allowed_fails_set_on_router(litellm_router_instance=litellm_router_instance) is False
    ):
        num_successes_this_minute: Final = get_deployment_successes_for_current_minute(
            litellm_router_instance=litellm_router_instance, deployment_id=deployment
        )
        num_fails_this_minute: Final = get_deployment_failures_for_current_minute(
            litellm_router_instance=litellm_router_instance, deployment_id=deployment
        )

        total_requests_this_minute: Final = num_successes_this_minute + num_fails_this_minute
        percent_fails = 0.0
        if total_requests_this_minute > 0:
            percent_fails = num_fails_this_minute / (num_successes_this_minute + num_fails_this_minute)
        verbose_router_logger.debug(
            "percent fails for deployment = %s, percent fails = %s, num successes = %s, num fails = %s",
            deployment,
            percent_fails,
            num_successes_this_minute,
            num_fails_this_minute,
        )

        exception_status_int: Final = cast_exception_status_to_int(exception_status)
        if exception_status_int == 429 and not is_single_deployment_model_group:
            return True
        elif percent_fails == 1.0 and total_requests_this_minute >= SINGLE_DEPLOYMENT_TRAFFIC_FAILURE_THRESHOLD:
            # Cooldown if all requests failed and we have reasonable traffic
            return True
        elif (
            percent_fails > DEFAULT_FAILURE_THRESHOLD_PERCENT
            and total_requests_this_minute >= DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS
            and not is_single_deployment_model_group  # by default we should avoid cooldowns on single deployment model groups
        ):
            # Only apply error rate cooldown when we have enough requests to make the percentage meaningful
            return True

        elif litellm._should_retry(status_code=cast_exception_status_to_int(exception_status)) is False:
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
    exception_status: str | int,
    deployment: str | None = None,
    time_to_cooldown: float | None = None,
    requested_model_group: str | None = None,
) -> bool:
    """
    Add a model to the list of models being cooled down for that minute, if it exceeds the allowed fails / minute

    or

    the exception is not one that should be immediately retried (e.g. 401)

    Returns:
    - True if the deployment should be put in cooldown
    - False if the deployment should not be put in cooldown
    """
    verbose_router_logger.debug("checks 'should_run_cooldown_logic'")

    if (
        _should_run_cooldown_logic(
            litellm_router_instance=litellm_router_instance,
            deployment=deployment,
            exception_status=exception_status,
            original_exception=original_exception,
            time_to_cooldown=time_to_cooldown,
        )
        is False
        or deployment is None
    ):
        verbose_router_logger.debug("should_run_cooldown_logic returned False")
        return False

    exception_status_int: Final = cast_exception_status_to_int(exception_status)
    verbose_router_logger.debug("Attempting to add %s to cooldown list", deployment)

    if _should_cooldown_deployment(
        litellm_router_instance=litellm_router_instance,
        deployment=deployment,
        exception_status=exception_status,
        original_exception=original_exception,
        requested_model_group=requested_model_group,
    ):
        litellm_router_instance.cooldown_cache.add_deployment_to_cooldown(
            model_id=deployment,
            original_exception=original_exception,
            exception_status=exception_status_int,
            cooldown_time=time_to_cooldown,
        )

        # Trigger cooldown callback handler
        asyncio.create_task(
            router_cooldown_event_callback(
                litellm_router_instance=litellm_router_instance,
                deployment_id=deployment,
                exception_status=exception_status,
                cooldown_time=time_to_cooldown,
            )
        )
        return True
    return False


async def _async_get_cooldown_deployments(
    litellm_router_instance: LitellmRouter,
    parent_otel_span: Span | None,
) -> list[str]:
    """
    Async implementation of '_get_cooldown_deployments'
    """
    model_ids: Final = litellm_router_instance.get_model_ids()
    cooldown_models: Final = await litellm_router_instance.cooldown_cache.async_get_active_cooldowns(
        model_ids=model_ids,
        parent_otel_span=parent_otel_span,
    )

    cached_value_deployment_ids = []
    if (
        cooldown_models is not None
        and isinstance(cooldown_models, list)
        and len(cooldown_models) > 0
        and isinstance(cooldown_models[0], tuple)
    ):
        cached_value_deployment_ids = [cv[0] for cv in cooldown_models]

    verbose_router_logger.debug("retrieve cooldown models: %s", cooldown_models)
    return cached_value_deployment_ids


async def _async_get_cooldown_deployments_with_debug_info(
    litellm_router_instance: LitellmRouter,
    parent_otel_span: Span | None,
) -> list[tuple]:
    """
    Async implementation of '_get_cooldown_deployments'
    """
    model_ids: Final = litellm_router_instance.get_model_ids()
    cooldown_models: Final = await litellm_router_instance.cooldown_cache.async_get_active_cooldowns(
        model_ids=model_ids, parent_otel_span=parent_otel_span
    )

    verbose_router_logger.debug("retrieve cooldown models: %s", cooldown_models)
    return cooldown_models


def _get_cooldown_deployments(litellm_router_instance: LitellmRouter, parent_otel_span: Span | None) -> list[str]:
    """
    Get the list of models being cooled down for this minute
    """
    # get the current cooldown list for that minute

    # ----------------------
    # Return cooldown models
    # ----------------------
    model_ids: Final = litellm_router_instance.get_model_ids()

    cooldown_models: Final = litellm_router_instance.cooldown_cache.get_active_cooldowns(
        model_ids=model_ids, parent_otel_span=parent_otel_span
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
    allowed_fails_override: int | None = None,
    cooldown_time_override: float | None = None,
    cache_key_suffix: str | None = None,
) -> bool:
    """
    Check if fails are within the allowed limit and update the number of fails.

    When *allowed_fails_override* / *cooldown_time_override* are supplied they
    take precedence over the router-level values (used by deployment-level overrides).

    When *cache_key_suffix* is supplied the fail counter is keyed as
    ``{deployment}:{cache_key_suffix}`` so that different exception types are
    tracked independently per deployment.

    Returns:
    - True if fails exceed the allowed limit (should cooldown)
    - False if fails are within the allowed limit (should not cooldown)
    """
    allowed_fails_from_policy: Final = litellm_router_instance.get_allowed_fails_from_policy(
        exception=original_exception
    )
    allowed_fails: Final = (
        allowed_fails_override
        if allowed_fails_override is not None
        else (
            allowed_fails_from_policy
            if allowed_fails_from_policy is not None
            else litellm_router_instance.allowed_fails
        )
    )
    cooldown_time: Final = (
        cooldown_time_override
        if cooldown_time_override is not None
        else (litellm_router_instance.cooldown_time or DEFAULT_COOLDOWN_TIME_SECONDS)
    )

    cache_key: Final = f"{deployment}:{cache_key_suffix}" if cache_key_suffix else deployment
    current_fails: Final = litellm_router_instance.failed_calls.get_cache(key=cache_key) or 0
    updated_fails: Final = current_fails + 1

    if updated_fails > allowed_fails:
        return True
    else:
        litellm_router_instance.failed_calls.set_cache(key=cache_key, value=updated_fails, ttl=cooldown_time)

    return False


def _is_allowed_fails_set_on_router(
    litellm_router_instance: LitellmRouter,
) -> bool:
    """
    Check if Router.allowed_fails is set or is Non-default Value

    Returns:
    - True if Router.allowed_fails is set or is Non-default Value
    - False if Router.allowed_fails is None or is Default Value
    """
    if litellm_router_instance.allowed_fails is None:
        return False
    if litellm_router_instance.allowed_fails != litellm.allowed_fails:
        return True
    return False


def cast_exception_status_to_int(exception_status: str | int) -> int:
    if isinstance(exception_status, str):
        try:
            exception_status = int(exception_status)
        except Exception:
            verbose_router_logger.debug(
                "Unable to cast exception status to int %s. Defaulting to status=500.", exception_status
            )
            exception_status = 500
    return exception_status
