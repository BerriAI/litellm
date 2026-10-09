"""Checks a proxy-made sub-call runs against the caller of the parent request.

The compaction summary call and the safeguards classifier call go to a model the
admin configured, not the one the client asked for, and they never pass back
through the proxy's auth, budget, or rate-limit hooks. These checks apply the
caller's model allowlists, per-model budgets, and RPM/TPM limits to that model
before the sub-call runs.
"""

from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, Optional, Protocol

import litellm
from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        RateLimitDescriptor,
        RateLimitDescriptorRateLimitObject,
        RateLimitResponse,
    )
    from litellm.router import Router


class _CreateRateLimitDescriptors(Protocol):
    def __call__(
        self,
        *,
        user_api_key_dict: "UserAPIKeyAuth",
        data: Mapping[str, str],
        rpm_limit_type: object,
        tpm_limit_type: object,
        model_has_failures: bool,
    ) -> "Sequence[RateLimitDescriptor]": ...


class _AddModelRateLimitDescriptor(Protocol):
    def __call__(
        self,
        *,
        user_api_key_dict: "UserAPIKeyAuth",
        requested_model: str,
        descriptors: "Sequence[RateLimitDescriptor]",
    ) -> None: ...


class _CreateOrgRateLimitDescriptors(Protocol):
    def __call__(
        self, user_api_key_dict: "UserAPIKeyAuth", requested_model: str | None = None
    ) -> "Sequence[RateLimitDescriptor]": ...


class _GetProxyHook(Protocol):
    def __call__(self, hook: str) -> object: ...


class _ShouldRateLimit(Protocol):
    def __call__(
        self,
        *,
        descriptors: "Sequence[RateLimitDescriptor]",
        parent_otel_span: object,
        read_only: bool,
    ) -> "Awaitable[RateLimitResponse]": ...


async def caller_may_use_model(
    user_api_key_auth: Optional["UserAPIKeyAuth"],
    model: str,
    llm_router: Optional["Router"],
) -> bool:
    """True when the caller passes the access, budget, and rate-limit checks for ``model``."""
    return (
        await caller_can_call_model(user_api_key_auth=user_api_key_auth, model=model, llm_router=llm_router)
        and await caller_within_model_budget(user_api_key_auth=user_api_key_auth, model=model)
        and await caller_within_model_rate_limit(user_api_key_auth=user_api_key_auth, model=model)
    )


async def caller_can_call_model(
    user_api_key_auth: Optional["UserAPIKeyAuth"],
    model: str,
    llm_router: Optional["Router"],
) -> bool:
    """Return True when every model-allowlist scope on the parent request is
    satisfied for ``model``.

    Mirrors the model-scope enforcement that ``litellm.proxy.auth.common_checks``
    runs for the client-requested model: key, team, user (personal), project, and
    team-member allowlists.

    Returns True (allow) when ``user_api_key_auth`` is not present — SDK
    callers and tests run outside the proxy, where no key/team policy exists.
    Returns False when any of the active allowlists denies the model
    (``ProxyException`` from ``_can_object_call_model`` / ``can_*_model``).
    Unexpected errors during an access check fail closed but are logged
    separately so operators can distinguish them from a real access-denied
    response. User and project lookup failures (object missing from cache or
    DB) skip the corresponding scope — matching ``common_checks``, which only
    enforces a scope when its backing object can be loaded. A failed team
    membership read (a database outage) fails closed instead, since a member
    whose limits cannot be read must not have the model invoked with those
    limits dropped.
    """
    if user_api_key_auth is None:
        return True
    try:
        from litellm.proxy._types import ProxyException
        from litellm.proxy.auth.auth_checks import (
            can_object_call_model,
            can_project_access_model,
            can_user_call_model,
            get_project_object,
            get_team_membership,
            get_user_object,
        )
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )
    except Exception:
        return True

    key_models: Final = list(getattr(user_api_key_auth, "models", None) or [])
    team_id: Final[str | None] = getattr(user_api_key_auth, "team_id", None)
    team_model_aliases: Final[dict[str, str] | None] = getattr(user_api_key_auth, "team_model_aliases", None)
    team_models: Final = list(getattr(user_api_key_auth, "team_models", None) or [])
    user_id: Final[str | None] = getattr(user_api_key_auth, "user_id", None)
    project_id: Final[str | None] = getattr(user_api_key_auth, "project_id", None)

    checks: Final[tuple[tuple[Literal["key", "team"], list[str]], ...]] = (
        ("key", key_models),
        ("team", team_models),
    )
    for object_type, models in checks:
        if not models:
            continue
        try:
            can_object_call_model(
                model=model,
                llm_router=llm_router,
                models=models,
                team_model_aliases=team_model_aliases,
                team_id=team_id,
                object_type=object_type,
            )
        except ProxyException:
            return False
        except Exception as e:
            verbose_logger.warning(
                "proxy sub-call: unexpected error during %s-level access check for model=%s; denying access: %s",
                object_type,
                model,
                e,
            )
            return False

    if user_id is not None and prisma_client is not None:
        try:
            user_obj = await get_user_object(
                user_id=user_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                user_id_upsert=False,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_logger.debug(
                "proxy sub-call: user object lookup failed for model=%s access check; skipping user-level scope: %s",
                model,
                e,
            )
            user_obj = None
        if user_obj is not None:
            try:
                await can_user_call_model(
                    model=model,
                    llm_router=llm_router,
                    user_object=user_obj,
                )
            except ProxyException:
                return False
            except Exception as e:
                verbose_logger.warning(
                    "proxy sub-call: unexpected error during user-level access check for model=%s; denying access: %s",
                    model,
                    e,
                )
                return False

    if project_id is not None and prisma_client is not None:
        try:
            project_obj = await get_project_object(
                project_id=project_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_logger.debug(
                "proxy sub-call: project object lookup failed for model=%s access check; "
                "skipping project-level scope: %s",
                model,
                e,
            )
            project_obj = None
        if project_obj is not None and project_obj.models:
            try:
                can_project_access_model(
                    model=model,
                    project_object=project_obj,
                    llm_router=llm_router,
                )
            except ProxyException:
                return False
            except Exception as e:
                verbose_logger.warning(
                    "proxy sub-call: unexpected error during project-level access check for model=%s; "
                    "denying access: %s",
                    model,
                    e,
                )
                return False

    if user_id is not None and team_id is not None and prisma_client is not None:
        try:
            team_membership = await get_team_membership(
                user_id=user_id,
                team_id=team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                proxy_logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            verbose_logger.warning(
                "proxy sub-call: team membership lookup failed for model=%s access check; denying access: %s",
                model,
                e,
            )
            return False
        member_allowed_models: Final = (
            team_membership.litellm_budget_table.allowed_models
            if team_membership is not None and team_membership.litellm_budget_table is not None
            else None
        )
        if member_allowed_models:
            try:
                can_object_call_model(
                    model=model,
                    llm_router=llm_router,
                    models=list(member_allowed_models),
                    team_model_aliases=team_model_aliases,
                    team_id=team_id,
                    object_type="team",
                )
            except ProxyException:
                return False
            except Exception as e:
                verbose_logger.warning(
                    "proxy sub-call: unexpected error during member-level access check for model=%s; "
                    "denying access: %s",
                    model,
                    e,
                )
                return False

    return True


async def caller_within_model_budget(
    user_api_key_auth: Optional["UserAPIKeyAuth"],
    model: str,
) -> bool:
    """Return True when the caller is within their per-model budget for ``model``.

    Mirrors the per-model budget enforcement that ``user_api_key_auth`` runs for
    the client-requested model. Returns True outside the proxy or when no
    per-model budget is configured.

    Every scope is checked because the sub-call's spend is charged to every
    scope: the key, team, user and end-user budgets ride into the sub-call's
    metadata, so skipping one of them would let the sub-call increment a
    counter it can never be refused by.
    """
    if user_api_key_auth is None:
        return True
    try:
        from litellm.proxy.proxy_server import model_max_budget_limiter
    except Exception:
        return True

    model_max_budget: Final = getattr(user_api_key_auth, "model_max_budget", None)
    token: Final = getattr(user_api_key_auth, "token", None)
    if isinstance(model_max_budget, dict) and model_max_budget and token is not None:
        try:
            await model_max_budget_limiter.is_key_within_model_budget(
                user_api_key_dict=user_api_key_auth,
                model=model,
            )
        except litellm.BudgetExceededError:
            return False
        except Exception as e:
            verbose_logger.warning(
                "proxy sub-call: unexpected error during key model-budget check for model=%s; denying: %s",
                model,
                e,
            )
            return False

    user_model_max_budget: Final = user_api_key_auth.user_model_max_budget
    user_id: Final = user_api_key_auth.user_id
    if isinstance(user_model_max_budget, dict) and user_model_max_budget and user_id is not None:
        try:
            await model_max_budget_limiter.is_user_within_model_budget(
                user_id=user_id,
                user_model_max_budget=user_model_max_budget,
                model=model,
            )
        except litellm.BudgetExceededError:
            return False
        except Exception as e:  # noqa: BLE001  # a budget gate denies on any failure, as the key and end-user scopes do
            verbose_logger.warning(
                "proxy sub-call: unexpected error during user model-budget check for model=%s; denying: %s",
                model,
                e,
            )
            return False

    team_model_max_budget: Final = user_api_key_auth.team_model_max_budget
    team_id: Final = user_api_key_auth.team_id
    if isinstance(team_model_max_budget, dict) and team_model_max_budget and team_id is not None:
        try:
            await model_max_budget_limiter.is_team_within_model_budget(
                team_id=team_id,
                team_model_max_budget=team_model_max_budget,
                key_model_max_budget=model_max_budget if isinstance(model_max_budget, dict) else None,
                model=model,
            )
        except litellm.BudgetExceededError:
            return False
        except Exception as e:  # noqa: BLE001  # a budget gate denies on any failure, as the other scopes do
            verbose_logger.warning(
                "proxy sub-call: unexpected error during team model-budget check for model=%s; denying: %s",
                model,
                e,
            )
            return False

    end_user_model_max_budget: Final[dict[str, object] | None] = getattr(
        user_api_key_auth, "end_user_model_max_budget", None
    )
    end_user_id: Final[str | None] = getattr(user_api_key_auth, "end_user_id", None)
    if isinstance(end_user_model_max_budget, dict) and end_user_model_max_budget and end_user_id is not None:
        try:
            await model_max_budget_limiter.is_end_user_within_model_budget(
                end_user_id=end_user_id,
                end_user_model_max_budget=end_user_model_max_budget,
                model=model,
            )
        except litellm.BudgetExceededError:
            return False
        except Exception as e:
            verbose_logger.warning(
                "proxy sub-call: unexpected error during end-user model-budget check for model=%s; denying: %s",
                model,
                e,
            )
            return False

    return True


def _without_parallel_request_gauges(
    descriptors: "Sequence[RateLimitDescriptor]",
) -> "tuple[RateLimitDescriptor, ...]":
    return tuple(_without_parallel_request_gauge(descriptor) for descriptor in descriptors)


def _without_parallel_request_gauge(descriptor: "RateLimitDescriptor") -> "RateLimitDescriptor":
    rate_limit: Final = descriptor.get("rate_limit")
    if rate_limit is None or rate_limit.get("max_parallel_requests") is None:
        return descriptor
    windowed_limits: Final[RateLimitDescriptorRateLimitObject] = {
        "requests_per_unit": rate_limit.get("requests_per_unit"),
        "tokens_per_unit": rate_limit.get("tokens_per_unit"),
        "window_size": rate_limit.get("window_size"),
    }
    return {**descriptor, "rate_limit": windowed_limits}


async def caller_within_model_rate_limit(
    user_api_key_auth: Optional["UserAPIKeyAuth"],
    model: str,
) -> bool:
    """Return True when the caller is within their configured RPM/TPM limits for ``model``.

    This mirrors the read side of
    ``_PROXY_MaxParallelRequestsHandler_v3.async_pre_call_hook`` for ``model``:
    it builds the same descriptor set and runs the check in ``read_only`` mode
    so no counter is reserved or incremented — the sub-call's actual usage is
    still charged exactly once by the limiter's post-call success hook (via the
    propagated ``litellm_metadata``). ``max_parallel_requests`` gauges are left
    out of the check: the sub-call runs inside the caller's already admitted
    request, whose own slot would otherwise count against it.

    Returns True (allow) outside the proxy, when the active limiter does not
    expose the read-only descriptor check (legacy limiter), or when the
    descriptor set cannot be built — the deny signals are a definitive
    ``OVER_LIMIT`` response and the limiter's own fail-closed rejection
    (``RateLimitUnverifiableError``, raised when ``fail_closed_rate_limit_enforcement``
    is on and the counters could not be verified), so any other internal error
    here lets the sub-call run rather than blocking every one.
    """
    if user_api_key_auth is None:
        return True
    try:
        from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitUnverifiableError
        from litellm.proxy.proxy_server import proxy_logging_obj
    except Exception:
        return True

    get_proxy_hook: Final[_GetProxyHook | None] = getattr(proxy_logging_obj, "get_proxy_hook", None)
    limiter: Final[object] = get_proxy_hook("parallel_request_limiter") if get_proxy_hook is not None else None
    should_rate_limit_check: Final[_ShouldRateLimit | None] = getattr(limiter, "should_rate_limit", None)
    create_descriptors: Final[_CreateRateLimitDescriptors | None] = getattr(
        limiter, "create_rate_limit_descriptors", None
    )
    add_team_descriptor: Final[_AddModelRateLimitDescriptor | None] = getattr(
        limiter, "_add_team_model_rate_limit_descriptor_from_metadata", None
    )
    add_project_descriptor: Final[_AddModelRateLimitDescriptor | None] = getattr(
        limiter, "_add_project_model_rate_limit_descriptor_from_metadata", None
    )
    create_org_descriptors: Final[_CreateOrgRateLimitDescriptors | None] = getattr(
        limiter, "create_organization_rate_limit_descriptor", None
    )
    if (
        limiter is None
        or should_rate_limit_check is None
        or create_descriptors is None
        or add_team_descriptor is None
        or add_project_descriptor is None
        or create_org_descriptors is None
    ):
        return True

    try:
        metadata: Final[Mapping[str, object]] = getattr(user_api_key_auth, "metadata", None) or {}
        data: Final = {"model": model}
        base_descriptors: Final = create_descriptors(
            user_api_key_dict=user_api_key_auth,
            data=data,
            rpm_limit_type=metadata.get("rpm_limit_type"),
            tpm_limit_type=metadata.get("tpm_limit_type"),
            model_has_failures=False,
        )
        add_team_descriptor(
            user_api_key_dict=user_api_key_auth,
            requested_model=model,
            descriptors=base_descriptors,
        )
        add_project_descriptor(
            user_api_key_dict=user_api_key_auth,
            requested_model=model,
            descriptors=base_descriptors,
        )
        descriptors: Final = _without_parallel_request_gauges(
            (*base_descriptors, *create_org_descriptors(user_api_key_auth, model))
        )
        if not descriptors:
            return True
        parent_otel_span: Final[object] = getattr(user_api_key_auth, "parent_otel_span", None)
        response: Final[RateLimitResponse] = await should_rate_limit_check(
            descriptors=descriptors,
            parent_otel_span=parent_otel_span,
            read_only=True,
        )
    except RateLimitUnverifiableError as e:
        verbose_logger.warning(
            "proxy sub-call: rate-limit counters for model=%s could not be verified; denying: %s",
            model,
            e.detail,
        )
        return False
    except Exception as e:
        verbose_logger.warning(
            "proxy sub-call: unexpected error during rate-limit check for model=%s; allowing: %s",
            model,
            e,
        )
        return True
    return response.get("overall_code") != "OVER_LIMIT"
