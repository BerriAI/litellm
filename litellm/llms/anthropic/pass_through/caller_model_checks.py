"""Checks a proxy-made sub-call runs against the caller of the parent request.

The compaction summary call and the safeguards classifier call go to a model the
admin configured, not the one the client asked for, and they never pass back
through the proxy's auth, budget, or rate-limit hooks. These checks apply the
caller's model allowlists, per-model budgets, and RPM/TPM limits to that model
before the sub-call runs.
"""

from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Literal, Optional, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from starlette.requests import Request

    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        RateLimitDescriptor,
        RateLimitDescriptorRateLimitObject,
        RateLimitResponse,
    )
    from litellm.router import Router
    from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

CallerModelVerdict: TypeAlias = Literal["allowed", "denied", "denied_for_now"]
_DEFAULT_SUB_CALL_ROUTE: Final = "/v1/messages"
_MODEL_SCOPED_DESCRIPTOR_PREFIX: Final = "model_per_"
_HEADERS: Final = TypeAdapter(Mapping[str, str])


class _TeamMemberModelBudgetLimiter(Protocol):
    async def is_team_member_within_model_budget(
        self,
        user_id: str,
        team_id: str,
        team_member_model_max_budget: Mapping[str, object],
        model: str,
    ) -> bool: ...


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


async def caller_model_verdict(
    user_api_key_auth: Optional["UserAPIKeyAuth"],
    model: str,
    llm_router: Optional["Router"],
    parent_litellm_metadata: Mapping[str, object] | None = None,
    parent_trace_id: str | None = None,
) -> CallerModelVerdict:
    """``denied`` when an allowlist or a per-model budget refuses ``model`` for the caller, which only an
    admin or the next budget window changes; ``denied_for_now`` when a rate-limit window is full or a check
    could not run, which the caller's next request may clear."""
    access: Final = await caller_can_call_model(
        user_api_key_auth=user_api_key_auth,
        model=model,
        llm_router=llm_router,
        parent_litellm_metadata=parent_litellm_metadata,
        parent_trace_id=parent_trace_id,
    )
    if access != "allowed":
        return access
    budget: Final = await caller_within_model_budget(user_api_key_auth=user_api_key_auth, model=model)
    if budget != "allowed":
        return budget
    return await caller_within_model_rate_limit(user_api_key_auth=user_api_key_auth, model=model)


async def caller_can_call_model(
    user_api_key_auth: Optional["UserAPIKeyAuth"],
    model: str,
    llm_router: Optional["Router"],
    parent_litellm_metadata: Mapping[str, object] | None = None,
    parent_trace_id: str | None = None,
) -> CallerModelVerdict:
    """Run the proxy's own model-access checks for ``model`` against the caller of the parent request.

    These are the checks the parent request passed for its own model, so every scope they cover (key, team,
    team member, user, project, end user, managed agent grants) applies to the sub-call's model too, and a
    deployment that skips them for the parent (no-auth dev mode, custom auth without common checks) skips
    them here. Budgets are left to ``caller_within_model_budget`` so nothing is reserved. ``denied`` answers
    a refusal, ``denied_for_now`` a check that could not run, and ``allowed`` a caller outside the proxy.
    """
    if user_api_key_auth is None:
        return "allowed"
    try:
        from fastapi import HTTPException

        from litellm.proxy._types import ProxyException
        from litellm.proxy.auth.user_api_key_auth import (
            enforce_key_and_fallback_model_access,  # pyright: ignore[reportUnknownVariableType]  # takes the bare dict and list the proxy passes
            run_centralized_common_checks,
        )
        from litellm.proxy.proxy_server import (
            llm_model_list,  # pyright: ignore[reportUnknownVariableType]  # proxy global has untyped deployment entries
        )
    except ImportError:
        return "allowed"

    route: Final = user_api_key_auth.request_route or _DEFAULT_SUB_CALL_ROUTE
    request: Final = _sub_call_request(route, parent_litellm_metadata)
    request_data: Final = _sub_call_request_data(model, parent_litellm_metadata, parent_trace_id)
    caller: Final = user_api_key_auth.model_copy(update={"budget_reservation": None})
    try:
        await enforce_key_and_fallback_model_access(
            valid_token=caller,
            request_data=request_data,
            route=route,
            request=request,
            llm_model_list=llm_model_list,  # pyright: ignore[reportUnknownArgumentType]  # proxy global has untyped deployment entries
            llm_router=llm_router,
        )
        await run_centralized_common_checks(caller, request, request_data, route, check_budgets=False)
    except (ProxyException, HTTPException, litellm.BudgetExceededError):
        return "denied"
    except Exception as e:  # noqa: BLE001  # a check that cannot run must not invoke the model with the caller's limits dropped
        verbose_logger.warning(
            "proxy sub-call: model access check for model=%s could not run; denying for now: %s", model, e
        )
        return "denied_for_now"
    return "allowed"


def _sub_call_request(route: str, parent_litellm_metadata: Mapping[str, object] | None) -> "Request":
    from starlette.requests import Request

    headers: Final = _parent_headers(parent_litellm_metadata)
    return Request(
        scope={
            "type": "http",
            "method": "POST",
            "path": route,
            "scheme": "http",
            "query_string": b"",
            "root_path": "",
            "headers": tuple(
                (name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in headers.items()
            ),
        }
    )


def _parent_headers(parent_litellm_metadata: Mapping[str, object] | None) -> Mapping[str, str]:
    if not parent_litellm_metadata:
        return {}
    try:
        return _HEADERS.validate_python(parent_litellm_metadata.get("headers") or {})
    except ValidationError:
        return {}


def _sub_call_request_data(
    model: str, parent_litellm_metadata: Mapping[str, object] | None, parent_trace_id: str | None
) -> dict[str, object]:  # mutable-ok: the proxy's checks merge metadata and tags into the body they are handed
    parent: Final[Mapping[str, object]] = parent_litellm_metadata or {}
    end_user_id: Final = parent.get("user_api_key_end_user_id")
    metadata_trace_id: Final = parent.get("trace_id")
    trace_id: Final = metadata_trace_id if isinstance(metadata_trace_id, str) and metadata_trace_id else parent_trace_id
    return {
        "model": model,
        **({"user": end_user_id} if isinstance(end_user_id, str) and end_user_id else {}),
        **({"litellm_trace_id": trace_id} if isinstance(trace_id, str) and trace_id else {}),
    }


async def caller_within_model_budget(
    user_api_key_auth: Optional["UserAPIKeyAuth"],
    model: str,
) -> CallerModelVerdict:
    """Return ``allowed`` when the caller is within their per-model budget for ``model``.

    Mirrors the per-model budget enforcement that ``user_api_key_auth`` runs for
    the client-requested model. Returns ``allowed`` outside the proxy or when no
    per-model budget is configured, ``denied`` over a budget, and ``denied_for_now``
    when a budget read fails.

    Every scope is checked because the sub-call's spend is charged to every
    scope: the key, team, user, team-member and end-user budgets ride into the
    sub-call's metadata, so skipping one of them would let the sub-call
    increment a counter it can never be refused by.
    """
    if user_api_key_auth is None:
        return "allowed"
    try:
        from litellm.proxy.proxy_server import model_max_budget_limiter
    except Exception:
        return "allowed"

    model_max_budget: Final = getattr(user_api_key_auth, "model_max_budget", None)
    token: Final = getattr(user_api_key_auth, "token", None)
    if isinstance(model_max_budget, dict) and model_max_budget and token is not None:
        try:
            await model_max_budget_limiter.is_key_within_model_budget(
                user_api_key_dict=user_api_key_auth,
                model=model,
            )
        except litellm.BudgetExceededError:
            return "denied"
        except Exception as e:
            verbose_logger.warning(
                "proxy sub-call: unexpected error during key model-budget check for model=%s; denying: %s",
                model,
                e,
            )
            return "denied_for_now"

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
            return "denied"
        except Exception as e:  # noqa: BLE001  # a budget gate denies on any failure, as the key and end-user scopes do
            verbose_logger.warning(
                "proxy sub-call: unexpected error during user model-budget check for model=%s; denying: %s",
                model,
                e,
            )
            return "denied_for_now"

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
            return "denied"
        except Exception as e:  # noqa: BLE001  # a budget gate denies on any failure, as the other scopes do
            verbose_logger.warning(
                "proxy sub-call: unexpected error during team model-budget check for model=%s; denying: %s",
                model,
                e,
            )
            return "denied_for_now"

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
            return "denied"
        except Exception as e:
            verbose_logger.warning(
                "proxy sub-call: unexpected error during end-user model-budget check for model=%s; denying: %s",
                model,
                e,
            )
            return "denied_for_now"

    return await _caller_within_team_member_model_budget(
        user_api_key_auth=user_api_key_auth,
        model_max_budget_limiter=model_max_budget_limiter,
        model=model,
    )


async def _caller_within_team_member_model_budget(
    user_api_key_auth: "UserAPIKeyAuth",
    model_max_budget_limiter: _TeamMemberModelBudgetLimiter,
    model: str,
) -> CallerModelVerdict:
    team_member_model_max_budget: Final = user_api_key_auth.team_member_model_max_budget
    user_id: Final = user_api_key_auth.user_id
    team_id: Final = user_api_key_auth.team_id
    if (
        not isinstance(team_member_model_max_budget, Mapping)
        or not team_member_model_max_budget
        or not isinstance(user_id, str)
        or not isinstance(team_id, str)
    ):
        return "allowed"
    try:
        await model_max_budget_limiter.is_team_member_within_model_budget(
            user_id=user_id,
            team_id=team_id,
            team_member_model_max_budget=team_member_model_max_budget,
            model=model,
        )
    except litellm.BudgetExceededError:
        return "denied"
    except Exception as e:  # noqa: BLE001  # a budget gate denies on any failure, as the other scopes do
        verbose_logger.warning(
            "proxy sub-call: unexpected error during team member model-budget check for model=%s; denying: %s",
            model,
            e,
        )
        return "denied_for_now"
    return "allowed"


def _sub_call_descriptors(
    descriptors: "Sequence[RateLimitDescriptor]",
) -> "tuple[RateLimitDescriptor, ...]":
    return tuple(_sub_call_descriptor(descriptor) for descriptor in descriptors)


def _sub_call_descriptor(descriptor: "RateLimitDescriptor") -> "RateLimitDescriptor":
    rate_limit: Final = descriptor.get("rate_limit")
    if rate_limit is None:
        return descriptor
    caller_wide: Final = not descriptor["key"].startswith(_MODEL_SCOPED_DESCRIPTOR_PREFIX)
    windowed_limits: Final[RateLimitDescriptorRateLimitObject] = {
        "requests_per_unit": None if caller_wide else rate_limit.get("requests_per_unit"),
        "tokens_per_unit": rate_limit.get("tokens_per_unit"),
        "window_size": rate_limit.get("window_size"),
    }
    return {**descriptor, "rate_limit": windowed_limits}


async def caller_within_model_rate_limit(
    user_api_key_auth: Optional["UserAPIKeyAuth"],
    model: str,
) -> CallerModelVerdict:
    """Return ``allowed`` when the caller is within their configured RPM/TPM limits for ``model``.

    This mirrors the read side of
    ``_PROXY_MaxParallelRequestsHandler_v3.async_pre_call_hook`` for ``model``:
    it builds the same descriptor set and runs the check in ``read_only`` mode
    so no counter is reserved or incremented. The sub-call's tokens are still
    charged once by the limiter's post-call success hook (via the propagated
    ``litellm_metadata``), but it takes no request slot, so it does not count
    toward the caller's RPM. ``max_parallel_requests`` gauges and the RPM of
    every caller-wide scope (key, user, team, org, ...) are left out of the
    check: the sub-call runs inside the caller's already admitted request,
    whose own slot would otherwise count against it. The RPM of a limit on
    ``model`` itself still gates the sub-call, as do token limits.

    Returns ``allowed`` outside the proxy, when the active limiter does not
    expose the read-only descriptor check (legacy limiter), or when the
    descriptor set cannot be built. ``denied_for_now`` comes from a definitive
    ``OVER_LIMIT`` response, which the next window clears, and from the limiter's own fail-closed rejection
    (``RateLimitUnverifiableError``, raised when ``fail_closed_rate_limit_enforcement``
    is on and the counters could not be verified), so any other internal error
    here lets the sub-call run rather than blocking every one.
    """
    if user_api_key_auth is None:
        return "allowed"
    try:
        from litellm.proxy.hooks.parallel_request_limiter_v3 import RateLimitUnverifiableError
        from litellm.proxy.proxy_server import proxy_logging_obj
    except Exception:
        return "allowed"

    get_proxy_hook: Final[_GetProxyHook | None] = getattr(proxy_logging_obj, "get_proxy_hook", None)
    limiter: Final[object] = get_proxy_hook("parallel_request_limiter") if get_proxy_hook is not None else None
    return await rate_limit_verdict(
        user_api_key_auth=user_api_key_auth, model=model, limiter=limiter, unverifiable=RateLimitUnverifiableError
    )


async def rate_limit_verdict(
    user_api_key_auth: "UserAPIKeyAuth",
    model: str,
    limiter: object,
    unverifiable: type[Exception],
) -> CallerModelVerdict:
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
        return "allowed"

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
        descriptors: Final = _sub_call_descriptors(
            (*base_descriptors, *create_org_descriptors(user_api_key_auth, model))
        )
        if not descriptors:
            return "allowed"
        parent_otel_span: Final[object] = getattr(user_api_key_auth, "parent_otel_span", None)
        response: Final[RateLimitResponse] = await should_rate_limit_check(
            descriptors=descriptors,
            parent_otel_span=parent_otel_span,
            read_only=True,
        )
    except unverifiable as e:
        verbose_logger.warning(
            "proxy sub-call: rate-limit counters for model=%s could not be verified; denying for now: %s",
            model,
            e,
        )
        return "denied_for_now"
    except Exception as e:
        verbose_logger.warning(
            "proxy sub-call: unexpected error during rate-limit check for model=%s; allowing: %s",
            model,
            e,
        )
        return "allowed"
    return "allowed" if response.get("overall_code") != "OVER_LIMIT" else "denied_for_now"
