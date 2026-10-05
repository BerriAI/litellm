"""
Enforce the caller's budget against router fallback targets.

Budget is checked once, during auth, against the *requested* model group. A zero-cost group takes
`_is_model_cost_zero`'s bypass and waives every budget check; the router then picks a fallback
target after auth, inside `run_async_fallback`, and nothing re-checks budget on the group that
actually bills. So a free model with a paid fallback spends without a gate.

This predicate is injected into the router to re-check budget for each fallback target before it is
attempted, mirroring `fallback_model_access.py`. It deliberately leaves the primary attempt alone:
a zero-cost model is never blocked by budget, and only the paid fallback is refused. On by default;
set `general_settings.enforce_fallback_budget: false` to restore the unguarded behaviour.

Scope: the key's and the user's `max_budget`. Not covered yet, and each needs a read-only evaluation
path before it can be: team, team-member, end-user, org, global and per-model budgets, whose
auth-path functions enforce rather than report (they raise), so reusing them would fire threshold
alerts and take spend reservations for a target that is then skipped; and the key's rolling
`budget_limits` windows, whose accumulated spend lives only in per-window counters
(`spend:key:{token}:window:{budget_duration}`), so enforcing them means more counter reads on the
fallback path rather than reusing state auth already loaded.

Two known limitations of that narrow scope, both shared with `fallback_model_access.py`:

* This reads the spend counter, it does not reserve against it. Requests already in flight all
  observe the same pre-billing figure, so a cap can be crossed by roughly the number of concurrent
  fallbacks times their cost. Auth-time enforcement avoids this by pre-filling the counter through
  `reserve_budget_for_request`, which the zero-cost bypass skips. Turning the soft cap into a hard
  one means reserving per fallback attempt and reconciling on completion.
* A request that reaches the router without `metadata["user_api_key_auth"]` is not restricted.
  Only `add_litellm_data_to_request` populates that key, so endpoints that assemble metadata by
  hand (for example `/queue/chat/completions`) fall through as unauthenticated.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _is_model_cost_zero,  # pyright: ignore[reportPrivateUsage]  # the zero-cost predicate the auth-time budget checks use; no public equivalent
)
from litellm.router import Router


class _RequestMetadata(BaseModel):
    user_api_key_auth: UserAPIKeyAuth | None = None


class _FallbackBudgetSettings(BaseModel):
    enforce_fallback_budget: bool = True


def _token_in_metadata(metadata: object) -> UserAPIKeyAuth | None:
    try:
        return _RequestMetadata.model_validate(metadata).user_api_key_auth
    except ValidationError:
        return None


def _user_api_key_auth_from_request(request_kwargs: Mapping[str, object]) -> UserAPIKeyAuth | None:
    return next(
        (
            token
            for field in ("metadata", "litellm_metadata")
            if (token := _token_in_metadata(request_kwargs.get(field))) is not None
        ),
        None,
    )


def _enforced_by_general_settings() -> bool:
    from litellm.proxy.proxy_server import general_settings

    return _FallbackBudgetSettings.model_validate(general_settings).enforce_fallback_budget


def _applies_user_budget_to_team_keys() -> bool:
    from litellm.proxy.proxy_server import general_settings

    return general_settings.get("apply_user_budget_to_team_keys") is True


async def _counter_spend(counter_key: str, fallback_spend: float, max_budget: float) -> float:
    """
    Read a spend counter the same way the auth-time budget checks do.

    `max_budget` is not advisory: it makes `get_current_spend` re-check the counter against the
    authoritative recorded spend before admitting. A counter restored from an older Redis snapshot
    reads as a hit rather than a clean miss, so without this the reseed path never runs and a
    stale-low counter would keep admitting paid fallbacks past the cap.
    """
    from litellm.proxy.proxy_server import get_current_spend

    return await get_current_spend(
        counter_key=counter_key,
        fallback_spend=fallback_spend,
        max_budget=max_budget,
    )


async def is_token_within_budget_for_model(*, model: str, valid_token: UserAPIKeyAuth, llm_router: Router) -> bool:
    """
    True when the key and the user behind it can still pay for `model`.

    A zero-cost fallback target is always allowed: refusing it would deny a request on spend some
    other model accrued, which is the same reasoning behind the auth-time bypass.
    """
    if _is_model_cost_zero(model=model, llm_router=llm_router):
        return True

    key_budget: Final = valid_token.max_budget
    if key_budget is not None and valid_token.token is not None:
        key_spend: Final = await _counter_spend(
            counter_key=f"spend:key:{valid_token.token}",
            fallback_spend=valid_token.spend or 0.0,
            max_budget=key_budget,
        )
        if key_spend >= key_budget:
            return False

    # Mirrors `_PROXY_MaxBudgetLimiter`: a team key does not carry the key owner's personal budget
    # unless the proxy opts in, so the personal cap must not gate the fallback either.
    user_budget: Final = valid_token.user_max_budget
    if (
        user_budget is not None
        and valid_token.user_id is not None
        and (valid_token.team_id is None or _applies_user_budget_to_team_keys())
    ):
        user_spend: Final = await _counter_spend(
            counter_key=f"spend:user:{valid_token.user_id}",
            fallback_spend=valid_token.user_spend or 0.0,
            max_budget=user_budget,
        )
        if user_spend >= user_budget:
            return False

    return True


@dataclass(frozen=True, slots=True)
class RouterFallbackBudgetCheck:
    """
    `FallbackBudgetCheck` for the proxy's router: while `is_enforced()` is true, a paid fallback
    target is attempted only when the caller is still within budget. Requests that carry no key
    (for example internal health checks) are not restricted.
    """

    is_enforced: Callable[[], bool]

    async def __call__(self, *, model: str, request_kwargs: Mapping[str, object], llm_router: Router) -> bool:
        if not self.is_enforced():
            return True
        valid_token: Final = _user_api_key_auth_from_request(request_kwargs)
        if valid_token is None:
            return True
        try:
            return await is_token_within_budget_for_model(model=model, valid_token=valid_token, llm_router=llm_router)
        except Exception as e:  # noqa: BLE001  # fail closed: a spend lookup failure must not bill the caller
            verbose_proxy_logger.warning("Skipping fallback to model=%s: budget lookup failed: %s", model, e)
            return False


router_fallback_budget_check: Final = RouterFallbackBudgetCheck(is_enforced=_enforced_by_general_settings)
