from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Final, TypeAlias

import pytest

import litellm
from litellm.caching import DualCache
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.tag import LiteLLM_TagTable
from litellm.proxy import proxy_server
from litellm.proxy._types import (
    CallInfo,
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTableCachedObj,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth import auth_checks
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    model_access_group_cache_key,
    tag_cache_key,
)
from litellm.proxy.spend_tracking import native_budgets
from litellm.proxy.utils import ProxyLogging
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget

TOKEN: Final = "hashed-checks"
TEAM: Final = "team-checks"
USER: Final = "user-checks"


class _RecordedAlerts(ProxyLogging):
    def __init__(self) -> None:
        super().__init__(user_api_key_cache=DualCache())
        self.sent: list[tuple[str, Mapping[str, object]]] = []  # mutable-ok: the alerts the check fired, in order

    async def budget_alerts(self, type: str, user_info: CallInfo) -> None:
        self.sent.append((type, user_info.model_dump()))


class _NoDatabase:
    pass


@dataclass(frozen=True, slots=True)
class Outcome:
    error: object
    alerts: tuple[tuple[str, Mapping[str, object]], ...]
    throttle_pct: object


Check: TypeAlias = Callable[[_RecordedAlerts, UserApiKeyCache], Awaitable[UserAPIKeyAuth | None]]


@dataclass(frozen=True, slots=True)
class Case:
    counters: Mapping[str, float]
    run: Check
    proxy_max_budget: float = 0.0


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[SpendCounterStore]:
    spend_counter_store: Final = SpendCounterStore(connect(None).spend_counters())
    monkeypatch.setattr(proxy_server, "spend_counter_store", spend_counter_store)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", 0.25)
    yield spend_counter_store
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # later tests run without a native owner


def _error_fields(error: BaseException | None) -> object:
    if error is None:
        return None
    return (
        type(error),
        str(error),
        {
            name: _comparable(getattr(error, name, None))
            for name in ("current_cost", "max_budget", "entity_type", "entity_id", "status_code", "message")
        },
    )


def _comparable(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return (type(value), value)


def jsonable(value: object) -> object:
    match value:
        case type():
            return f"{value.__module__}.{value.__qualname__}"
        case Enum():
            return jsonable(value.value)
        case datetime():
            return value.isoformat()
        case float() if math.isnan(value):
            return "nan"
        case float() if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        case bool() | int() | float() | str() | None:
            return value
        case Mapping():
            return {str(key): jsonable(item) for key, item in value.items()}
        case tuple() | list():
            return [jsonable(item) for item in value]
        case _:
            return repr(value)


def outcome_record(outcome: Outcome) -> object:
    return jsonable({"error": outcome.error, "alerts": outcome.alerts, "throttle_pct": outcome.throttle_pct})


GOLDEN: Final = Path(__file__).parent / "native_budgets_golden" / "native_budget_checks.json"


async def _outcome(store: SpendCounterStore, case: Case, native: bool) -> Outcome:
    native_budgets._installed.clear()  # pyright: ignore[reportPrivateUsage]  # each run picks its own side
    if native:
        callbacks: Final[list[object]] = []  # mutable-ok: install_native_budgets inserts into litellm.callbacks
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv(native_budgets.NATIVE_BUDGETS_ENV, "true")
            native_budgets.install_native_budgets(store, callbacks)
    store.flush_local()
    for counter, value in case.counters.items():
        store.set_local(counter, value)
    alerts: Final = _RecordedAlerts()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(litellm, "max_budget", case.proxy_max_budget)
        token, error = await _run(case.run, alerts)
    await asyncio.sleep(0)
    return Outcome(
        error=_error_fields(error),
        alerts=tuple(alerts.sent),
        throttle_pct=None if token is None else _comparable(token.budget_throttle_pct),
    )


async def _run(check: Check, alerts: _RecordedAlerts) -> tuple[UserAPIKeyAuth | None, BaseException | None]:
    try:
        return await check(alerts, UserApiKeyCache()), None
    except Exception as raised:  # noqa: BLE001  # the raised error is the observable being compared
        return None, raised


def _key(
    max_budget: float | None = 10.0,
    key_alias: str | None = None,
    key_name: str | None = None,
    throttle: bool = False,
    soft_budget: float | None = None,
    spend: float = 0.0,
    org_id: str | None = None,
    budget_limits: list[dict[str, object]] | None = None,  # mutable-ok: the field's own type
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        token=TOKEN,
        max_budget=max_budget,
        spend=spend,
        user_id=USER,
        team_id=TEAM,
        key_alias=key_alias,
        key_name=key_name,
        soft_budget=soft_budget,
        org_id=org_id,
        budget_limits=budget_limits,
        tpm_limit=100 if throttle else None,
        metadata={"throttle_on_budget_exceeded": True} if throttle else {},
    )


def _key_max(
    max_budget: float, key_alias: str | None = None, key_name: str | None = None, throttle: bool = False
) -> Check:
    async def run(alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> UserAPIKeyAuth:
        token: Final = _key(max_budget, key_alias=key_alias, key_name=key_name, throttle=throttle)
        await auth_checks.virtual_key_max_budget_check(valid_token=token, proxy_logging_obj=alerts)
        return token

    return run


def _key_window(max_budget: float) -> Check:
    async def run(_alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> UserAPIKeyAuth:
        token: Final = _key(budget_limits=[{"budget_duration": "1d", "max_budget": max_budget}])
        await auth_checks._virtual_key_multi_budget_check(valid_token=token)  # pyright: ignore[reportPrivateUsage]  # the window check has no public alias
        return token

    return run


def _key_soft(spend: float, soft_budget: float | None) -> Check:
    async def run(alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> UserAPIKeyAuth:
        token: Final = _key(soft_budget=soft_budget, spend=spend)
        await auth_checks.virtual_key_soft_budget_check(valid_token=token, proxy_logging_obj=alerts)
        return token

    return run


def _team(
    max_budget: float | None = None,
    spend: float = 0.0,
    soft_budget: float | None = None,
    budget_limits: list[dict[str, object]] | None = None,  # mutable-ok: the field's own type
) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=TEAM,
        max_budget=max_budget,
        spend=spend,
        soft_budget=soft_budget,
        budget_limits=budget_limits,
        metadata={"soft_budget_alerting_emails": "a@b.c"},
    )


def _team_max(max_budget: float) -> Check:
    async def run(alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> None:
        await auth_checks.team_max_budget_check(
            team_object=_team(max_budget=max_budget), valid_token=_key(), proxy_logging_obj=alerts
        )

    return run


def _team_window(max_budget: float) -> Check:
    async def run(_alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> None:
        await auth_checks._team_multi_budget_check(  # pyright: ignore[reportPrivateUsage]  # the window check has no public alias
            team_object=_team(budget_limits=[{"budget_duration": "7d", "max_budget": max_budget}])
        )

    return run


def _team_soft(spend: float, soft_budget: float) -> Check:
    async def run(alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> None:
        await auth_checks._team_soft_budget_check(  # pyright: ignore[reportPrivateUsage]  # the soft check has no public alias
            team_object=_team(spend=spend, soft_budget=soft_budget),
            valid_token=_key(),
            proxy_logging_obj=alerts,
        )

    return run


def _team_member(max_budget: float) -> Check:
    async def run(alerts: _RecordedAlerts, cache: UserApiKeyCache) -> None:
        await auth_checks._check_team_member_budget(  # pyright: ignore[reportPrivateUsage]  # the common-checks member check has no public alias
            team_object=_team(),
            user_object=None,
            valid_token=_key(),
            prisma_client=None,
            user_api_key_cache=cache,
            proxy_logging_obj=alerts,
            team_membership=LiteLLM_TeamMembership(
                user_id=USER, team_id=TEAM, spend=0.0, litellm_budget_table=LiteLLM_BudgetTable(max_budget=max_budget)
            ),
            team_membership_loaded=True,
        )

    return run


def _project(
    spend: float = 0.0, max_budget: float | None = None, soft_budget: float | None = None
) -> LiteLLM_ProjectTableCachedObj:
    return LiteLLM_ProjectTableCachedObj(
        project_id="project-checks",
        spend=spend,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=max_budget, soft_budget=soft_budget),
    )


def _project_max(max_budget: float) -> Check:
    async def run(alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> None:
        await auth_checks._project_max_budget_check(  # pyright: ignore[reportPrivateUsage]  # the project check has no public alias
            project_object=_project(max_budget=max_budget), valid_token=_key(), proxy_logging_obj=alerts
        )

    return run


def _project_soft(spend: float, soft_budget: float) -> Check:
    async def run(alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> None:
        await auth_checks._project_soft_budget_check(  # pyright: ignore[reportPrivateUsage]  # the project check has no public alias
            project_object=_project(spend, soft_budget=soft_budget), valid_token=_key(), proxy_logging_obj=alerts
        )

    return run


def _organization(max_budget: float) -> Check:
    async def run(alerts: _RecordedAlerts, cache: UserApiKeyCache) -> None:
        await cache.async_set_cache(
            key="org_id:org-checks:with_budget",
            value=LiteLLM_OrganizationTable(
                organization_id="org-checks",
                budget_id="b",
                created_by="admin",
                updated_by="admin",
                litellm_budget_table=LiteLLM_BudgetTable(max_budget=max_budget),
            ),
            model_type=LiteLLM_OrganizationTable,
        )
        await auth_checks.organization_max_budget_check(
            valid_token=_key(org_id="org-checks"),
            team_object=None,
            prisma_client=_NoDatabase(),  # pyright: ignore[reportArgumentType]  # the cached org means the DB is never read
            user_api_key_cache=cache,
            proxy_logging_obj=alerts,
        )

    return run


def _tag(max_budget: float) -> Check:
    async def run(alerts: _RecordedAlerts, cache: UserApiKeyCache) -> None:
        await cache.async_set_cache(
            key=tag_cache_key("tag-checks"),
            value=LiteLLM_TagTable(
                tag_name="tag-checks", litellm_budget_table=LiteLLM_BudgetTable(max_budget=max_budget)
            ),
            model_type=LiteLLM_TagTable,
        )
        await auth_checks.tag_max_budget_check_for_tags(
            tags=["tag-checks"],
            prisma_client=_NoDatabase(),  # pyright: ignore[reportArgumentType]  # the cached tag means the DB is never read
            user_api_key_cache=cache,
            proxy_logging_obj=alerts,
            valid_token=_key(),
        )

    return run


def _access_group(max_budget: float) -> Check:
    async def run(_alerts: _RecordedAlerts, cache: UserApiKeyCache) -> None:
        await cache.async_set_cache(
            key=model_access_group_cache_key("group-checks"),
            value=ModelAccessGroupBudget(access_group_name="group-checks", max_budget=max_budget),
            model_type=ModelAccessGroupBudget,
        )
        await auth_checks._model_access_group_max_budget_check(  # pyright: ignore[reportPrivateUsage]  # the group check has no public alias
            matched_model_access_groups=["group-checks"],
            prisma_client=_NoDatabase(),  # pyright: ignore[reportArgumentType]  # the cached group means the DB is never read
            user_api_key_cache=cache,
        )

    return run


def _end_user(max_budget: float) -> Check:
    async def run(_alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> None:
        await auth_checks.check_end_user_budget(
            end_user_obj=LiteLLM_EndUserTable(
                user_id="customer-checks",
                blocked=False,
                litellm_budget_table=LiteLLM_BudgetTable(max_budget=max_budget),
            ),
            route="/chat/completions",
        )

    return run


def _proxy(spend: float) -> Check:
    async def run(_alerts: _RecordedAlerts, _cache: UserApiKeyCache) -> None:
        auth_checks._global_proxy_budget_check(  # pyright: ignore[reportPrivateUsage]  # the global check has no public alias
            global_proxy_spend=spend, skip_budget_checks=False, route="/chat/completions"
        )

    return run


KEY_COUNTER: Final = f"spend:key:{TOKEN}"
TEAM_COUNTER: Final = f"spend:team:{TEAM}"
MEMBER_COUNTER: Final = f"spend:team_member:{USER}:{TEAM}"
NAN: Final = float("nan")

CASES: Final[Mapping[str, Case]] = {
    "key_max_passes": Case({KEY_COUNTER: 4.0}, _key_max(10.0)),
    "key_max_refuses": Case({KEY_COUNTER: 12.5}, _key_max(10.0)),
    "key_max_refuses_at_the_boundary": Case({KEY_COUNTER: 10.0}, _key_max(10.0)),
    "key_max_refuses_naming_a_masked_key": Case(
        {KEY_COUNTER: 12.0}, _key_max(10.0, key_alias="ops", key_name="sk-...abcd")
    ),
    "key_max_refuses_hiding_an_unmasked_key_name": Case({KEY_COUNTER: 12.0}, _key_max(10.0, key_name="raw-secret")),
    "key_max_ignores_a_nan_budget": Case({KEY_COUNTER: 12.0}, _key_max(NAN)),
    "key_max_throttles_an_opted_in_key": Case({KEY_COUNTER: 12.0}, _key_max(10.0, throttle=True)),
    "key_window_passes": Case({f"{KEY_COUNTER}:window:1d": 1.0}, _key_window(5.0)),
    "key_window_refuses_at_the_boundary": Case({f"{KEY_COUNTER}:window:1d": 5.0}, _key_window(5.0)),
    "key_window_rounds_like_python": Case({f"{KEY_COUNTER}:window:1d": 0.12345}, _key_window(0.125)),
    "key_soft_stays_quiet_under": Case({}, _key_soft(4.0, 5.0)),
    "key_soft_alerts_at_the_boundary": Case({}, _key_soft(5.0, 5.0)),
    "key_soft_ignores_a_zero_budget": Case({}, _key_soft(5.0, 0.0)),
    "team_max_passes_at_the_boundary": Case({TEAM_COUNTER: 10.0}, _team_max(10.0)),
    "team_max_refuses_and_alerts": Case({TEAM_COUNTER: 10.5}, _team_max(10.0)),
    "team_window_passes": Case({f"{TEAM_COUNTER}:window:7d": 2.0}, _team_window(3.0)),
    "team_window_refuses_at_the_boundary": Case({f"{TEAM_COUNTER}:window:7d": 3.0}, _team_window(3.0)),
    "team_soft_stays_quiet_under": Case({}, _team_soft(1.0, 2.0)),
    "team_soft_alerts_at_the_boundary": Case({}, _team_soft(2.0, 2.0)),
    "team_member_passes": Case({MEMBER_COUNTER: 1.0}, _team_member(2.0)),
    "team_member_refuses_at_the_boundary": Case({MEMBER_COUNTER: 2.0}, _team_member(2.0)),
    "team_member_zero_budget_refuses": Case({MEMBER_COUNTER: 0.0}, _team_member(0.0)),
    "project_max_passes": Case({"spend:project:project-checks": 1.0}, _project_max(2.0)),
    "project_max_refuses_and_alerts_at_the_boundary": Case({"spend:project:project-checks": 2.0}, _project_max(2.0)),
    "project_soft_stays_quiet_under": Case({}, _project_soft(1.0, 2.0)),
    "project_soft_alerts_at_the_boundary": Case({}, _project_soft(2.0, 2.0)),
    "organization_passes": Case({"spend:org:org-checks": 4.99}, _organization(5.0)),
    "organization_refuses_and_alerts_at_the_boundary": Case({"spend:org:org-checks": 5.0}, _organization(5.0)),
    "organization_zero_budget_refuses": Case({"spend:org:org-checks": 0.0}, _organization(0.0)),
    "tag_passes_at_the_boundary": Case({"spend:tag:tag-checks": 3.0}, _tag(3.0)),
    "tag_refuses_over": Case({"spend:tag:tag-checks": 3.5}, _tag(3.0)),
    "tag_refuses_a_nan_budget": Case({"spend:tag:tag-checks": 1.0}, _tag(NAN)),
    "access_group_passes": Case({"spend:model_access_group:group-checks": 1.0}, _access_group(3.0)),
    "access_group_refuses_at_the_boundary": Case({"spend:model_access_group:group-checks": 3.0}, _access_group(3.0)),
    "access_group_skips_a_non_positive_budget": Case(
        {"spend:model_access_group:group-checks": 9.0}, _access_group(0.0)
    ),
    "end_user_passes_at_the_boundary": Case({"spend:end_user:customer-checks": 4.0}, _end_user(4.0)),
    "end_user_refuses_over": Case({"spend:end_user:customer-checks": 4.5}, _end_user(4.0)),
    "end_user_passes_a_nan_budget": Case({"spend:end_user:customer-checks": 4.5}, _end_user(NAN)),
    "proxy_passes_at_the_boundary": Case({}, _proxy(100.0), proxy_max_budget=100.0),
    "proxy_refuses_over_an_int_budget": Case({}, _proxy(100.5), proxy_max_budget=100),
}


@pytest.mark.asyncio
@pytest.mark.parametrize(("name", "case"), CASES.items(), ids=CASES.keys())
async def test_native_check_decides_like_python(store: SpendCounterStore, name: str, case: Case) -> None:
    python: Final = json.loads(GOLDEN.read_text())["python"][name]
    native: Final = await _outcome(store, case, native=True)

    refused: Final = isinstance(python["error"], list) and python["error"][0] == jsonable(litellm.BudgetExceededError)
    assert refused == ("_refuses" in name), f"the Python reference did not do what {name} says: {python['error']}"
    assert outcome_record(native) == python
