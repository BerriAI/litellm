import math
from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.proxy.auth.core.answers import (
    AdminKey,
    KeyFacts,
    ModelBudgetScope,
    NonAdminKey,
    OwnerFacts,
    PresentedKey,
    TeamMember,
    TeamMemberBudget,
)
from litellm.proxy.auth.core.inputs import RequestFacts, KeySettings
from litellm.proxy.auth.core.outcome import (
    CustomIdentity,
    NoCustomIdentity,
    ReplacedKey,
    Allow,
    CachedAdminKey,
    Deny,
    ExpiredKey,
    KeyBlocked,
    KeyBudgetExceeded,
    KeyWithoutToken,
    MalformedKey,
    MasterKey,
    MasterKeyOnlyRoute,
    MissingKey,
    ModelBudgetExceeded,
    NoDatabase,
    NoMasterKey,
    NotAVirtualKey,
    Outcome,
    OwnerDeactivated,
    TeamMemberBudgetExceeded,
    UserAuthDisabled,
    UserAuthRoute,
    VirtualKey,
)
from litellm.proxy.auth.core.key import hash_token
from litellm.proxy.auth.core.python_core import PythonCore

MASTER: Final = "sk-master"
NOW: Final = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
SETTINGS: Final = KeySettings(
    master_key=MASTER,
    allow_user_auth=False,
    has_db=True,
    master_key_only_routes=frozenset({"/global/spend/reset"}),
    proxy_max_budget=0.0,
)
PLAIN_KEY: Final = KeyFacts(
    has_token=True,
    blocked=False,
    expires=None,
    spend=0.0,
    soft_budget=None,
    max_budget=None,
    user_id=None,
    team_member=None,
    budget_throttle_pct=None,
)
CHECKED: Final = (
    "prepare_key",
    "inspect_key",
    "enforce_model_access",
    "check_budget_exemption",
    "send_key_budget_alerts",
    "read_model_budget_scope",
    "load_key_context",
)

NO_MODEL_BUDGETS: Final = ModelBudgetScope(
    models=(),
    rewritable_model=None,
    key=None,
    user_has_model_budgets=False,
)

Call = tuple[str, object]


@dataclass
class RecordingHandler:
    presented: PresentedKey | None = None
    key: KeyFacts = PLAIN_KEY
    owner: OwnerFacts | None = None
    exempt: bool = False
    team_member_budget: TeamMemberBudget | None = None
    key_spend: float = 0.0
    proxy_spend: float | None = None
    model_budget_scope: ModelBudgetScope = NO_MODEL_BUDGETS
    user_model_budget_exceeded: ModelBudgetExceeded | None = None
    calls: tuple[Call, ...] = ()

    def _record(self, name: str, payload: object = None) -> None:
        self.calls = (*self.calls, (name, payload))

    def model_budgets(self) -> "RecordingHandler":
        return self

    async def resolve_end_user(self, prefetch_key: bool) -> None:
        self._record("resolve_end_user", prefetch_key)

    async def resolve_presented_key(self, hashed_token: str) -> PresentedKey | None:
        self._record("resolve_presented_key", hashed_token)
        return self.presented

    async def evict_presented_key(self, hashed_token: str) -> None:
        self._record("evict_presented_key", hashed_token)

    async def load_key(self, hashed_token: str) -> None:
        self._record("load_key", hashed_token)

    async def prepare_key(self, loaded_from_db: bool) -> None:
        self._record("prepare_key", loaded_from_db)

    async def inspect_key(self) -> KeyFacts:
        self._record("inspect_key")
        return self.key

    async def enforce_model_access(self) -> None:
        self._record("enforce_model_access")

    async def load_key_owner(self, user_id: str) -> OwnerFacts | None:
        self._record("load_key_owner", user_id)
        return self.owner

    async def check_budget_exemption(self) -> bool:
        self._record("check_budget_exemption")
        return self.exempt

    async def read_team_member_budget(self, member: TeamMember) -> TeamMemberBudget | None:
        self._record("read_team_member_budget", member)
        return self.team_member_budget

    async def alert_team_member_budget(self, member: TeamMember, budget: TeamMemberBudget) -> None:
        self._record("alert_team_member_budget", budget)

    async def send_key_budget_alerts(self) -> None:
        self._record("send_key_budget_alerts")

    async def read_key_spend(self, max_budget: float) -> float:
        self._record("read_key_spend", max_budget)
        return self.key_spend

    async def alert_key_budget(self, spend: float) -> None:
        self._record("alert_key_budget", spend)

    async def alert_key_soft_budget(self) -> None:
        self._record("alert_key_soft_budget")

    async def read_model_budget_scope(self) -> ModelBudgetScope:
        self._record("read_model_budget_scope")
        return self.model_budget_scope

    async def check_key_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        self._record("check_key_model_budget", model)
        return None

    async def authorize_budget_fallback(self, model: str) -> bool:
        self._record("authorize_budget_fallback", model)
        return True

    async def rewrite_request_model(self, model: str) -> None:
        self._record("rewrite_request_model", model)

    async def check_user_model_budget(self, model: str) -> ModelBudgetExceeded | None:
        self._record("check_user_model_budget", model)
        return self.user_model_budget_exceeded

    async def load_key_context(self) -> None:
        self._record("load_key_context")

    async def read_proxy_spend(self) -> float | None:
        self._record("read_proxy_spend")
        return self.proxy_spend

    async def alert_proxy_budget(self, spend: float) -> None:
        self._record("alert_proxy_budget", spend)


def _facts(token: str | None, route: str = "/chat/completions", is_llm_api_route: bool = True) -> RequestFacts:
    return RequestFacts(route=route, token=token, is_llm_api_route=is_llm_api_route)


async def _run(decision: Awaitable[Outcome], handler: RecordingHandler) -> tuple[Outcome, tuple[str, ...]]:
    outcome: Final = await decision
    return outcome, tuple(name for name, _ in handler.calls)


async def _decide(
    token: str | None,
    *,
    route: str = "/chat/completions",
    settings: KeySettings = SETTINGS,
    handler: RecordingHandler | None = None,
) -> tuple[Outcome, tuple[str, ...]]:
    host: Final = handler or RecordingHandler()
    return await _run(PythonCore().keys().authenticate(host, _facts(token, route), settings, NOW), host)


async def _check(
    key: KeyFacts,
    *,
    is_llm_api_route: bool = True,
    settings: KeySettings = SETTINGS,
    owner: OwnerFacts | None = None,
    exempt: bool = False,
    team_member_budget: TeamMemberBudget | None = None,
    key_spend: float = 0.0,
) -> tuple[Outcome, tuple[str, ...]]:
    handler: Final = RecordingHandler(
        key=key, owner=owner, exempt=exempt, team_member_budget=team_member_budget, key_spend=key_spend
    )
    facts: Final = _facts("hash", is_llm_api_route=is_llm_api_route)
    return await _run(PythonCore().keys().check(handler, facts, settings, NOW), handler)


def _allowed(budget_throttle_pct: float | None = None) -> Outcome:
    return Allow(VirtualKey(budget_throttle_pct=budget_throttle_pct))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "token", "settings", "expected"),
    [
        ("/chat/completions", None, KeySettings(None, False, True, frozenset(), 0.0), Allow(NoMasterKey(token=None))),
        (
            "/chat/completions",
            "sk-x",
            KeySettings(None, False, False, frozenset(), 0.0),
            Allow(NoMasterKey(token="sk-x")),
        ),
        ("/chat/completions", None, SETTINGS, Deny(MissingKey())),
        ("/chat/completions", "", SETTINGS, Deny(MalformedKey())),
        ("/user/auth", "", KeySettings(MASTER, True, True, frozenset(), 0.0), Deny(MalformedKey())),
        ("/user/auth", "sk-x", KeySettings(MASTER, True, True, frozenset(), 0.0), Allow(UserAuthRoute())),
        ("/user/auth", "sk-x", SETTINGS, Deny(UserAuthDisabled())),
        ("/user/auth", MASTER, SETTINGS, Deny(UserAuthDisabled())),
    ],
)
async def test_credential_decisions_do_no_io(route: str, token: str | None, settings: KeySettings, expected: Outcome):
    assert await _decide(token, route=route, settings=settings) == (expected, ())


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/chat/completions", "/global/spend/reset"])
async def test_master_key_resolves_end_user_without_key_prefetch(route: str):
    handler: Final = RecordingHandler()
    assert await _decide(MASTER, route=route, handler=handler) == (Allow(MasterKey()), ("resolve_end_user",))
    assert handler.calls == (("resolve_end_user", False),)


@pytest.mark.asyncio
async def test_unknown_virtual_key_is_loaded_from_the_db_then_checked():
    handler: Final = RecordingHandler()
    outcome, calls = await _decide("sk-x", handler=handler)
    assert outcome == _allowed()
    assert calls == ("resolve_end_user", "resolve_presented_key", "load_key", *CHECKED)
    assert handler.calls[2] == ("load_key", hash_token("sk-x"))
    assert handler.calls[3] == ("prepare_key", True)


@pytest.mark.asyncio
async def test_cached_virtual_key_is_checked_without_a_db_load():
    handler: Final = RecordingHandler(presented=NonAdminKey())
    outcome, calls = await _decide("sk-x", handler=handler)
    assert outcome == _allowed()
    assert calls == ("resolve_end_user", "resolve_presented_key", *CHECKED)
    assert handler.calls[2] == ("prepare_key", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("expires", [None, NOW + timedelta(seconds=1), NOW])
async def test_unexpired_cached_admin_key_skips_route_db_and_key_checks(expires: datetime | None):
    no_db: Final = KeySettings(MASTER, False, False, frozenset({"/global/spend/reset"}), 0.0)
    outcome, calls = await _decide(
        "sk-admin",
        route="/global/spend/reset",
        settings=no_db,
        handler=RecordingHandler(presented=AdminKey(expires=expires)),
    )
    assert (outcome, calls) == (Allow(CachedAdminKey()), ("resolve_end_user", "resolve_presented_key"))


@pytest.mark.asyncio
async def test_expired_cached_admin_key_is_evicted_then_denied_with_the_presented_key():
    expires: Final = NOW - timedelta(seconds=1)
    handler: Final = RecordingHandler(presented=AdminKey(expires=expires))
    outcome, calls = await _decide("sk-admin", handler=handler)
    assert outcome == Deny(ExpiredKey(expires=expires, now=NOW, key_label="sk-admin"))
    assert calls == ("resolve_end_user", "resolve_presented_key", "evict_presented_key")
    assert handler.calls[-1] == ("evict_presented_key", hash_token("sk-admin"))


@pytest.mark.asyncio
@pytest.mark.parametrize("presented", [None, NonAdminKey()])
async def test_non_admin_key_cannot_use_master_key_only_routes(presented: PresentedKey | None):
    outcome, calls = await _decide("sk-x", route="/global/spend/reset", handler=RecordingHandler(presented=presented))
    assert (outcome, calls) == (Deny(MasterKeyOnlyRoute()), ("resolve_end_user", "resolve_presented_key"))


@pytest.mark.asyncio
@pytest.mark.parametrize("presented", [None, NonAdminKey()])
async def test_virtual_key_needs_a_database(presented: PresentedKey | None):
    no_db: Final = KeySettings(MASTER, False, False, frozenset(), 0.0)
    outcome, _ = await _decide("sk-x", settings=no_db, handler=RecordingHandler(presented=presented))
    assert outcome == Deny(NoDatabase())


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["eyJhbGciOiJIUzI1NiJ9.e30.sig", "a" * 64, "sk"])
async def test_uncached_token_without_virtual_key_prefix_is_rejected_before_the_db(token: str):
    assert await _decide(token) == (Deny(NotAVirtualKey()), ("resolve_end_user", "resolve_presented_key"))


@pytest.mark.asyncio
async def test_expired_virtual_key_is_labelled_by_the_hash_it_was_loaded_with():
    expires: Final = NOW - timedelta(seconds=1)
    outcome, _ = await _decide("sk-x", handler=RecordingHandler(key=replace(PLAIN_KEY, expires=expires)))
    assert outcome == Deny(ExpiredKey(expires=expires, now=NOW, key_label=hash_token("sk-x")))


@pytest.mark.asyncio
async def test_blocked_key_is_denied_before_model_access():
    assert await _check(replace(PLAIN_KEY, blocked=True)) == (Deny(KeyBlocked()), ("inspect_key",))


@pytest.mark.asyncio
async def test_deactivated_owner_is_denied_after_model_access():
    outcome, calls = await _check(replace(PLAIN_KEY, user_id="u1"), owner=OwnerFacts(deactivated=True))
    assert (outcome, calls) == (
        Deny(OwnerDeactivated(user_id="u1")),
        ("inspect_key", "enforce_model_access", "load_key_owner"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, OwnerFacts(deactivated=False)])
async def test_active_or_missing_owner_continues(owner: OwnerFacts | None):
    outcome, _ = await _check(replace(PLAIN_KEY, user_id="u1"), owner=owner)
    assert outcome == _allowed()


@pytest.mark.asyncio
async def test_exempt_request_skips_every_budget_check_but_not_expiry():
    key: Final = replace(
        PLAIN_KEY,
        user_id="u1",
        team_member=TeamMember(user_id="u1", team_id="t1"),
        max_budget=1.0,
        spend=5.0,
        soft_budget=1.0,
    )
    assert await _check(key, exempt=True) == (
        _allowed(),
        ("inspect_key", "enforce_model_access", "load_key_owner", "check_budget_exemption", "load_key_context"),
    )
    expired: Final = replace(key, expires=NOW - timedelta(seconds=1))
    outcome, _ = await _check(expired, exempt=True)
    assert outcome == Deny(ExpiredKey(expires=NOW - timedelta(seconds=1), now=NOW, key_label="hash"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budget", "denied"),
    [
        (TeamMemberBudget(spend=10.0, max_budget=10.0), True),
        (TeamMemberBudget(spend=10.1, max_budget=10.0), True),
        (TeamMemberBudget(spend=9.9, max_budget=10.0), False),
        (None, False),
    ],
)
async def test_team_member_budget_alerts_then_denies_at_the_limit(budget: TeamMemberBudget | None, denied: bool):
    key: Final = replace(PLAIN_KEY, user_id="u1", team_member=TeamMember(user_id="u1", team_id="t1"))
    outcome, calls = await _check(key, team_member_budget=budget)
    if denied:
        assert budget is not None
        assert outcome == Deny(
            TeamMemberBudgetExceeded(user_id="u1", team_id="t1", spend=budget.spend, max_budget=budget.max_budget)
        )
        assert calls[-2:] == ("read_team_member_budget", "alert_team_member_budget")
    else:
        assert outcome == _allowed()
        assert "alert_team_member_budget" not in calls


@pytest.mark.asyncio
async def test_team_member_budget_is_read_only_for_a_team_member():
    key: Final = replace(PLAIN_KEY, user_id="u1")
    _, calls = await _check(key, team_member_budget=TeamMemberBudget(spend=1.0, max_budget=0.5))
    assert "read_team_member_budget" not in calls


@pytest.mark.asyncio
async def test_team_member_budget_needs_a_database():
    key: Final = replace(PLAIN_KEY, user_id="u1", team_member=TeamMember(user_id="u1", team_id="t1"))
    no_db: Final = KeySettings(MASTER, False, False, frozenset(), 0.0)
    _, calls = await _check(key, settings=no_db, team_member_budget=TeamMemberBudget(spend=1.0, max_budget=0.5))
    assert "read_team_member_budget" not in calls


@pytest.mark.asyncio
async def test_team_member_budget_is_checked_before_expiry():
    key: Final = replace(
        PLAIN_KEY, user_id="u1", team_member=TeamMember(user_id="u1", team_id="t1"), expires=NOW - timedelta(seconds=1)
    )
    outcome, _ = await _check(key, team_member_budget=TeamMemberBudget(spend=1.0, max_budget=0.5))
    assert outcome == Deny(TeamMemberBudgetExceeded(user_id="u1", team_id="t1", spend=1.0, max_budget=0.5))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spend", "max_budget", "throttle", "expected"),
    [
        (5.0, 5.0, None, Deny(KeyBudgetExceeded(spend=5.0, max_budget=5.0))),
        (6.0, 5.0, None, Deny(KeyBudgetExceeded(spend=6.0, max_budget=5.0))),
        (4.9, 5.0, None, _allowed()),
        (6.0, 5.0, 0.5, _allowed(budget_throttle_pct=0.5)),
        (4.9, 5.0, 0.5, _allowed()),
        (
            1e9,
            math.inf,
            None,
            _allowed(),
        ),
        (
            1e9,
            math.nan,
            None,
            _allowed(),
        ),
    ],
)
async def test_key_budget_denies_or_throttles_at_the_limit(
    spend: float, max_budget: float, throttle: float | None, expected: Outcome
):
    key: Final = replace(PLAIN_KEY, max_budget=max_budget, budget_throttle_pct=throttle)
    handler: Final = RecordingHandler(key=key, key_spend=spend)
    outcome, calls = await _run(PythonCore().keys().check(handler, _facts("hash"), SETTINGS, NOW), handler)
    assert outcome == expected
    assert calls[3:6] == ("send_key_budget_alerts", "read_key_spend", "alert_key_budget")
    assert handler.calls[5] == ("alert_key_budget", spend)
    assert ("read_model_budget_scope" in calls) == isinstance(expected, Allow)


@pytest.mark.asyncio
async def test_key_budget_is_enforced_only_on_llm_routes():
    key: Final = replace(PLAIN_KEY, max_budget=1.0)
    outcome, calls = await _check(key, is_llm_api_route=False, key_spend=2.0)
    assert outcome == _allowed()
    assert "read_key_spend" not in calls
    assert "read_model_budget_scope" in calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spend", "soft_budget", "alerted"),
    [
        (5.0, 5.0, True),
        (6.0, 5.0, True),
        (4.9, 5.0, False),
        (10.0, None, False),
        (10.0, 0.0, False),
        (10.0, math.nan, False),
    ],
)
async def test_soft_budget_alerts_at_the_threshold_before_model_budgets(
    spend: float, soft_budget: float | None, alerted: bool
):
    outcome, calls = await _check(replace(PLAIN_KEY, spend=spend, soft_budget=soft_budget))
    assert outcome == _allowed()
    soft_alert: Final = ("alert_key_soft_budget",) if alerted else ()
    assert calls[3:] == ("send_key_budget_alerts", *soft_alert, "read_model_budget_scope", "load_key_context")


@pytest.mark.asyncio
async def test_throttled_key_still_gets_soft_and_model_budget_checks():
    key: Final = replace(PLAIN_KEY, max_budget=5.0, budget_throttle_pct=0.5, spend=6.0, soft_budget=1.0)
    outcome, calls = await _check(key, key_spend=6.0)
    assert outcome == _allowed(budget_throttle_pct=0.5)
    assert calls[-4:] == ("alert_key_budget", "alert_key_soft_budget", "read_model_budget_scope", "load_key_context")


@pytest.mark.asyncio
@pytest.mark.parametrize("exempt", [False, True])
@pytest.mark.parametrize(
    ("proxy_max_budget", "has_db", "proxy_spend", "tail"),
    [
        (0.0, True, 50.0, (("load_key_context", None),)),
        (-1.0, True, 50.0, (("load_key_context", None),)),
        (math.nan, True, 50.0, (("load_key_context", None),)),
        (10.0, False, 50.0, (("load_key_context", None),)),
        (10.0, True, None, (("load_key_context", None), ("read_proxy_spend", None))),
        (
            10.0,
            True,
            50.0,
            (("load_key_context", None), ("read_proxy_spend", None), ("alert_proxy_budget", 50.0)),
        ),
    ],
)
async def test_proxy_budget_only_alerts_and_runs_for_exempt_requests_too(
    exempt: bool, proxy_max_budget: float, has_db: bool, proxy_spend: float | None, tail: tuple[Call, ...]
):
    settings: Final = replace(SETTINGS, proxy_max_budget=proxy_max_budget, has_db=has_db)
    handler: Final = RecordingHandler(exempt=exempt, proxy_spend=proxy_spend)
    outcome, _ = await _run(PythonCore().keys().check(handler, _facts("hash"), settings, NOW), handler)
    assert outcome == _allowed()
    assert handler.calls[-len(tail) :] == tail


@pytest.mark.asyncio
@pytest.mark.parametrize("exempt", [False, True])
async def test_key_without_token_is_denied_after_its_context_loads(exempt: bool):
    settings: Final = replace(SETTINGS, proxy_max_budget=10.0)
    handler: Final = RecordingHandler(key=replace(PLAIN_KEY, has_token=False), exempt=exempt, proxy_spend=1.0)
    outcome, calls = await _run(PythonCore().keys().check(handler, _facts("hash"), settings, NOW), handler)
    assert outcome == Deny(KeyWithoutToken())
    assert calls[-3:] == ("load_key_context", "read_proxy_spend", "alert_proxy_budget")


@pytest.mark.asyncio
async def test_a_model_budget_denial_stops_the_key_chain_before_its_context_loads():
    exceeded: Final = ModelBudgetExceeded(
        message="over", current_cost=2.0, max_budget=1.0, entity_type="user", entity_id="u-1"
    )
    scope: Final = replace(NO_MODEL_BUDGETS, models=("gpt-4o",), user_has_model_budgets=True)
    handler: Final = RecordingHandler(model_budget_scope=scope, user_model_budget_exceeded=exceeded)
    outcome, calls = await _run(PythonCore().keys().check(handler, _facts("hash"), SETTINGS, NOW), handler)
    assert outcome == Deny(exceeded)
    assert calls[-2:] == ("read_model_budget_scope", "check_user_model_budget")


@dataclass
class ScriptedCustomAuth:
    answer: CustomIdentity | ReplacedKey | NoCustomIdentity
    asked: list[RequestFacts] = field(default_factory=list)  # mutable-ok: records host calls under test

    async def custom_auth(self, facts: RequestFacts) -> CustomIdentity | ReplacedKey | NoCustomIdentity:
        self.asked.append(facts)
        return self.answer


@pytest.mark.parametrize(
    "answer",
    [CustomIdentity(user_id="u", team_id="t", hashed_token=None), ReplacedKey(token="sk-new"), NoCustomIdentity()],
)
async def test_custom_auth_returns_what_the_custom_function_resolved_for_the_request(
    answer: CustomIdentity | ReplacedKey | NoCustomIdentity,
) -> None:
    host: Final = ScriptedCustomAuth(answer)
    facts: Final = RequestFacts(route="/chat/completions", token="sk-1", is_llm_api_route=True)
    assert await PythonCore().custom().authenticate(host, facts) == answer
    assert host.asked == [facts]
