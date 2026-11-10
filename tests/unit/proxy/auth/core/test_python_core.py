import math
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
    BudgetExemption,
    PresentedKey,
    TeamMember,
    UserModelBudgets,
    TeamMemberBudget,
)
from litellm.proxy.auth.core.identity import (
    CachedAdminKey,
    CustomIdentity,
    ExpiredKey,
    KeyAuthentication,
    KeyBlocked,
    KeyIdentity,
    KeyWithoutToken,
    MalformedKey,
    MasterKey,
    MissingKey,
    NoCustomIdentity,
    NoDatabase,
    NoMasterKey,
    NotAVirtualKey,
    OwnerDeactivated,
    ReplacedKey,
    UserAuthRoute,
    VirtualKey,
)
from litellm.proxy.auth.core.inputs import AuthorizeSettings, KeySettings, RequestFacts
from litellm.proxy.auth.core.key import hash_token
from litellm.proxy.auth.core.outcome import (
    Allow,
    Deny,
    KeyBudgetExceeded,
    MasterKeyOnlyRoute,
    ModelBudgetExceeded,
    Outcome,
    TeamMemberBudgetExceeded,
    UserAuthDisabled,
)
from litellm.proxy.auth.core.python_core import PythonCore

MASTER: Final = "sk-master"
MASTER_ONLY_ROUTE: Final = "/global/spend/reset"
NOW: Final = datetime(2026, 11, 6, 12, tzinfo=timezone.utc)
KEY_SETTINGS: Final = KeySettings(master_key=MASTER, has_db=True)
NO_DB: Final = KeySettings(master_key=MASTER, has_db=False)
AUTHORIZE_SETTINGS: Final = AuthorizeSettings(
    allow_user_auth=False,
    has_db=True,
    master_key_only_routes=frozenset({MASTER_ONLY_ROUTE}),
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
AUTHORIZED: Final = (
    "enforce_model_access",
    "read_budget_exemption",
    "send_key_budget_alerts",
    "read_model_budget_scope",
    "load_key_context",
)
AUTHORIZE_CALLS: Final = frozenset(
    {
        *AUTHORIZED,
        "read_team_member_budget",
        "alert_team_member_budget",
        "read_key_spend",
        "alert_key_budget",
        "alert_key_soft_budget",
        "read_proxy_spend",
        "alert_proxy_budget",
    }
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

    async def read_budget_exemption(self) -> BudgetExemption:
        self._record("read_budget_exemption")
        return BudgetExemption(route_skips_budgets=False, mcp_discovery=False, model_is_free=self.exempt)

    def attach_user_model_budgets(self, budgets: UserModelBudgets | None) -> None:
        self._record("attach_user_model_budgets", budgets)

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


def _calls(handler: RecordingHandler) -> tuple[str, ...]:
    return tuple(name for name, _ in handler.calls)


async def _authenticate(
    token: str | None,
    *,
    route: str = "/chat/completions",
    settings: KeySettings = KEY_SETTINGS,
    handler: RecordingHandler | None = None,
) -> tuple[KeyAuthentication, tuple[str, ...]]:
    host: Final = handler or RecordingHandler()
    authentication: Final = await PythonCore().keys().authenticate(host, _facts(token, route), settings, NOW)
    return authentication, _calls(host)


async def _authorize(
    identity: KeyIdentity,
    *,
    route: str = "/chat/completions",
    is_llm_api_route: bool = True,
    settings: AuthorizeSettings = AUTHORIZE_SETTINGS,
    handler: RecordingHandler | None = None,
) -> tuple[Outcome, tuple[str, ...]]:
    host: Final = handler or RecordingHandler()
    facts: Final = _facts("hash", route, is_llm_api_route)
    outcome: Final = await PythonCore().authorize().authorize(host, identity, facts, settings)
    return outcome, _calls(host)


async def _authorize_key(
    key: KeyFacts,
    *,
    exempt: bool = False,
    team_member_budget: TeamMemberBudget | None = None,
    key_spend: float = 0.0,
    model_budget_scope: ModelBudgetScope = NO_MODEL_BUDGETS,
    user_model_budget_exceeded: ModelBudgetExceeded | None = None,
) -> tuple[Outcome, tuple[str, ...]]:
    handler: Final = RecordingHandler(
        key=key,
        exempt=exempt,
        team_member_budget=team_member_budget,
        key_spend=key_spend,
        model_budget_scope=model_budget_scope,
        user_model_budget_exceeded=user_model_budget_exceeded,
    )
    return await _authorize(VirtualKey(key=key), handler=handler)


def _allowed(budget_throttle_pct: float | None = None) -> Outcome:
    return Allow(budget_throttle_pct=budget_throttle_pct)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "token", "settings", "expected"),
    [
        ("/chat/completions", None, KeySettings(None, True), NoMasterKey(token=None)),
        ("/chat/completions", "sk-x", KeySettings(None, False), NoMasterKey(token="sk-x")),
        ("/chat/completions", None, KEY_SETTINGS, MissingKey()),
        ("/chat/completions", "", KEY_SETTINGS, MalformedKey()),
        ("/user/auth", "", KEY_SETTINGS, MalformedKey()),
        ("/user/auth", "sk-x", KEY_SETTINGS, UserAuthRoute()),
        ("/user/auth", MASTER, KEY_SETTINGS, UserAuthRoute()),
    ],
)
async def test_credential_decisions_do_no_io(
    route: str, token: str | None, settings: KeySettings, expected: KeyAuthentication
):
    assert await _authenticate(token, route=route, settings=settings) == (expected, ())


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/chat/completions", MASTER_ONLY_ROUTE])
async def test_master_key_resolves_end_user_without_key_prefetch(route: str):
    handler: Final = RecordingHandler()
    assert await _authenticate(MASTER, route=route, handler=handler) == (MasterKey(), ("resolve_end_user",))
    assert handler.calls == (("resolve_end_user", False),)


@pytest.mark.asyncio
async def test_unknown_virtual_key_is_loaded_from_the_db_then_verified():
    handler: Final = RecordingHandler()
    authentication, calls = await _authenticate("sk-x", handler=handler)
    assert authentication == VirtualKey(key=PLAIN_KEY)
    assert calls == ("resolve_end_user", "resolve_presented_key", "load_key", "prepare_key", "inspect_key")
    assert handler.calls[2] == ("load_key", hash_token("sk-x"))
    assert handler.calls[3] == ("prepare_key", True)


@pytest.mark.asyncio
async def test_cached_virtual_key_is_verified_without_a_db_load():
    handler: Final = RecordingHandler(presented=NonAdminKey())
    authentication, calls = await _authenticate("sk-x", handler=handler)
    assert authentication == VirtualKey(key=PLAIN_KEY)
    assert calls == ("resolve_end_user", "resolve_presented_key", "prepare_key", "inspect_key")
    assert handler.calls[2] == ("prepare_key", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("expires", [None, NOW + timedelta(seconds=1), NOW])
async def test_unexpired_cached_admin_key_skips_the_db_and_key_checks(expires: datetime | None):
    authentication, calls = await _authenticate(
        "sk-admin", route=MASTER_ONLY_ROUTE, settings=NO_DB, handler=RecordingHandler(presented=AdminKey(expires))
    )
    assert (authentication, calls) == (CachedAdminKey(), ("resolve_end_user", "resolve_presented_key"))


@pytest.mark.asyncio
async def test_expired_cached_admin_key_is_evicted_then_denied_with_the_presented_key():
    expires: Final = NOW - timedelta(seconds=1)
    handler: Final = RecordingHandler(presented=AdminKey(expires=expires))
    authentication, calls = await _authenticate("sk-admin", handler=handler)
    assert authentication == ExpiredKey(expires=expires, now=NOW, key_label="sk-admin")
    assert calls == ("resolve_end_user", "resolve_presented_key", "evict_presented_key")
    assert handler.calls[-1] == ("evict_presented_key", hash_token("sk-admin"))


@pytest.mark.asyncio
@pytest.mark.parametrize("presented", [None, NonAdminKey()])
async def test_virtual_key_needs_a_database(presented: PresentedKey | None):
    authentication, _ = await _authenticate("sk-x", settings=NO_DB, handler=RecordingHandler(presented=presented))
    assert authentication == NoDatabase()


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["eyJhbGciOiJIUzI1NiJ9.e30.sig", "a" * 64, "sk"])
async def test_uncached_token_without_virtual_key_prefix_is_rejected_before_the_db(token: str):
    assert await _authenticate(token) == (NotAVirtualKey(), ("resolve_end_user", "resolve_presented_key"))


@pytest.mark.asyncio
async def test_expired_virtual_key_is_labelled_by_the_hash_it_was_loaded_with():
    expires: Final = NOW - timedelta(seconds=1)
    authentication, _ = await _authenticate("sk-x", handler=RecordingHandler(key=replace(PLAIN_KEY, expires=expires)))
    assert authentication == ExpiredKey(expires=expires, now=NOW, key_label=hash_token("sk-x"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "owner", "expected", "calls"),
    [
        (replace(PLAIN_KEY, blocked=True), None, KeyBlocked(), ("inspect_key",)),
        (
            replace(PLAIN_KEY, user_id="u1"),
            OwnerFacts(deactivated=True, model_budgets=None),
            OwnerDeactivated(user_id="u1"),
            ("inspect_key", "load_key_owner", "attach_user_model_budgets"),
        ),
        (replace(PLAIN_KEY, has_token=False), None, KeyWithoutToken(), ("inspect_key",)),
        (
            replace(PLAIN_KEY, user_id="u1"),
            OwnerFacts(deactivated=False, model_budgets=None),
            None,
            ("inspect_key", "load_key_owner", "attach_user_model_budgets"),
        ),
        (replace(PLAIN_KEY, user_id="u1"), None, None, ("inspect_key", "load_key_owner")),
    ],
)
async def test_verify_decides_whether_the_resolved_key_is_usable(
    key: KeyFacts, owner: OwnerFacts | None, expected: KeyAuthentication | None, calls: tuple[str, ...]
):
    handler: Final = RecordingHandler(key=key, owner=owner)
    verified: Final = await PythonCore().keys().verify(handler, _facts("hash"), NOW)
    assert verified == (VirtualKey(key=key) if expected is None else expected)
    assert _calls(handler) == calls


@pytest.mark.asyncio
@pytest.mark.parametrize("deactivated", [False, True])
async def test_the_owner_model_budget_is_attached_to_the_key_even_when_the_owner_is_deactivated(deactivated: bool):
    budgets: Final = UserModelBudgets(limits={"gpt-4o": {"budget_limit": 1.0, "time_period": "1mo"}})
    handler: Final = RecordingHandler(
        key=replace(PLAIN_KEY, user_id="u1"), owner=OwnerFacts(deactivated=deactivated, model_budgets=budgets)
    )
    _ = await PythonCore().keys().verify(handler, _facts("hash"), NOW)
    assert handler.calls[2] == ("attach_user_model_budgets", budgets)


@pytest.mark.asyncio
async def test_a_key_without_an_owner_row_keeps_its_own_model_budget():
    handler: Final = RecordingHandler(key=replace(PLAIN_KEY, user_id="u1"), owner=None)
    _ = await PythonCore().keys().verify(handler, _facts("hash"), NOW)
    assert "attach_user_model_budgets" not in _calls(handler)


@pytest.mark.asyncio
async def test_verify_labels_an_expired_key_with_the_hash_it_was_given():
    expired: Final = replace(PLAIN_KEY, expires=NOW - timedelta(seconds=1))
    verified: Final = await PythonCore().keys().verify(RecordingHandler(key=expired), _facts("hash"), NOW)
    assert verified == ExpiredKey(expires=NOW - timedelta(seconds=1), now=NOW, key_label="hash")


@pytest.mark.asyncio
async def test_expired_key_over_its_team_member_budget_is_denied_as_expired_without_a_budget_alert():
    key: Final = replace(
        PLAIN_KEY, user_id="u1", team_member=TeamMember(user_id="u1", team_id="t1"), expires=NOW - timedelta(seconds=1)
    )
    handler: Final = RecordingHandler(key=key, team_member_budget=TeamMemberBudget(spend=1.0, max_budget=0.5))
    authentication, calls = await _authenticate("sk-x", handler=handler)
    assert authentication == ExpiredKey(expires=NOW - timedelta(seconds=1), now=NOW, key_label=hash_token("sk-x"))
    assert AUTHORIZE_CALLS.isdisjoint(calls)


@pytest.mark.asyncio
async def test_deactivated_owner_is_denied_before_model_access_is_checked():
    handler: Final = RecordingHandler(
        key=replace(PLAIN_KEY, user_id="u1"), owner=OwnerFacts(deactivated=True, model_budgets=None)
    )
    authentication, calls = await _authenticate("sk-x", handler=handler)
    assert authentication == OwnerDeactivated(user_id="u1")
    assert "enforce_model_access" not in calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("handler", "settings", "expected"),
    [
        (RecordingHandler(), NO_DB, NoDatabase()),
        (RecordingHandler(key=replace(PLAIN_KEY, blocked=True)), KEY_SETTINGS, KeyBlocked()),
    ],
)
async def test_master_key_only_route_no_longer_hides_why_a_key_is_unusable(
    handler: RecordingHandler, settings: KeySettings, expected: KeyAuthentication
):
    authentication, _ = await _authenticate("sk-x", route=MASTER_ONLY_ROUTE, settings=settings, handler=handler)
    assert authentication == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("identity", "allow_user_auth", "expected"),
    [
        (UserAuthRoute(), True, _allowed()),
        (UserAuthRoute(), False, Deny(UserAuthDisabled())),
        (NoMasterKey(token="sk-x"), False, _allowed()),
        (MasterKey(), False, _allowed()),
        (CachedAdminKey(), False, _allowed()),
    ],
)
async def test_non_virtual_key_identities_are_authorized_without_io(
    identity: KeyIdentity, allow_user_auth: bool, expected: Outcome
):
    settings: Final = replace(AUTHORIZE_SETTINGS, allow_user_auth=allow_user_auth)
    assert await _authorize(identity, route=MASTER_ONLY_ROUTE, settings=settings) == (expected, ())


@pytest.mark.asyncio
async def test_virtual_key_on_a_master_key_only_route_is_denied_before_any_io():
    assert await _authorize(VirtualKey(key=PLAIN_KEY), route=MASTER_ONLY_ROUTE) == (Deny(MasterKeyOnlyRoute()), ())


@pytest.mark.asyncio
async def test_plain_virtual_key_runs_every_authorization_step_in_order():
    assert await _authorize_key(PLAIN_KEY) == (_allowed(), AUTHORIZED)


@pytest.mark.asyncio
async def test_exempt_request_skips_every_budget_check():
    key: Final = replace(
        PLAIN_KEY,
        user_id="u1",
        team_member=TeamMember(user_id="u1", team_id="t1"),
        max_budget=1.0,
        spend=5.0,
        soft_budget=1.0,
    )
    assert await _authorize_key(key, exempt=True) == (
        _allowed(),
        ("enforce_model_access", "read_budget_exemption", "load_key_context"),
    )


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
    outcome, calls = await _authorize_key(key, team_member_budget=budget)
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
    _, calls = await _authorize_key(
        replace(PLAIN_KEY, user_id="u1"), team_member_budget=TeamMemberBudget(spend=1.0, max_budget=0.5)
    )
    assert "read_team_member_budget" not in calls


@pytest.mark.asyncio
async def test_team_member_budget_needs_a_database():
    key: Final = replace(PLAIN_KEY, user_id="u1", team_member=TeamMember(user_id="u1", team_id="t1"))
    handler: Final = RecordingHandler(key=key, team_member_budget=TeamMemberBudget(spend=1.0, max_budget=0.5))
    _, calls = await _authorize(
        VirtualKey(key=key), settings=replace(AUTHORIZE_SETTINGS, has_db=False), handler=handler
    )
    assert "read_team_member_budget" not in calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("spend", "max_budget", "throttle", "expected"),
    [
        (5.0, 5.0, None, Deny(KeyBudgetExceeded(spend=5.0, max_budget=5.0))),
        (6.0, 5.0, None, Deny(KeyBudgetExceeded(spend=6.0, max_budget=5.0))),
        (4.9, 5.0, None, _allowed()),
        (6.0, 5.0, 0.5, _allowed(budget_throttle_pct=0.5)),
        (4.9, 5.0, 0.5, _allowed()),
        (1e9, math.inf, None, _allowed()),
        (1e9, math.nan, None, _allowed()),
    ],
)
async def test_key_budget_denies_or_throttles_at_the_limit(
    spend: float, max_budget: float, throttle: float | None, expected: Outcome
):
    key: Final = replace(PLAIN_KEY, max_budget=max_budget, budget_throttle_pct=throttle)
    handler: Final = RecordingHandler(key=key, key_spend=spend)
    outcome, calls = await _authorize(VirtualKey(key=key), handler=handler)
    assert outcome == expected
    assert calls[2:5] == ("send_key_budget_alerts", "read_key_spend", "alert_key_budget")
    assert handler.calls[4] == ("alert_key_budget", spend)
    assert ("read_model_budget_scope" in calls) == isinstance(expected, Allow)


@pytest.mark.asyncio
async def test_key_budget_is_enforced_only_on_llm_routes():
    key: Final = replace(PLAIN_KEY, max_budget=1.0)
    outcome, calls = await _authorize(
        VirtualKey(key=key), is_llm_api_route=False, handler=RecordingHandler(key=key, key_spend=2.0)
    )
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
    outcome, calls = await _authorize_key(replace(PLAIN_KEY, spend=spend, soft_budget=soft_budget))
    assert outcome == _allowed()
    soft_alert: Final = ("alert_key_soft_budget",) if alerted else ()
    assert calls[2:] == ("send_key_budget_alerts", *soft_alert, "read_model_budget_scope", "load_key_context")


@pytest.mark.asyncio
async def test_throttled_key_still_gets_soft_and_model_budget_checks():
    key: Final = replace(PLAIN_KEY, max_budget=5.0, budget_throttle_pct=0.5, spend=6.0, soft_budget=1.0)
    outcome, calls = await _authorize_key(key, key_spend=6.0)
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
    settings: Final = replace(AUTHORIZE_SETTINGS, proxy_max_budget=proxy_max_budget, has_db=has_db)
    handler: Final = RecordingHandler(exempt=exempt, proxy_spend=proxy_spend)
    outcome, _ = await _authorize(VirtualKey(key=PLAIN_KEY), settings=settings, handler=handler)
    assert outcome == _allowed()
    assert handler.calls[-len(tail) :] == tail


@pytest.mark.asyncio
async def test_a_model_budget_denial_stops_authorization_before_the_key_context_loads():
    exceeded: Final = ModelBudgetExceeded(
        message="over", current_cost=2.0, max_budget=1.0, entity_type="user", entity_id="u-1"
    )
    scope: Final = replace(NO_MODEL_BUDGETS, models=("gpt-4o",), user_has_model_budgets=True)
    outcome, calls = await _authorize_key(PLAIN_KEY, model_budget_scope=scope, user_model_budget_exceeded=exceeded)
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
