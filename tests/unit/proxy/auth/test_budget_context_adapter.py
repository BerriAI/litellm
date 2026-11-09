from collections.abc import Mapping
from typing import Final, Literal
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm._auth import budget_context as core
from litellm._auth.failure import NativeAuthRuntimeError
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy._types import LiteLLM_EndUserTable, LiteLLM_TeamMembership, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth import auth_checks
from litellm.proxy.auth import budget_context_adapter as adapter
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from tests.unit.proxy.auth import phase1_budget_auth_reference as auth_reference
from tests.unit.proxy.auth import phase1_budget_context_reference as legacy


@pytest.fixture(params=(False, True), ids=("reference", "native"))
def native(request: pytest.FixtureRequest) -> bool:
    return request.param is True  # pyright: ignore[reportAny]  # fixture parameters are the two literal booleans above


@pytest.mark.parametrize("scope", ("Team", "EndUser", "CustomEndUser"))
@pytest.mark.parametrize("identity", (None, "", "entity"))
@pytest.mark.parametrize("budget_count", (0, 1, 2))
@pytest.mark.parametrize("model_count", (0, 1, 3))
def test_native_model_read_selection(
    scope: Literal["Team", "EndUser", "CustomEndUser"], identity: str | None, budget_count: int, model_count: int
) -> None:
    facts: Final = core.ModelCheckFacts(scope, identity, budget_count, model_count)
    expected: Final = (
        tuple(range(model_count)) if budget_count and identity is not None and (scope != "EndUser" or identity) else ()
    )
    assert core.model_checks(facts) == expected


@pytest.mark.parametrize("keep", (False, True))
@pytest.mark.parametrize("present", (False, True))
@pytest.mark.parametrize("limit", (None, 0, -1, 1))
@pytest.mark.parametrize("models", (None, 0, 2))
def test_native_token_assignment_table(keep: bool, present: bool, limit: int | None, models: int | None) -> None:
    facts: Final = core.TokenDefaultsFacts(core.BudgetFields(limit, limit, limit, limit, models), keep, present)
    expected: Final = tuple(
        name
        for name, value in (("MaxBudget", limit), ("Tpm", limit), ("Rpm", limit), ("Tpd", limit), ("Models", models))
        if present and (not keep or value is None)
    )
    assert core.token_assignments(facts) == expected


@pytest.mark.parametrize("mode", ("Models", "Token", "Default"))
@pytest.mark.parametrize("object_present", (False, True))
@pytest.mark.parametrize("identity", (None, "", "eu"))
@pytest.mark.parametrize("db", (False, True))
@pytest.mark.parametrize("key_id", (None, "", "key"))
def test_end_user_initial_table(
    mode: Literal["Models", "Token", "Default"],
    object_present: bool,
    identity: str | None,
    db: bool,
    key_id: str | None,
) -> None:
    facts: Final = core.EndUserFacts(mode, object_present, identity, db, key_id)
    expected: Final = (
        "Own"
        if mode != "Default" and object_present
        else "None"
        if mode == "Models" and (not identity or not db)
        else "ReadKey"
        if key_id is not None
        else "ReadGlobalConfig"
    )
    assert core.end_user_source(facts) == expected


@pytest.mark.parametrize("key_row", (False, True))
@pytest.mark.parametrize("global_id", (None, "", "proxy"))
@pytest.mark.parametrize("global_read", (False, True))
@pytest.mark.parametrize("global_row", (False, True))
def test_end_user_fallback_stages(key_row: bool, global_id: str | None, global_read: bool, global_row: bool) -> None:
    facts: Final = core.EndUserFacts(
        "Default",
        False,
        None,
        True,
        "key",
        key_read=True,
        key_budget=core.BudgetFields() if key_row else None,
        global_config_read=True,
        global_budget_id=global_id,
        global_read=global_read,
        global_budget=core.BudgetFields() if global_row else None,
    )
    expected: Final = (
        "Key"
        if key_row
        else "None"
        if global_id is None
        else "ReadGlobal"
        if not global_read
        else "Global"
        if global_row
        else "None"
    )
    assert core.end_user_source(facts) == expected


@pytest.mark.parametrize("default_id", (None, "", "default"))
@pytest.mark.parametrize("member_id", (None, "", "member", "default"))
@pytest.mark.parametrize("member_models", (None, 0, 2))
@pytest.mark.parametrize("default_models", (None, 0, 2))
@pytest.mark.parametrize("skip", (False, True))
def test_member_snapshot_table(
    default_id: str | None, member_id: str | None, member_models: int | None, default_models: int | None, skip: bool
) -> None:
    facts: Final = core.MemberFacts(
        "",
        "",
        True,
        default_id,
        skip,
        membership_read=True,
        membership=core.MembershipSnapshot(member_id, member_models),
        default_read=True,
        default_budget=core.BudgetFields(model_budget_count=default_models),
    )
    expected: Final = (
        "Member"
        if member_id is not None and member_id != default_id and member_models
        else "Done"
        if default_id is None
        else "Default"
        if default_models is not None
        else "Clear"
    )
    assert core.member_source(facts) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", (None, "", "entity"))
@pytest.mark.parametrize("scope", ("Team", "EndUser"))
@pytest.mark.parametrize("fail", (False, True))
async def test_selected_model_reads_keep_order_and_stop_on_failure(
    native: bool, identity: str | None, scope: Literal["Team", "EndUser"], fail: bool
) -> None:
    failure: Final = RuntimeError("counter failed")
    read: Final = AsyncMock(side_effect=(True, failure, True) if fail else None, return_value=True)
    limiter: Final = Mock()
    limiter.is_team_within_model_budget = read
    limiter.is_end_user_within_model_budget = read
    caps: Final[Mapping[str, object]] = {"a": {"max_budget": 1}}
    token: Final = UserAPIKeyAuth(team_id=identity, team_model_max_budget=dict(caps))
    models: Final = ["a", "a", "b"]
    coroutine: Final = (
        (adapter.check_team_model_budget if native else legacy._check_team_model_budget)(token, limiter, models)  # pyright: ignore[reportPrivateUsage]  # Phase 1 leaf is the unchanged reference implementation
        if scope == "Team"
        else (adapter.check_end_user_model_budget if native else legacy._check_end_user_model_budget)(  # pyright: ignore[reportPrivateUsage]  # Phase 1 leaf is the unchanged reference implementation
            identity, caps, limiter, models, "team"
        )
    )
    active: Final = identity is not None and (scope == "Team" or bool(identity))
    if fail and active:
        with pytest.raises(RuntimeError, match="counter failed") as raised:
            await coroutine
        assert raised.value is failure
    else:
        await coroutine
    assert read.await_count == (2 if fail else 3) if active else read.await_count == 0
    assert tuple(item.kwargs["model"] for item in read.await_args_list) == tuple(models[: read.await_count])


@pytest.mark.asyncio
@pytest.mark.parametrize("own", (False, True))
@pytest.mark.parametrize("own_row", (False, True))
@pytest.mark.parametrize("identity", (None, "", "eu"))
@pytest.mark.parametrize("key_found", (False, True))
async def test_end_user_source_read_order_and_identity(
    native: bool, monkeypatch: pytest.MonkeyPatch, own: bool, own_row: bool, identity: str | None, key_found: bool
) -> None:
    budget: Final = LiteLLM_BudgetTable(model_max_budget={"m": {"max_budget": 1}})
    end_user: Final = (
        LiteLLM_EndUserTable(user_id="eu", blocked=False, litellm_budget_table=budget if own_row else None)
        if own
        else None
    )
    read: Final = AsyncMock(side_effect=(budget,) if key_found else (None, budget))
    monkeypatch.setattr(auth_checks, "get_default_end_user_budget", read)
    monkeypatch.setattr(auth_reference, "get_default_end_user_budget", read)
    monkeypatch.setattr(litellm, "max_end_user_budget_id", "proxy")
    fn: Final = adapter.end_user_model_max_budget if native else legacy._end_user_model_max_budget  # pyright: ignore[reportPrivateUsage]  # Phase 1 leaf is the unchanged reference implementation
    value: Final = await fn(identity, end_user, "key", Mock(), UserApiKeyCache(), None)
    expected: Final = getattr(budget, "model_max_budget", None) if (own and own_row) or (not own and identity) else None
    assert value is expected
    assert read.await_count == (0 if own or not identity else 1 if key_found else 2)
    if read.await_count:
        assert read.await_args_list[0].kwargs["budget_id"] == "key"


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", (False, True))
@pytest.mark.parametrize("present", (False, True))
async def test_token_defaults_preserve_caps_and_projection_identity(native: bool, keep: bool, present: bool) -> None:
    budget: Final = LiteLLM_BudgetTable(
        max_budget=9, tpm_limit=8, rpm_limit=7, tpd_limit=6, model_max_budget={"m": {"max_budget": 5}}
    )
    token: Final = UserAPIKeyAuth(
        end_user_max_budget=0 if present else None,
        end_user_tpm_limit=0 if present else None,
        end_user_model_max_budget={} if present else None,
    )
    original_models: Final = token.end_user_model_max_budget
    own: Final = LiteLLM_EndUserTable(user_id="eu", blocked=False, litellm_budget_table=budget)
    fn: Final = (
        adapter.apply_key_end_user_default_budget_to_token
        if native
        else legacy._apply_key_end_user_default_budget_to_token  # pyright: ignore[reportPrivateUsage]  # Phase 1 leaf is the unchanged reference implementation
    )
    await fn(token, own, "key", Mock(), UserApiKeyCache(), None, keep)
    assert token.end_user_max_budget == (0 if keep and present else 9)
    assert token.end_user_tpm_limit == (0 if keep and present else 8)
    assert (token.end_user_rpm_limit, token.end_user_tpd_limit) == (7, 6)
    assert token.end_user_model_max_budget is (
        original_models if keep and present else getattr(budget, "model_max_budget", None)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("default_id", (None, "", "default"))
@pytest.mark.parametrize("member_id", (None, "default", "override"))
@pytest.mark.parametrize("member_caps", (None, {}, {"member": {"max_budget": 1}}))
@pytest.mark.parametrize("default_caps", (None, {}, {"default": {"max_budget": 2}}))
async def test_member_default_read_and_mutation_order(
    native: bool,
    monkeypatch: pytest.MonkeyPatch,
    default_id: str | None,
    member_id: str | None,
    member_caps: Mapping[str, object] | None,
    default_caps: Mapping[str, object] | None,
) -> None:
    member_budget: Final = LiteLLM_BudgetTable(model_max_budget=dict(member_caps) if member_caps is not None else None)
    default_budget: Final = LiteLLM_BudgetTable(
        model_max_budget=dict(default_caps) if default_caps is not None else None
    )
    membership: Final = LiteLLM_TeamMembership(
        user_id="", team_id="", budget_id=member_id, litellm_budget_table=member_budget
    )
    member_read: Final = AsyncMock(return_value=membership)
    default_read: Final = AsyncMock(return_value=default_budget)
    monkeypatch.setattr(auth_checks, "get_team_membership", member_read)
    monkeypatch.setattr(auth_reference, "get_team_membership", member_read)
    monkeypatch.setattr(legacy, "get_team_membership", member_read)
    monkeypatch.setattr(auth_checks, "get_team_member_default_budget", default_read)
    monkeypatch.setattr(auth_reference, "get_team_member_default_budget", default_read)
    monkeypatch.setattr(legacy, "get_team_member_default_budget", default_read)
    token: Final = UserAPIKeyAuth(user_id="", team_id="", team_member_model_max_budget={"old": {}})
    prior: Final = token.team_member_model_max_budget
    team: Final = LiteLLM_TeamTable(team_id="", metadata={"team_member_budget_id": default_id})
    fn: Final = (
        adapter.load_team_member_default_model_budget if native else legacy._load_team_member_default_model_budget  # pyright: ignore[reportPrivateUsage]  # Phase 1 leaf is the unchanged reference implementation
    )
    await fn(token, team, Mock(), UserApiKeyCache(), False, None)
    override: Final = member_id is not None and member_id != default_id and bool(member_caps)
    expected: Final = (
        getattr(member_budget, "model_max_budget", None)
        if override
        else getattr(default_budget, "model_max_budget", None)
        if default_id is not None
        else prior
    )
    assert token.team_member_model_max_budget is expected
    member_read.assert_awaited_once()
    assert default_read.await_count == (0 if override or default_id is None else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("skip", (False, True))
@pytest.mark.parametrize("during_default", (False, True))
async def test_member_lookup_error_policy(
    native: bool, monkeypatch: pytest.MonkeyPatch, skip: bool, during_default: bool
) -> None:
    error: Final = RuntimeError("budget read failed")
    member_read: Final = AsyncMock(return_value=None, side_effect=None if during_default else error)
    default_read: Final = AsyncMock(side_effect=error)
    monkeypatch.setattr(auth_checks, "get_team_membership", member_read)
    monkeypatch.setattr(auth_reference, "get_team_membership", member_read)
    monkeypatch.setattr(legacy, "get_team_membership", member_read)
    monkeypatch.setattr(auth_checks, "get_team_member_default_budget", default_read)
    monkeypatch.setattr(auth_reference, "get_team_member_default_budget", default_read)
    monkeypatch.setattr(legacy, "get_team_member_default_budget", default_read)
    token: Final = UserAPIKeyAuth(user_id="u", team_id="t", team_member_model_max_budget={"old": {}})
    prior: Final = token.team_member_model_max_budget
    team: Final = LiteLLM_TeamTable(team_id="t", metadata={"team_member_budget_id": "default"})
    fn: Final = (
        adapter.load_team_member_default_model_budget if native else legacy._load_team_member_default_model_budget  # pyright: ignore[reportPrivateUsage]  # Phase 1 leaf is the unchanged reference implementation
    )
    pending: Final = fn(token, team, Mock(), UserApiKeyCache(), skip, None)
    if skip:
        await pending
    else:
        with pytest.raises(RuntimeError, match="budget read failed") as raised:
            await pending
        assert raised.value is error
    assert token.team_member_model_max_budget is prior


@pytest.mark.asyncio
async def test_member_native_failure_never_becomes_skippable_lookup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    error: Final = NativeAuthRuntimeError("native planner failed")
    read: Final = AsyncMock(side_effect=error)
    monkeypatch.setattr(auth_checks, "get_team_membership", read)
    monkeypatch.setattr(auth_reference, "get_team_membership", read)
    with pytest.raises(NativeAuthRuntimeError, match="native planner failed") as raised:
        await adapter.load_team_member_default_model_budget(
            UserAPIKeyAuth(user_id="u", team_id="t"), None, Mock(), UserApiKeyCache(), True, None
        )
    assert raised.value is error
    read.assert_awaited_once()


@pytest.mark.asyncio
async def test_member_planner_failure_stops_before_io(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.rust_bridge import _native

    native: Final = Mock(side_effect=RuntimeError("decode failed"))
    read: Final = AsyncMock()
    monkeypatch.setattr(_native, "request_auth_context_member", native)
    monkeypatch.setattr(auth_checks, "get_team_membership", read)
    monkeypatch.setattr(auth_reference, "get_team_membership", read)
    with pytest.raises(NativeAuthRuntimeError, match="decode failed"):
        await adapter.load_team_member_default_model_budget(
            UserAPIKeyAuth(user_id="u", team_id="t"), None, Mock(), UserApiKeyCache(), True, None
        )
    read.assert_not_awaited()
    native.assert_called_once()
