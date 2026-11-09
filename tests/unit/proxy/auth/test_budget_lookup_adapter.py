from collections.abc import Mapping
from datetime import datetime, timezone
from math import inf, isnan, nan
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import TypeAdapter

from litellm._auth.failure import NativeAuthRuntimeError
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_OrganizationTable, LiteLLM_TeamMembership, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth import auth_checks, common_budget_adapter, common_budget_lookups
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging

from . import phase1_budget_lookup_reference as reference
from .test_common_budget_adapter import observe

_NOW: Final = datetime(2025, 1, 1, tzinfo=timezone.utc)
_CALL: Final = TypeAdapter(tuple[str, tuple[object, ...], Mapping[str, object]])


def normalized_calls(events: Mock) -> tuple[tuple[str, tuple[object, ...], Mapping[str, object]], ...]:
    return tuple(
        (
            name,
            args,
            MappingProxyType(
                {key: "NaN" if isinstance(value, float) and isnan(value) else value for key, value in kwargs.items()}
            ),
        )
        for name, args, kwargs in (_CALL.validate_python(value) for value in events.mock_calls)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("token_id", "team_id"), ((None, None), (None, "team-org"), ("", "team-org"), ("key-org", "team-org"))
)
@pytest.mark.parametrize(("token_present", "client_present"), ((True, True), (False, True), (True, False)))
@pytest.mark.parametrize("row_state", ("present", "missing", "failed"))
@pytest.mark.parametrize("limit", (None, 1.0, inf))
async def test_organization_source_and_raw_reads_match_phase1(
    monkeypatch: pytest.MonkeyPatch,
    token_id: str | None,
    team_id: str | None,
    token_present: bool,
    client_present: bool,
    row_state: str,
    limit: float | None,
) -> None:
    row: Final = LiteLLM_OrganizationTable.model_construct(
        organization_id="row", spend=2.0, budget_id="budget", litellm_budget_table=LiteLLM_BudgetTable(max_budget=limit)
    )
    reader: Final = AsyncMock(
        return_value=row if row_state == "present" else None,
        side_effect=RuntimeError("unavailable") if row_state == "failed" else None,
    )
    counter: Final = AsyncMock(return_value=2.0)
    monkeypatch.setattr(reference, "get_org_object", reader)
    monkeypatch.setattr(auth_checks, "get_org_object", reader)
    monkeypatch.setattr(proxy_server, "get_current_spend", counter)
    old: Final = UserAPIKeyAuth(org_id=token_id, token="token", key_alias="alias")
    new: Final = old.model_copy(deep=True)
    team: Final = LiteLLM_TeamTable(team_id="team", organization_id=team_id)
    prisma: Final = Mock(spec=PrismaClient) if client_present else None
    cache: Final = UserApiKeyCache()
    alerts: Final = AsyncMock()
    old_log: Final = Mock(spec=ProxyLogging, budget_alerts=alerts)
    new_log: Final = old_log
    before: Final = await observe(
        lambda: reference.organization_max_budget_check(old if token_present else None, team, prisma, cache, old_log),
        old_log,
        old,
    )
    old_reads: Final = (tuple(reader.await_args_list), tuple(counter.await_args_list))
    reader.reset_mock()
    counter.reset_mock()
    alerts.reset_mock()
    after: Final = await observe(
        lambda: common_budget_lookups.organization_max_budget_check(
            new if token_present else None, team, prisma, cache, new_log
        ),
        new_log,
        new,
    )
    assert after == before
    assert (tuple(reader.await_args_list), tuple(counter.await_args_list)) == old_reads
    assert new.model_dump() == old.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("loaded", (False, True))
@pytest.mark.parametrize("member_limit", (None, 0.0, 1.0, inf, nan))
@pytest.mark.parametrize("default_id", (None, "", "default", 1))
@pytest.mark.parametrize("missing_member", (False, True))
async def test_member_default_and_counter_order_match_phase1(
    monkeypatch: pytest.MonkeyPatch,
    loaded: bool,
    member_limit: float | None,
    default_id: object,
    missing_member: bool,
) -> None:
    row: Final = LiteLLM_TeamMembership(
        team_id="team", user_id="user", spend=2.0, litellm_budget_table=LiteLLM_BudgetTable(max_budget=member_limit)
    )
    membership: Final = None if missing_member else row
    member_read: Final = AsyncMock(return_value=membership)
    default_read: Final = AsyncMock(return_value=LiteLLM_BudgetTable(max_budget=1))
    counter: Final = AsyncMock(return_value=2.0)
    clock: Final = Mock(return_value=_NOW)
    events: Final = Mock()
    events.attach_mock(member_read, "membership")
    events.attach_mock(default_read, "default")
    events.attach_mock(counter, "counter")
    events.attach_mock(clock, "clock")
    for module in (reference, auth_checks):
        monkeypatch.setattr(module, "get_team_membership", member_read)
        monkeypatch.setattr(module, "get_team_member_default_budget", default_read)
        monkeypatch.setattr(module, "get_utc_datetime", clock)
    monkeypatch.setattr(proxy_server, "get_current_spend", counter)
    team: Final = LiteLLM_TeamTable(
        team_id="team",
        metadata={
            "team_member_budget_id": default_id,
            "team_member_max_budget_alert_emails": {"50": ["owner"]},
        },
    )
    key: Final = UserAPIKeyAuth(user_id="user")
    prisma: Final = Mock(spec=PrismaClient)
    cache: Final = UserApiKeyCache()
    alerts: Final = AsyncMock()
    old_log: Final = Mock(spec=ProxyLogging, budget_alerts=alerts)
    new_log: Final = old_log
    before: Final = await observe(
        lambda: reference._check_team_member_budget(team, None, key, prisma, cache, old_log, membership, loaded),  # pyright: ignore[reportPrivateUsage]  # exact Phase 1 reference
        old_log,
        key,
    )
    old_reads: Final = normalized_calls(events)
    events.reset_mock()
    alerts.reset_mock()
    after: Final = await observe(
        lambda: common_budget_lookups.check_team_member_budget(
            team, None, key, prisma, cache, new_log, membership, loaded
        ),
        new_log,
        key,
    )
    assert after == before
    assert normalized_calls(events) == old_reads


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw", (None, "", " a,b , ,a", "\x1ca\x1f,\u0085b\u00a0", [" a ", "", "\x1c", 4, "a"], ("a",), 4, {"a": 1})
)
@pytest.mark.parametrize(
    ("limit", "spend", "token"),
    ((1, 2, True), (1, 0, True), (0, 0, True), (1, 2, False), (None, 2, True), (nan, 2, True)),
)
async def test_soft_email_normalization_matches_phase1(
    raw: object,
    limit: float | None,
    spend: float,
    token: bool,
) -> None:
    team: Final = LiteLLM_TeamTable(
        team_id="team", soft_budget=limit, spend=spend, metadata={"soft_budget_alerting_emails": raw}
    )
    key: Final = UserAPIKeyAuth()
    old_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    new_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    before: Final = await observe(
        lambda: reference._team_soft_budget_check(team, key if token else None, old_log),  # pyright: ignore[reportPrivateUsage]  # exact Phase 1 reference
        old_log,
        key,
    )
    after: Final = await observe(
        lambda: common_budget_adapter.team_soft_budget_check(team, key if token else None, new_log),
        new_log,
        key,
    )
    assert after == before


@pytest.mark.asyncio
async def test_organization_native_failure_from_lookup_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = NativeAuthRuntimeError("native failed")
    reader: Final = AsyncMock(side_effect=failure)
    monkeypatch.setattr(auth_checks, "get_org_object", reader)
    with pytest.raises(NativeAuthRuntimeError) as raised:
        await common_budget_lookups.organization_max_budget_check(
            UserAPIKeyAuth(org_id="org"), None, Mock(spec=PrismaClient), UserApiKeyCache(), Mock(spec=ProxyLogging)
        )
    assert raised.value is failure
    assert reader.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("team_present", "team_id", "token_present", "user_id"),
    (
        (False, "team", True, "user"),
        (True, None, True, "user"),
        (True, "team", False, "user"),
        (True, "team", True, None),
        (True, "", True, ""),
    ),
)
async def test_member_identity_stages_match_phase1(
    monkeypatch: pytest.MonkeyPatch,
    team_present: bool,
    team_id: str | None,
    token_present: bool,
    user_id: str | None,
) -> None:
    team: Final = LiteLLM_TeamTable.model_construct(team_id=team_id) if team_present else None
    key: Final = UserAPIKeyAuth(user_id=user_id)
    reader: Final = AsyncMock(return_value=None)
    clock: Final = Mock(return_value=_NOW)
    for module in (reference, auth_checks):
        monkeypatch.setattr(module, "get_team_membership", reader)
        monkeypatch.setattr(module, "get_utc_datetime", clock)
    logging: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    cache: Final = UserApiKeyCache()
    before: Final = await observe(
        lambda: reference._check_team_member_budget(team, None, key if token_present else None, None, cache, logging),  # pyright: ignore[reportPrivateUsage]  # exact Phase 1 reference
        logging,
        key,
    )
    reads: Final = (tuple(reader.await_args_list), clock.call_count)
    reader.reset_mock()
    clock.reset_mock()
    after: Final = await observe(
        lambda: common_budget_lookups.check_team_member_budget(
            team, None, key if token_present else None, None, cache, logging
        ),
        logging,
        key,
    )
    assert after == before
    assert (tuple(reader.await_args_list), clock.call_count) == reads


@pytest.mark.asyncio
@pytest.mark.parametrize("native", (False, True))
async def test_soft_budget_skip_does_not_read_metadata(native: bool) -> None:
    team: Final = LiteLLM_TeamTable(team_id="team", soft_budget=10, spend=0)
    object.__setattr__(team, "metadata", object())
    alerts: Final = AsyncMock()
    logging: Final = Mock(spec=ProxyLogging, budget_alerts=alerts)
    policy: Final = common_budget_adapter.team_soft_budget_check if native else reference._team_soft_budget_check  # pyright: ignore[reportPrivateUsage]  # exact Phase 1 reference
    result: Final = await policy(team, UserAPIKeyAuth(), logging)
    assert result is None
    alerts.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", (None, [], 1, "metadata", {1: "ignored", "soft_budget_alerting_emails": " a "}))
async def test_soft_metadata_shape_preserves_phase1_projection(raw: object) -> None:
    team: Final = LiteLLM_TeamTable(team_id="team", soft_budget=1, spend=2)
    object.__setattr__(team, "metadata", raw)
    key: Final = UserAPIKeyAuth()
    old_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    new_log: Final = Mock(spec=ProxyLogging, budget_alerts=AsyncMock())
    before: Final = await observe(
        lambda: reference._team_soft_budget_check(team, key, old_log),  # pyright: ignore[reportPrivateUsage]  # exact Phase 1 reference
        old_log,
        key,
    )
    after: Final = await observe(lambda: common_budget_adapter.team_soft_budget_check(team, key, new_log), new_log, key)
    assert after == before
