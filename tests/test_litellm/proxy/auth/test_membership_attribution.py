"""Tests for membership-based usage attribution.

Covers the two opt-in settings introduced alongside
``litellm/proxy/auth/membership_attribution.py``:

- ``track_spend_across_all_user_teams``
- ``enforce_rate_limits_across_all_user_teams``

The most important cases here are the OFF cases. Both settings default to off,
and every one of those tests is a regression guard proving the default path is
byte-for-byte what it was before the feature existed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy._types import Litellm_EntityType, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import (
    _attributed_orgs_max_budget_check,
    _attributed_teams_max_budget_check,
)
from litellm.proxy.auth.membership_attribution import (
    attributed_org_ids,
    attributed_team_ids,
    attribution_targets,
    rate_limit_attribution_enabled,
    resolve_membership_attribution,
    spend_attribution_enabled,
)
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup


# --------------------------------------------------------------------------- #
# settings gates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("general_settings", [None, {}, {"track_spend_across_all_user_teams": False}])
def test_spend_attribution_defaults_off(general_settings):
    assert spend_attribution_enabled(general_settings) is False


@pytest.mark.parametrize("general_settings", [None, {}, {"enforce_rate_limits_across_all_user_teams": False}])
def test_rate_limit_attribution_defaults_off(general_settings):
    assert rate_limit_attribution_enabled(general_settings) is False


def test_settings_are_independent():
    """An operator must be able to take spend attribution without rate limits."""
    spend_only = {"track_spend_across_all_user_teams": True}
    assert spend_attribution_enabled(spend_only) is True
    assert rate_limit_attribution_enabled(spend_only) is False


# --------------------------------------------------------------------------- #
# target resolution
# --------------------------------------------------------------------------- #
def test_attribution_targets_falls_back_to_stamped_id():
    assert attribution_targets(None, "team-a") == ("team-a",)
    assert attribution_targets([], "team-a") == ("team-a",)
    assert attribution_targets(None, None) == ()


def test_attribution_targets_dedupes_and_preserves_order():
    """The stamped team is normally also in the membership list; charge it once."""
    assert attribution_targets(["team-a", "team-b", "team-a", ""], "team-a") == ("team-a", "team-b")


def test_read_helpers_fall_back_to_stamped_values():
    token = UserAPIKeyAuth(api_key="sk-1", team_id="team-a", org_id="org-1")
    assert attributed_team_ids(token) == ("team-a",)
    assert attributed_org_ids(token) == ("org-1",)


def test_attributed_fields_cannot_be_forged_from_input():
    """Server-only fields. A caller who could set these would pick their own
    budget and rate-limit buckets."""
    forged = UserAPIKeyAuth(
        api_key="sk-1",
        team_id="team-a",
        attributed_team_ids=["team-with-huge-budget"],
        attributed_org_ids=["org-with-huge-budget"],
        attributed_team_limits={"team-with-huge-budget": {"rpm": 10**9}},
    )
    assert forged.attributed_team_ids is None
    assert forged.attributed_org_ids is None
    assert forged.attributed_team_limits is None


# --------------------------------------------------------------------------- #
# resolver
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_resolver_is_noop_when_both_settings_off():
    """Not even a cache read on the default path."""
    token = UserAPIKeyAuth(api_key="sk-1", team_id="team-a")
    with patch("litellm.proxy.auth.auth_checks.get_team_object", new=AsyncMock()) as mock_get_team:
        await resolve_membership_attribution(
            user_api_key_auth_obj=token,
            user_object=LiteLLM_UserTable(user_id="u1", teams=["team-a", "team-b"]),
            general_settings={},
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
        )
    mock_get_team.assert_not_called()
    assert token.attributed_team_ids is None


@pytest.mark.asyncio
async def test_resolver_collects_every_membership_with_stamped_team_first():
    token = UserAPIKeyAuth(api_key="sk-1", team_id="team-a")

    def _team(team_id: str, rpm=None, tpm=None, org="org-1"):
        obj = MagicMock()
        obj.team_id = team_id
        obj.rpm_limit = rpm
        obj.tpm_limit = tpm
        obj.organization_id = org
        return obj

    async def _fake_get_team_object(team_id, **kwargs):
        return _team(team_id, rpm=10 if team_id == "team-b" else None)

    with patch("litellm.proxy.auth.auth_checks.get_team_object", new=AsyncMock(side_effect=_fake_get_team_object)):
        await resolve_membership_attribution(
            user_api_key_auth_obj=token,
            user_object=LiteLLM_UserTable(user_id="u1", teams=["team-b", "team-a", "team-c"]),
            general_settings={"track_spend_across_all_user_teams": True},
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
        )

    # stamped team leads, membership list follows, no duplicate of team-a
    assert token.attributed_team_ids == ("team-a", "team-b", "team-c")
    assert token.attributed_team_limits["team-b"] == {"rpm": 10, "tpm": None}
    assert token.attributed_org_ids == ("org-1",)


@pytest.mark.asyncio
async def test_resolver_fails_open_on_unresolvable_team():
    """A deleted or erroring team is skipped, never raised: attribution must not
    turn an already-authorized request into a 500."""
    token = UserAPIKeyAuth(api_key="sk-1", team_id="team-a")

    async def _fake_get_team_object(team_id, **kwargs):
        if team_id == "team-broken":
            raise Exception("team row is gone")
        obj = MagicMock()
        obj.rpm_limit = None
        obj.tpm_limit = None
        obj.organization_id = "org-1"
        return obj

    with patch("litellm.proxy.auth.auth_checks.get_team_object", new=AsyncMock(side_effect=_fake_get_team_object)):
        await resolve_membership_attribution(
            user_api_key_auth_obj=token,
            user_object=LiteLLM_UserTable(user_id="u1", teams=["team-broken", "team-a"]),
            general_settings={"track_spend_across_all_user_teams": True},
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
        )

    assert token.attributed_team_ids == ("team-a",)


# --------------------------------------------------------------------------- #
# spend fan-out
# --------------------------------------------------------------------------- #
def _enqueued(mock_add_update, entity_type):
    return [
        c.kwargs["update"]["entity_id"]
        for c in mock_add_update.call_args_list
        if c.kwargs["update"]["entity_type"] == entity_type
    ]


@pytest.mark.asyncio
async def test_update_team_db_single_team_when_attribution_off():
    """Regression guard for the default path."""
    writer = DBSpendUpdateWriter()
    writer.spend_update_queue.add_update = AsyncMock()

    await writer._update_team_db(
        response_cost=1.0,
        team_id="team-a",
        user_id="u1",
        prisma_client=MagicMock(),
    )

    assert _enqueued(writer.spend_update_queue.add_update, Litellm_EntityType.TEAM) == ["team-a"]
    assert _enqueued(writer.spend_update_queue.add_update, Litellm_EntityType.TEAM_MEMBER) == [
        "team_id::team-a::user_id::u1"
    ]


@pytest.mark.asyncio
async def test_update_team_db_charges_every_attributed_team():
    writer = DBSpendUpdateWriter()
    writer.spend_update_queue.add_update = AsyncMock()

    await writer._update_team_db(
        response_cost=1.0,
        team_id="team-a",
        user_id="u1",
        prisma_client=MagicMock(),
        attributed_team_ids=["team-a", "team-b", "team-c"],
    )

    assert _enqueued(writer.spend_update_queue.add_update, Litellm_EntityType.TEAM) == [
        "team-a",
        "team-b",
        "team-c",
    ]
    assert _enqueued(writer.spend_update_queue.add_update, Litellm_EntityType.TEAM_MEMBER) == [
        "team_id::team-a::user_id::u1",
        "team_id::team-b::user_id::u1",
        "team_id::team-c::user_id::u1",
    ]


@pytest.mark.asyncio
async def test_update_org_db_charges_every_attributed_org():
    writer = DBSpendUpdateWriter()
    writer.spend_update_queue.add_update = AsyncMock()

    await writer._update_org_db(
        response_cost=1.0,
        org_id="org-1",
        prisma_client=MagicMock(),
        attributed_org_ids=["org-1", "org-2"],
    )

    assert _enqueued(writer.spend_update_queue.add_update, Litellm_EntityType.ORGANIZATION) == [
        "org-1",
        "org-2",
    ]


@pytest.mark.asyncio
async def test_daily_team_rollup_emits_one_row_per_team():
    """LiteLLM_DailyTeamSpend is already unique per team+date+key+model, so N
    teams means N distinct transaction keys and no migration."""
    writer = DBSpendUpdateWriter()
    writer.daily_team_spend_update_queue.add_update = AsyncMock()
    writer._common_add_spend_log_transaction_to_daily_transaction = AsyncMock(
        return_value={"date": "2026-08-21", "spend": 1.0, "api_requests": 1, "endpoint": "/chat/completions"}
    )

    payload = {
        "team_id": "team-a",
        "api_key": "hashed-key",
        "model": "gpt-4o",
        "custom_llm_provider": "openai",
    }

    await writer.add_spend_log_transaction_to_daily_team_transaction(
        payload=payload,
        prisma_client=MagicMock(),
        attributed_team_ids=["team-a", "team-b"],
    )

    keys = [list(c.kwargs["update"].keys())[0] for c in writer.daily_team_spend_update_queue.add_update.call_args_list]
    assert len(keys) == 2
    assert keys[0].startswith("team-a_2026-08-21_")
    assert keys[1].startswith("team-b_2026-08-21_")

    team_ids = [
        list(c.kwargs["update"].values())[0]["team_id"]
        for c in writer.daily_team_spend_update_queue.add_update.call_args_list
    ]
    assert team_ids == ["team-a", "team-b"]


@pytest.mark.asyncio
async def test_daily_team_rollup_single_row_when_attribution_off():
    writer = DBSpendUpdateWriter()
    writer.daily_team_spend_update_queue.add_update = AsyncMock()
    writer._common_add_spend_log_transaction_to_daily_transaction = AsyncMock(
        return_value={"date": "2026-08-21", "spend": 1.0, "api_requests": 1, "endpoint": ""}
    )

    await writer.add_spend_log_transaction_to_daily_team_transaction(
        payload={
            "team_id": "team-a",
            "api_key": "hashed-key",
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
        },
        prisma_client=MagicMock(),
    )

    assert writer.daily_team_spend_update_queue.add_update.call_count == 1


# --------------------------------------------------------------------------- #
# rate-limit fan-out
# --------------------------------------------------------------------------- #
def _limiter():
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3,
    )

    return _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=MagicMock())


def test_rate_limit_descriptors_not_added_when_setting_off():
    limiter = _limiter()
    token = UserAPIKeyAuth(api_key="sk-1", team_id="team-a")
    token.attributed_team_limits = {"team-b": {"rpm": 10, "tpm": None}}

    with patch("litellm.proxy.proxy_server.general_settings", {}):
        descriptors = limiter._attributed_team_rate_limit_descriptors(user_api_key_dict=token)
    assert descriptors == ()


def test_rate_limit_descriptors_added_for_other_teams_only():
    limiter = _limiter()
    token = UserAPIKeyAuth(api_key="sk-1", team_id="team-a")
    token.attributed_team_limits = {
        "team-a": {"rpm": 5, "tpm": None},  # stamped team: already emitted elsewhere
        "team-b": {"rpm": 10, "tpm": None},
        "team-c": {"rpm": None, "tpm": None},  # no limits: nothing to enforce
        "team-d": {"rpm": None, "tpm": 900},
    }

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"enforce_rate_limits_across_all_user_teams": True},
    ):
        descriptors = limiter._attributed_team_rate_limit_descriptors(user_api_key_dict=token)

    assert [d["value"] for d in descriptors] == ["team-b", "team-d"]
    # same namespace as the stamped-team descriptor, so a team shares one bucket
    assert {d["key"] for d in descriptors} == {"team"}
    assert descriptors[0]["rate_limit"]["requests_per_unit"] == 10
    assert descriptors[1]["rate_limit"]["tokens_per_unit"] == 900


# --------------------------------------------------------------------------- #
# budget enforcement across memberships
# --------------------------------------------------------------------------- #
SPEND_ON = {"track_spend_across_all_user_teams": True}


def _proxy_logging():
    from litellm.proxy.utils import ProxyLogging

    obj = ProxyLogging(user_api_key_cache=None)
    obj.budget_alerts = AsyncMock()
    return obj


def _team(team_id, max_budget=None, spend=0.0, org="org-1"):
    obj = MagicMock()
    obj.team_id = team_id
    obj.team_alias = f"{team_id}-alias"
    obj.max_budget = max_budget
    obj.spend = spend
    obj.organization_id = org
    return obj


def _token_with_memberships():
    token = UserAPIKeyAuth(api_key="sk-1", team_id="team-a", org_id="org-1")
    token.attributed_team_ids = ["team-a", "team-b"]
    token.attributed_org_ids = ["org-1", "org-2"]
    return token


async def _team_budget_error(token, general_settings):
    """The budget error the team gate raised, or None if the caller got through.

    Asserting on this instead of on which internals ran keeps these tests about
    the only thing a caller can observe: blocked, or not blocked.
    """
    try:
        await _attributed_teams_max_budget_check(
            valid_token=token,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
            proxy_logging_obj=_proxy_logging(),
            general_settings=general_settings,
        )
    except litellm.BudgetExceededError as e:
        return e
    return None


async def _org_budget_error(token, general_settings):
    """The budget error the org gate raised, or None if the caller got through."""
    try:
        await _attributed_orgs_max_budget_check(
            valid_token=token,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
            proxy_logging_obj=_proxy_logging(),
            general_settings=general_settings,
        )
    except litellm.BudgetExceededError as e:
        return e
    return None


def _over_budget_team(team_id, **kwargs):
    """A team whose budget is already blown, for proving a gate is inert."""
    return _team(team_id, max_budget=1.0, spend=999.0)


async def _spend_always_over(counter_key, fallback_spend, max_budget=None, **kwargs):
    return 999.0


@pytest.mark.asyncio
async def test_attributed_team_budget_check_noop_when_setting_off():
    """Populated memberships must not gate budgets unless the SPEND setting is on.

    Enabling only rate-limit attribution must not silently add budget gates, so
    a caller whose other team is wildly over budget still gets through.
    """
    token = _token_with_memberships()
    with (
        patch(
            "litellm.proxy.auth.auth_checks.get_team_object",
            new=AsyncMock(side_effect=lambda team_id, **kwargs: _over_budget_team(team_id)),
        ),
        patch("litellm.proxy.proxy_server.get_current_spend", _spend_always_over),
    ):
        error = await _team_budget_error(token, {"enforce_rate_limits_across_all_user_teams": True})

    assert error is None


@pytest.mark.asyncio
async def test_attributed_team_budget_check_raises_for_other_team():
    """A team the caller merely belongs to can now block them."""
    token = _token_with_memberships()

    async def _get_team(team_id, **kwargs):
        return _team(team_id, max_budget=10.0)

    async def _spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        return 25.0 if counter_key == "spend:team:team-b" else 0.0

    with (
        patch("litellm.proxy.auth.auth_checks.get_team_object", new=AsyncMock(side_effect=_get_team)),
        patch("litellm.proxy.proxy_server.get_current_spend", _spend),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc:
            await _attributed_teams_max_budget_check(
                valid_token=token,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=_proxy_logging(),
                general_settings=SPEND_ON,
            )

    assert exc.value.entity_id == "team-b"
    assert exc.value.entity_type == Litellm_EntityType.TEAM.value
    assert exc.value.current_cost == 25.0


@pytest.mark.asyncio
async def test_attributed_team_budget_check_skips_stamped_team():
    """The stamped team is _team_max_budget_check's job. Checking it here too
    would raise twice and fire a duplicate budget alert."""
    token = _token_with_memberships()
    seen = []

    async def _get_team(team_id, **kwargs):
        seen.append(team_id)
        return _team(team_id, max_budget=100.0)

    async def _spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        return 0.0

    with (
        patch("litellm.proxy.auth.auth_checks.get_team_object", new=AsyncMock(side_effect=_get_team)),
        patch("litellm.proxy.proxy_server.get_current_spend", _spend),
    ):
        await _attributed_teams_max_budget_check(
            valid_token=token,
            prisma_client=MagicMock(),
            user_api_key_cache=MagicMock(),
            proxy_logging_obj=_proxy_logging(),
            general_settings=SPEND_ON,
        )

    assert seen == ["team-b"]


@pytest.mark.asyncio
async def test_attributed_team_budget_check_passes_under_budget():
    """A membership with room left does not block the caller."""
    token = _token_with_memberships()

    async def _get_team(team_id, **kwargs):
        return _team(team_id, max_budget=100.0)

    async def _spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        return 5.0

    with (
        patch("litellm.proxy.auth.auth_checks.get_team_object", new=AsyncMock(side_effect=_get_team)),
        patch("litellm.proxy.proxy_server.get_current_spend", _spend),
    ):
        error = await _team_budget_error(token, SPEND_ON)

    assert error is None


@pytest.mark.asyncio
async def test_attributed_team_budget_check_ignores_unloadable_team():
    """A team that cannot be loaded contributes no ceiling rather than a 500.

    The caller is neither blocked nor served an internal error.
    """
    token = _token_with_memberships()

    async def _get_team(team_id, **kwargs):
        raise Exception("team row is gone")

    with patch("litellm.proxy.auth.auth_checks.get_team_object", new=AsyncMock(side_effect=_get_team)):
        error = await _team_budget_error(token, SPEND_ON)

    assert error is None


@pytest.mark.asyncio
async def test_attributed_org_budget_check_raises_for_other_org():
    """Only reachable when the caller's teams span several organizations."""
    token = _token_with_memberships()

    org = MagicMock()
    org.spend = 0.0
    org.litellm_budget_table = MagicMock()
    org.litellm_budget_table.max_budget = 50.0

    async def _spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        return 90.0 if counter_key == "spend:org:org-2" else 0.0

    with (
        patch("litellm.proxy.auth.auth_checks.get_org_object", new=AsyncMock(return_value=org)),
        patch("litellm.proxy.proxy_server.get_current_spend", _spend),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc:
            await _attributed_orgs_max_budget_check(
                valid_token=token,
                prisma_client=MagicMock(),
                user_api_key_cache=MagicMock(),
                proxy_logging_obj=_proxy_logging(),
                general_settings=SPEND_ON,
            )

    assert exc.value.entity_id == "org-2"
    assert exc.value.entity_type == Litellm_EntityType.ORGANIZATION.value


@pytest.mark.asyncio
async def test_attributed_org_budget_check_noop_when_setting_off():
    """An over-budget non-stamped org does not block while the setting is off."""
    token = _token_with_memberships()
    org = MagicMock()
    org.spend = 999.0
    org.litellm_budget_table = MagicMock()
    org.litellm_budget_table.max_budget = 1.0

    with (
        patch("litellm.proxy.auth.auth_checks.get_org_object", new=AsyncMock(return_value=org)),
        patch("litellm.proxy.proxy_server.get_current_spend", _spend_always_over),
    ):
        error = await _org_budget_error(token, {})

    assert error is None


# --------------------------------------------------------------------------- #
# request-metadata stamping
# --------------------------------------------------------------------------- #
def test_metadata_stamped_only_when_setting_on():
    from litellm.proxy.hooks.proxy_track_cost_callback import _metadata_id_list

    token = _token_with_memberships()

    data_off = {"metadata": {}}
    with patch("litellm.proxy.proxy_server.general_settings", {}):
        LiteLLMProxyRequestSetup._add_attributed_membership_metadata(
            data=data_off, user_api_key_dict=token, _metadata_variable_name="metadata"
        )
    assert data_off["metadata"] == {}

    data_on = {"metadata": {}}
    with patch("litellm.proxy.proxy_server.general_settings", SPEND_ON):
        LiteLLMProxyRequestSetup._add_attributed_membership_metadata(
            data=data_on, user_api_key_dict=token, _metadata_variable_name="metadata"
        )
    assert data_on["metadata"]["user_api_key_attributed_team_ids"] == ("team-a", "team-b")
    assert data_on["metadata"]["user_api_key_attributed_org_ids"] == ("org-1", "org-2")

    # and the cost callback reads back exactly what was stamped
    assert _metadata_id_list(data_on["metadata"], "user_api_key_attributed_team_ids") == ("team-a", "team-b")


def test_metadata_id_list_returns_none_when_absent():
    """None, not [], so a writer can tell "attribution off" from "on, nothing
    resolved"."""
    from litellm.proxy.hooks.proxy_track_cost_callback import _metadata_id_list

    assert _metadata_id_list({}, "user_api_key_attributed_team_ids") is None
    assert _metadata_id_list({"user_api_key_attributed_team_ids": []}, "user_api_key_attributed_team_ids") is None
    assert _metadata_id_list({"user_api_key_attributed_team_ids": "nope"}, "user_api_key_attributed_team_ids") is None
