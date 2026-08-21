"""Tests for membership-based usage attribution.

Covers the two opt-in settings introduced alongside
``litellm/proxy/auth/membership_attribution.py``:

- ``track_spend_across_all_user_teams``
- ``enforce_rate_limits_across_all_user_teams``

The most important cases here are the OFF cases. Both settings default to off,
and every one of those tests is a regression guard proving the default path is
byte-for-byte what it was before the feature existed.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import Litellm_EntityType, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.membership_attribution import (
    attributed_org_ids,
    attributed_team_ids,
    attribution_targets,
    rate_limit_attribution_enabled,
    resolve_membership_attribution,
    spend_attribution_enabled,
)
from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter


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
    assert attribution_targets(None, "team-a") == ["team-a"]
    assert attribution_targets([], "team-a") == ["team-a"]
    assert attribution_targets(None, None) == []


def test_attribution_targets_dedupes_and_preserves_order():
    """The stamped team is normally also in the membership list; charge it once."""
    assert attribution_targets(["team-a", "team-b", "team-a", ""], "team-a") == ["team-a", "team-b"]


def test_read_helpers_fall_back_to_stamped_values():
    token = UserAPIKeyAuth(api_key="sk-1", team_id="team-a", org_id="org-1")
    assert attributed_team_ids(token) == ["team-a"]
    assert attributed_org_ids(token) == ["org-1"]


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
    assert token.attributed_team_ids == ["team-a", "team-b", "team-c"]
    assert token.attributed_team_limits["team-b"] == {"rpm": 10, "tpm": None}
    assert token.attributed_org_ids == ["org-1"]


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

    assert token.attributed_team_ids == ["team-a"]


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

    descriptors: list = []
    with patch("litellm.proxy.proxy_server.general_settings", {}):
        limiter._add_attributed_team_rate_limit_descriptors(user_api_key_dict=token, descriptors=descriptors)
    assert descriptors == []


def test_rate_limit_descriptors_added_for_other_teams_only():
    limiter = _limiter()
    token = UserAPIKeyAuth(api_key="sk-1", team_id="team-a")
    token.attributed_team_limits = {
        "team-a": {"rpm": 5, "tpm": None},  # stamped team: already emitted elsewhere
        "team-b": {"rpm": 10, "tpm": None},
        "team-c": {"rpm": None, "tpm": None},  # no limits: nothing to enforce
        "team-d": {"rpm": None, "tpm": 900},
    }

    descriptors: list = []
    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"enforce_rate_limits_across_all_user_teams": True},
    ):
        limiter._add_attributed_team_rate_limit_descriptors(user_api_key_dict=token, descriptors=descriptors)

    assert [d["value"] for d in descriptors] == ["team-b", "team-d"]
    # same namespace as the stamped-team descriptor, so a team shares one bucket
    assert {d["key"] for d in descriptors} == {"team"}
    assert descriptors[0]["rate_limit"]["requests_per_unit"] == 10
    assert descriptors[1]["rate_limit"]["tokens_per_unit"] == 900
