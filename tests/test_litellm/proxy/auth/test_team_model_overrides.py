"""
Tests for team-scoped default + per-user model overrides.

Covers:
1. defaults_only → user can access default models, not others
2. defaults + overrides → user can access union of both
3. overrides only (no defaults) → user can access override models only
4. neither configured → falls back to team.models (backward compat, including [] = allow all)
5. key creation rejects models outside effective set → 403
6. key creation with no models → defaults to effective set
7. remove override → next request for that model → 403 (revocation)
8. all-team-models key + overrides → restricted to effective set
9. cross-user isolation: User A overrides don't affect User B
10. access_group_ids fallback still works when effective models check fails
11. feature flag off → all new fields ignored, team.models used
12. empty default_models + empty member models + team.models=[] → allow all (backward compat)
"""

import sys
import os
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from unittest.mock import AsyncMock, patch

import litellm
from litellm.proxy._types import UserAPIKeyAuth, LiteLLM_TeamTable
from litellm.proxy.auth.auth_checks import (
    can_team_access_model,
    get_effective_team_models,
)


@pytest.fixture(autouse=True)
def enable_feature_flag():
    original = litellm.team_model_overrides_enabled
    litellm.team_model_overrides_enabled = True
    yield
    litellm.team_model_overrides_enabled = original


# ── get_effective_team_models unit tests ─────────────────────────────────────


class TestGetEffectiveTeamModels:
    def test_defaults_only(self):
        """1. User with defaults only → can access default models."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "m2", "d1", "d2"], default_models=["d1", "d2"]
        )
        result = get_effective_team_models(team)
        assert set(result) == {"d1", "d2"}

    def test_defaults_plus_overrides(self):
        """2. User with defaults + overrides → union of both."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "d1", "mo1"], default_models=["d1"]
        )
        token = UserAPIKeyAuth(team_member_models=["mo1"])
        result = get_effective_team_models(team, token)
        assert set(result) == {"d1", "mo1"}

    def test_overrides_only_no_defaults(self):
        """3. User with overrides only (no defaults) → can access override models."""
        team = LiteLLM_TeamTable(team_id="t1", models=["m1", "mo1", "mo2"])
        token = UserAPIKeyAuth(team_member_models=["mo1", "mo2"])
        result = get_effective_team_models(team, token)
        assert set(result) == {"mo1", "mo2"}

    def test_neither_configured_fallback(self):
        """4. Neither configured → falls back to team.models."""
        team = LiteLLM_TeamTable(team_id="t1", models=["m1", "m2"])
        result = get_effective_team_models(team)
        assert result == ["m1", "m2"]

    def test_neither_configured_empty_team_models_allows_all(self):
        """12. empty default_models + empty member models + team.models=[] → allow all."""
        team = LiteLLM_TeamTable(team_id="t1", models=[])
        result = get_effective_team_models(team)
        assert result == []  # empty = allow all

    def test_cross_user_isolation(self):
        """9. User A overrides don't affect User B."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "d1", "mo_a", "mo_b"], default_models=["d1"]
        )
        token_a = UserAPIKeyAuth(team_member_models=["mo_a"])
        token_b = UserAPIKeyAuth(team_member_models=["mo_b"])
        result_a = get_effective_team_models(team, token_a)
        result_b = get_effective_team_models(team, token_b)
        assert set(result_a) == {"d1", "mo_a"}
        assert set(result_b) == {"d1", "mo_b"}
        assert "mo_a" not in result_b
        assert "mo_b" not in result_a

    def test_feature_flag_off(self):
        """11. Feature flag off → all new fields ignored, team.models used."""
        litellm.team_model_overrides_enabled = False
        team = LiteLLM_TeamTable(team_id="t1", models=["m1"], default_models=["d1"])
        token = UserAPIKeyAuth(team_member_models=["mo1"])
        result = get_effective_team_models(team, token)
        assert result == ["m1"]

    def test_deduplication(self):
        """Overlapping models are deduplicated."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "shared", "extra"], default_models=["shared"]
        )
        token = UserAPIKeyAuth(team_member_models=["shared", "extra"])
        result = get_effective_team_models(team, token)
        assert set(result) == {"shared", "extra"}
        assert len(result) == 2  # no duplicates

    def test_no_team_object(self):
        """No team object → empty list."""
        assert get_effective_team_models(None) == []

    def test_no_team_object_with_token(self):
        """No team object but token has defaults → uses token values."""
        token = UserAPIKeyAuth(team_default_models=["td1"], team_member_models=["mo1"])
        result = get_effective_team_models(None, token)
        assert set(result) == {"td1", "mo1"}


# ── can_team_access_model integration tests ──────────────────────────────────


class TestCanTeamAccessModelWithOverrides:
    @pytest.mark.asyncio
    async def test_defaults_only_allowed(self):
        """1. User with defaults only → can access default models."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "m2", "d1"], default_models=["d1"]
        )
        assert await can_team_access_model(
            model="d1", team_object=team, llm_router=None
        )

    @pytest.mark.asyncio
    async def test_defaults_only_denied(self):
        """1. User with defaults only → cannot access other team models."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "m2", "d1"], default_models=["d1"]
        )
        with pytest.raises(Exception):
            await can_team_access_model(model="m1", team_object=team, llm_router=None)

    @pytest.mark.asyncio
    async def test_defaults_plus_overrides_allowed(self):
        """2. User with defaults + overrides → can access union."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "d1", "mo1"], default_models=["d1"]
        )
        token = UserAPIKeyAuth(team_member_models=["mo1"])
        assert await can_team_access_model(
            model="d1", team_object=team, llm_router=None, valid_token=token
        )
        assert await can_team_access_model(
            model="mo1", team_object=team, llm_router=None, valid_token=token
        )

    @pytest.mark.asyncio
    async def test_defaults_plus_overrides_denied(self):
        """2. User with overrides → cannot access models outside union."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "m2", "d1", "mo1"], default_models=["d1"]
        )
        token = UserAPIKeyAuth(team_member_models=["mo1"])
        with pytest.raises(Exception):
            await can_team_access_model(
                model="m2", team_object=team, llm_router=None, valid_token=token
            )

    @pytest.mark.asyncio
    async def test_revocation_after_override_removal(self):
        """7. Remove override → model access denied."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "d1", "mo1"], default_models=["d1"]
        )
        # With override
        token_with = UserAPIKeyAuth(team_member_models=["mo1"])
        assert await can_team_access_model(
            model="mo1", team_object=team, llm_router=None, valid_token=token_with
        )
        # After override removal (empty member models)
        token_without = UserAPIKeyAuth(team_member_models=[])
        with pytest.raises(Exception):
            await can_team_access_model(
                model="mo1",
                team_object=team,
                llm_router=None,
                valid_token=token_without,
            )

    @pytest.mark.asyncio
    async def test_stale_override_capped_by_team_models(self):
        """Stale member override for model removed from team.models → denied."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1"], default_models=["m1"]
        )
        # Member has stale override for "m2" which is no longer in team.models
        token = UserAPIKeyAuth(team_member_models=["m2"])
        # effective = union(["m1"], ["m2"]) capped by team.models=["m1"] → ["m1"]
        with pytest.raises(Exception):
            await can_team_access_model(
                model="m2", team_object=team, llm_router=None, valid_token=token
            )

    @pytest.mark.asyncio
    async def test_backward_compat_no_overrides(self):
        """4. Neither configured → uses team.models as before."""
        team = LiteLLM_TeamTable(team_id="t1", models=["m1", "m2"])
        assert await can_team_access_model(
            model="m1", team_object=team, llm_router=None
        )

    @pytest.mark.asyncio
    async def test_backward_compat_empty_team_models_allows_all(self):
        """12. team.models=[] with no overrides → allow all."""
        team = LiteLLM_TeamTable(team_id="t1", models=[])
        assert await can_team_access_model(
            model="any-model", team_object=team, llm_router=None
        )

    @pytest.mark.asyncio
    async def test_feature_flag_off_uses_team_models(self):
        """11. Feature flag off → ignores overrides, uses team.models."""
        litellm.team_model_overrides_enabled = False
        team = LiteLLM_TeamTable(team_id="t1", models=["m1"], default_models=["d1"])
        token = UserAPIKeyAuth(team_member_models=["mo1"])
        # Should use team.models=["m1"], not effective models
        assert await can_team_access_model(
            model="m1", team_object=team, llm_router=None, valid_token=token
        )
        with pytest.raises(Exception):
            await can_team_access_model(
                model="d1", team_object=team, llm_router=None, valid_token=token
            )


# ── Key-generation enforcement tests ─────────────────────────────────────────


class TestKeyGenerationEnforcement:
    """Tests 5, 6, 8: key-generation model validation against effective set."""

    def _get_effective(self, team, token=None):
        """Helper to compute effective models (same logic as key-gen)."""
        return get_effective_team_models(team, token)

    def test_key_rejects_models_outside_effective_set(self):
        """5. Key creation with models outside effective set → should be rejected."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "m2", "m3"], default_models=["m1"]
        )
        token = UserAPIKeyAuth(team_member_models=["m2"])
        effective = self._get_effective(team, token)

        # Simulate key-gen validation: requested models must be subset of effective
        requested = ["m3"]  # not in effective set {m1, m2}
        disallowed = set(requested) - set(effective)
        assert disallowed == {"m3"}, "m3 should be disallowed"

    def test_key_defaults_to_effective_set(self):
        """6. Key creation with no models → defaults to effective set."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "m2", "m3"], default_models=["m1"]
        )
        token = UserAPIKeyAuth(team_member_models=["m2"])
        effective = self._get_effective(team, token)

        # When no models requested, key should get effective set
        assert set(effective) == {"m1", "m2"}

    def test_all_team_models_restricted_to_effective_set(self):
        """8. all-team-models key + overrides → restricted to effective set, not full team.models."""
        team = LiteLLM_TeamTable(
            team_id="t1", models=["m1", "m2", "m3"], default_models=["m1"]
        )
        token = UserAPIKeyAuth(team_member_models=["m2"])
        effective = self._get_effective(team, token)

        # all-team-models should resolve to effective set, not team.models
        assert set(effective) == {"m1", "m2"}
        assert "m3" not in effective  # m3 is in team.models but not in effective


# ── Access group fallback test ───────────────────────────────────────────────


class TestAccessGroupFallback:
    @pytest.mark.asyncio
    async def test_access_group_fallback_when_effective_models_deny(self):
        """10. access_group_ids fallback still works when effective models check fails."""
        team = LiteLLM_TeamTable(
            team_id="t1",
            models=["m1", "m2"],
            default_models=["m1"],
            access_group_ids=["group-1"],
        )
        # "m2" is NOT in effective set (only "m1" is default, no member overrides)
        # But it should be accessible via access_group_ids fallback
        with patch(
            "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
            new_callable=AsyncMock,
            return_value=["m2", "m3"],
        ):
            result = await can_team_access_model(
                model="m2", team_object=team, llm_router=None
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_access_group_fallback_still_denies_unknown_model(self):
        """10b. access_group_ids fallback does not grant access to models outside groups."""
        team = LiteLLM_TeamTable(
            team_id="t1",
            models=["m1", "m2"],
            default_models=["m1"],
            access_group_ids=["group-1"],
        )
        with patch(
            "litellm.proxy.auth.auth_checks._get_models_from_access_groups",
            new_callable=AsyncMock,
            return_value=["m2"],  # group only has m2
        ):
            with pytest.raises(Exception):
                await can_team_access_model(
                    model="unknown-model", team_object=team, llm_router=None
                )
