"""
Tests for litellm/proxy/management_endpoints/common_utils.py

Specifically tests that _update_metadata_fields does not trigger premium
user checks when premium fields are present but empty.

Related: https://github.com/BerriAI/litellm/issues/20534
"""

from unittest.mock import patch

import pytest

from litellm.proxy.management_endpoints.common_utils import (
    _has_non_empty_value,
    _update_metadata_fields,
)


class TestHasNonEmptyValue:
    """Tests for the _has_non_empty_value helper."""

    def test_none_is_empty(self):
        assert _has_non_empty_value(None) is False

    def test_empty_list_is_empty(self):
        assert _has_non_empty_value([]) is False

    def test_empty_string_is_empty(self):
        assert _has_non_empty_value("") is False

    def test_blank_string_is_empty(self):
        assert _has_non_empty_value("   ") is False

    def test_non_empty_list_has_value(self):
        assert _has_non_empty_value(["policy-a"]) is True

    def test_non_empty_string_has_value(self):
        assert _has_non_empty_value("30d") is True

    def test_dict_has_value(self):
        assert _has_non_empty_value({"key": "val"}) is True

    def test_empty_dict_has_value(self):
        # empty dict is not None/list/str, so it counts as non-empty
        assert _has_non_empty_value({}) is True


class TestUpdateMetadataFieldsPremiumCheck:
    """
    Tests that _update_metadata_fields skips premium user checks for empty
    values but still enforces them for real values.

    Issue: The UI sends the full form on every team update, including premium
    fields like `policies: []`. The backend was treating these empty values
    as premium feature usage and returning 403.
    """

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_empty_policies_skips_premium_check(self, mock_check):
        """policies: [] should NOT trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "team_alias": "my-team",
            "policies": [],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_empty_guardrails_skips_premium_check(self, mock_check):
        """guardrails: [] should NOT trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "guardrails": [],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_empty_string_team_member_key_duration_skips_premium_check(
        self, mock_check
    ):
        """team_member_key_duration: '' should NOT trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "team_member_key_duration": "",
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
        side_effect=Exception("Should not be called"),
    )
    def test_full_ui_payload_with_empty_premium_fields_skips_premium_check(
        self, mock_check
    ):
        """A realistic UI payload with all empty premium fields should not 403."""
        updated_kv = {
            "team_id": "team-123",
            "team_alias": "renamed-team",
            "models": ["gpt-4o"],
            "max_budget": 200,
            "policies": [],
            "guardrails": [],
            "logging": [],
            "team_member_key_duration": "",
            "prompts": [],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_not_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
    )
    def test_non_empty_policies_triggers_premium_check(self, mock_check):
        """policies: ['real-policy'] SHOULD trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "policies": ["real-policy"],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
    )
    def test_non_empty_guardrails_triggers_premium_check(self, mock_check):
        """guardrails: ['my-guardrail'] SHOULD trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "guardrails": ["my-guardrail"],
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_called()

    @patch(
        "litellm.proxy.management_endpoints.common_utils._premium_user_check",
    )
    def test_non_empty_team_member_key_duration_triggers_premium_check(
        self, mock_check
    ):
        """team_member_key_duration: '30d' SHOULD trigger premium user check."""
        updated_kv = {
            "team_id": "team-123",
            "team_member_key_duration": "30d",
        }
        _update_metadata_fields(updated_kv)
        mock_check.assert_called()


# ---------------------------------------------------------------------------
# Tests for _upsert_budget_and_membership clone-on-write (LIT-3359)
# ---------------------------------------------------------------------------

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.common_utils import (
    _upsert_budget_and_membership,
)


def _make_tx(*, ref_count: int, default_budget_fields=None):
    """Build a minimal mock prisma transaction with the methods the
    function uses: count, update, find_unique, create, upsert."""

    default_budget_fields = default_budget_fields or {
        "budget_id": "shared-bdg-1",
        "max_budget": 100.0,
        "soft_budget": None,
        "max_parallel_requests": None,
        "tpm_limit": None,
        "rpm_limit": None,
        "model_max_budget": None,
        "budget_duration": None,
        "allowed_models": None,
    }

    tx = SimpleNamespace()
    # Mock find_first returns a sentinel object iff there is at least one
    # OTHER membership pointing at this budget_id (i.e. ref_count >= 2).
    other_ref = object() if ref_count >= 2 else None
    tx.litellm_teammembership = SimpleNamespace(
        find_first=AsyncMock(return_value=other_ref),
        update=AsyncMock(),
        upsert=AsyncMock(),
    )
    fake_budget_row = MagicMock()
    fake_budget_row.model_dump = MagicMock(return_value=default_budget_fields)
    created = SimpleNamespace(budget_id="new-bdg-uuid")
    tx.litellm_budgettable = SimpleNamespace(
        update=AsyncMock(),
        find_unique=AsyncMock(return_value=fake_budget_row),
        create=AsyncMock(return_value=created),
    )
    return tx


@pytest.mark.asyncio
class TestUpsertBudgetMembershipSharedRow:
    """LIT-3359: when multiple memberships still point at the same budget_id,
    `_upsert_budget_and_membership` MUST clone-on-write — never mutate the
    shared row in place."""

    async def test_shared_row_triggers_clone_on_write(self):
        """Two memberships share budget_id -> create new budget, repoint."""
        tx = _make_tx(ref_count=2)
        await _upsert_budget_and_membership(
            tx,
            team_id="t1",
            user_id="alice",
            max_budget=42.0,
            existing_budget_id="shared-bdg-1",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
            team_default_budget_id=None,
        )
        # Must NOT have mutated the shared row in place
        tx.litellm_budgettable.update.assert_not_called()
        # Must have created a NEW private budget for Alice
        tx.litellm_budgettable.create.assert_called_once()
        create_data = tx.litellm_budgettable.create.call_args.kwargs["data"]
        assert create_data["max_budget"] == 42.0
        # Must have re-linked Alice's membership to the new budget
        tx.litellm_teammembership.upsert.assert_called_once()
        upsert_kwargs = tx.litellm_teammembership.upsert.call_args.kwargs
        assert upsert_kwargs["where"]["user_id_team_id"]["user_id"] == "alice"
        assert (
            upsert_kwargs["data"]["update"]["litellm_budget_table"]["connect"][
                "budget_id"
            ]
            == "new-bdg-uuid"
        )

    async def test_solo_membership_still_updates_in_place(self):
        """Single membership owning its budget_id keeps the cheaper in-place
        update path (no regression in normal behaviour)."""
        tx = _make_tx(ref_count=1)
        await _upsert_budget_and_membership(
            tx,
            team_id="t1",
            user_id="alice",
            max_budget=42.0,
            existing_budget_id="solo-bdg",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
            team_default_budget_id=None,
        )
        tx.litellm_budgettable.update.assert_called_once()
        update_args = tx.litellm_budgettable.update.call_args.kwargs
        assert update_args["where"]["budget_id"] == "solo-bdg"
        assert update_args["data"]["max_budget"] == 42.0
        tx.litellm_budgettable.create.assert_not_called()
        tx.litellm_teammembership.upsert.assert_not_called()

    async def test_clone_seeds_unchanged_fields_from_shared_row(self):
        """When forking off a shared row, fields the caller did NOT change
        (e.g. tpm_limit, allowed_models) must carry over from the shared row
        into the new private budget."""
        shared = {
            "budget_id": "shared-bdg-2",
            "max_budget": 100.0,
            "soft_budget": 80.0,
            "max_parallel_requests": 5,
            "tpm_limit": 1000,
            "rpm_limit": 60,
            "model_max_budget": {"gpt-4o": 50.0},
            "budget_duration": "30d",
            "allowed_models": ["gpt-4o", "claude-3-haiku"],
        }
        tx = _make_tx(ref_count=3, default_budget_fields=shared)
        await _upsert_budget_and_membership(
            tx,
            team_id="t1",
            user_id="alice",
            max_budget=42.0,  # only changing this
            existing_budget_id="shared-bdg-2",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
            team_default_budget_id=None,
        )
        tx.litellm_budgettable.update.assert_not_called()
        create_data = tx.litellm_budgettable.create.call_args.kwargs["data"]
        # Caller override wins:
        assert create_data["max_budget"] == 42.0
        # Carry-over from shared row:
        assert create_data["soft_budget"] == 80.0
        assert create_data["max_parallel_requests"] == 5
        assert create_data["tpm_limit"] == 1000
        assert create_data["rpm_limit"] == 60
        assert create_data["budget_duration"] == "30d"
        assert create_data["allowed_models"] == ["gpt-4o", "claude-3-haiku"]
        assert create_data["model_max_budget"] == {"gpt-4o": 50.0}

    async def test_team_default_match_still_clones(self):
        """Pre-existing behaviour: membership pointing at team_default_budget_id
        clones-on-write regardless of ref-count."""
        tx = _make_tx(ref_count=1)  # only one ref, but it's the team default
        await _upsert_budget_and_membership(
            tx,
            team_id="t1",
            user_id="alice",
            max_budget=42.0,
            existing_budget_id="team-default-bdg",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
            team_default_budget_id="team-default-bdg",
        )
        tx.litellm_budgettable.update.assert_not_called()
        tx.litellm_budgettable.create.assert_called_once()
        tx.litellm_teammembership.upsert.assert_called_once()
        # And we did NOT bother running the ref-count probe in this path
        tx.litellm_teammembership.find_first.assert_not_called()

    async def test_count_probe_failure_defaults_to_clone_on_write(self):
        """If the ref-count probe raises, the function must take the safer
        clone-on-write branch (never silently fall through to in-place update
        of a possibly-shared row)."""
        tx = _make_tx(ref_count=1)
        tx.litellm_teammembership.find_first = AsyncMock(
            side_effect=RuntimeError("simulated DB failure")
        )
        await _upsert_budget_and_membership(
            tx,
            team_id="t1",
            user_id="alice",
            max_budget=42.0,
            existing_budget_id="some-bdg",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
            team_default_budget_id=None,
        )
        tx.litellm_budgettable.update.assert_not_called()
        tx.litellm_budgettable.create.assert_called_once()

    async def test_all_limits_none_still_disconnects(self):
        """When every limit is None the function must disconnect the
        budget (and not run the ref-count probe at all)."""
        tx = _make_tx(ref_count=2)
        await _upsert_budget_and_membership(
            tx,
            team_id="t1",
            user_id="alice",
            max_budget=None,
            existing_budget_id="shared-bdg-1",
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
            team_default_budget_id=None,
        )
        tx.litellm_teammembership.update.assert_called_once()
        tx.litellm_teammembership.upsert.assert_not_called()
        tx.litellm_teammembership.find_first.assert_not_called()
