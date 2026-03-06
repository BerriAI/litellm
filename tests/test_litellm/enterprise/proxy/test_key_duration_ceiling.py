"""
Tests for add_team_member_key_duration ceiling behavior.

team_member_key_duration acts as a ceiling:
- Not provided by user → apply team max as default
- Provided and shorter than team max → respect user's value
- No team / no metadata / no team_member_key_duration → leave unchanged
- Service account (user_id=None) → leave unchanged
"""

import importlib.util
import os
import sys

import pytest
from unittest.mock import MagicMock

from litellm.proxy._types import GenerateKeyRequest, LiteLLM_TeamTable

_ENTERPRISE_SOURCE = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "..",
        "enterprise", "litellm_enterprise", "proxy",
        "management_endpoints", "key_management_endpoints.py",
    )
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(_ENTERPRISE_SOURCE),
    reason=f"Enterprise source not available at {_ENTERPRISE_SOURCE}",
)


def _load_local_add_team_member_key_duration():
    """Load add_team_member_key_duration from the local enterprise source tree."""
    module_name = "_local_enterprise_key_management_endpoints"
    # Remove cached version so we always reload from the local file
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, _ENTERPRISE_SOURCE)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.add_team_member_key_duration


def _make_team(duration: str) -> LiteLLM_TeamTable:
    team = MagicMock(spec=LiteLLM_TeamTable)
    team.metadata = {"team_member_key_duration": duration}
    return team


class TestAddTeamMemberKeyDuration:
    def test_no_duration_provided_uses_team_max(self):
        """User omits duration → team max is applied as the default."""
        fn = _load_local_add_team_member_key_duration()

        data = GenerateKeyRequest(user_id="user-1")
        assert "duration" not in data.model_fields_set

        result = fn(_make_team("30d"), data)

        assert result.duration == "30d"

    def test_shorter_duration_respected(self):
        """User provides a duration shorter than team max → their value is kept."""
        fn = _load_local_add_team_member_key_duration()

        data = GenerateKeyRequest(user_id="user-1", duration="5d")
        assert "duration" in data.model_fields_set

        result = fn(_make_team("30d"), data)

        assert result.duration == "5d"

    def test_duration_equal_to_team_max_respected(self):
        """User provides duration equal to team max → their value is kept."""
        fn = _load_local_add_team_member_key_duration()

        data = GenerateKeyRequest(user_id="user-1", duration="30d")
        result = fn(_make_team("30d"), data)

        assert result.duration == "30d"

    def test_no_team_returns_unchanged(self):
        """team_table=None → data is returned unchanged."""
        fn = _load_local_add_team_member_key_duration()

        data = GenerateKeyRequest(user_id="user-1")
        result = fn(None, data)

        assert result.duration is None

    def test_no_metadata_returns_unchanged(self):
        """Team has no metadata → data is returned unchanged."""
        fn = _load_local_add_team_member_key_duration()

        team = MagicMock(spec=LiteLLM_TeamTable)
        team.metadata = None

        data = GenerateKeyRequest(user_id="user-1")
        result = fn(team, data)

        assert result.duration is None

    def test_no_team_member_key_duration_returns_unchanged(self):
        """team_member_key_duration absent from metadata → data is returned unchanged."""
        fn = _load_local_add_team_member_key_duration()

        team = MagicMock(spec=LiteLLM_TeamTable)
        team.metadata = {}

        data = GenerateKeyRequest(user_id="user-1")
        result = fn(team, data)

        assert result.duration is None

    def test_service_account_returns_unchanged(self):
        """user_id=None (service account) → team duration is NOT applied."""
        fn = _load_local_add_team_member_key_duration()

        data = GenerateKeyRequest(user_id=None)
        result = fn(_make_team("30d"), data)

        # Verify the service-account guard prevented the team max from being applied
        assert result.duration is None, "Service account should not inherit team duration"
        assert result.duration != "30d"
