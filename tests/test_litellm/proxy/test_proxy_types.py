import asyncio
import importlib
import json
import os
import socket
import subprocess
import sys
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import click
import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path


def test_audit_log_masking():
    from datetime import datetime

    from litellm.proxy._types import LiteLLM_AuditLogs

    audit_log = LiteLLM_AuditLogs(
        id="123",
        updated_at=datetime.now(),
        changed_by="test",
        changed_by_api_key="test",
        table_name="LiteLLM_VerificationToken",
        object_id="test",
        action="updated",
        updated_values=json.dumps({"key": "sk-1234567890", "token": "1q2132r222"}),
        before_value=json.dumps({"key": "sk-1234567890", "token": "1q2132r222"}),
    )

    print(audit_log.updated_values)
    json_updated_values = json.loads(audit_log.updated_values)
    assert json_updated_values["token"] == "1q2132r222"
    assert json_updated_values["key"] == "sk-1*****7890"
    assert audit_log.before_value
    json_before_value = json.loads(audit_log.before_value)
    assert json_before_value["token"] == "1q2132r222"
    assert json_before_value["key"] == "sk-1*****7890"


def test_internal_jobs_user_has_proxy_admin_role():
    """
    Test that the internal jobs system user has PROXY_ADMIN role.

    This is critical for key rotation to work properly. The system user needs
    PROXY_ADMIN role to bypass team permission checks in
    TeamMemberPermissionChecks.can_team_member_execute_key_management_endpoint()

    Regression test for: https://github.com/BerriAI/litellm/pull/21896
    """
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    # Get the system user used for internal jobs like key rotation
    system_user = UserAPIKeyAuth.get_litellm_internal_jobs_user_api_key_auth()

    # Verify the system user has PROXY_ADMIN role
    assert system_user.user_role == LitellmUserRoles.PROXY_ADMIN

    # Verify other expected properties
    assert system_user.user_id == "system"
    assert system_user.team_id == "system"
    assert system_user.team_alias == "system"


def test_team_membership_optional_budget_table_defaults_to_none():
    """A team member with no joined budget table must construct cleanly.

    Pydantic v2 treats ``Optional[X]`` without an explicit default as a
    *required* field (unlike v1, where ``Optional`` implied a ``None``
    default). ``LiteLLM_TeamMembership.litellm_budget_table`` was declared
    without ``= None``, so building the model from a team-member row with
    ``budget_id = NULL`` (no budget table joined) raised "Field required" and
    the request failed with HTTP 401 instead of being processed.

    Regression test for https://github.com/BerriAI/litellm/issues/30437.
    """
    from litellm.proxy._types import LiteLLM_TeamMembership

    membership = LiteLLM_TeamMembership(
        user_id="user-123",
        team_id="team-456",
        spend=4.73,
    )

    assert membership.litellm_budget_table is None
    # The rpm/tpm helpers must stay safe when no budget row is joined.
    assert membership.safe_get_team_member_rpm_limit() is None
    assert membership.safe_get_team_member_tpm_limit() is None
