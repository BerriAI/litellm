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


def test_log_filters_config_defaults():
    from litellm.proxy._types import LogFiltersConfig

    config = LogFiltersConfig()
    assert config.excluded_uvicorn_access_paths == []
    assert config.exclude_health_check_paths is True


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


def test_team_membership_null_budget_table():
    """
    Regression test for: LiteLLM_TeamMembership.litellm_budget_table missing = None.
    In Pydantic v2, Optional[T] without a default is required; rows with budget_id=null
    raised a validation error and returned 401.
    Related: https://github.com/BerriAI/litellm/issues/28689
    """
    from litellm.proxy._types import LiteLLM_TeamMembership

    membership = LiteLLM_TeamMembership(user_id="u1", team_id="t1")
    assert membership.litellm_budget_table is None

    membership_explicit = LiteLLM_TeamMembership(
        user_id="u1", team_id="t1", litellm_budget_table=None
    )
    assert membership_explicit.litellm_budget_table is None


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


def test_user_api_key_auth_hashes_authorization_header_form_of_key():
    from litellm.proxy._types import UserAPIKeyAuth

    raw_key = "sk-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
    baseline = UserAPIKeyAuth(api_key=raw_key)

    for header_form in (
        f"Bearer {raw_key}",
        f"bearer {raw_key}",
        f"BEARER {raw_key}",
        f"BeArEr {raw_key}",
    ):
        from_header = UserAPIKeyAuth(api_key=header_form)
        assert from_header.api_key == baseline.api_key
        assert from_header.token == baseline.token
        assert not from_header.api_key.lower().startswith("bearer")


def test_proxy_exception_str_returns_message():
    """ProxyException must stringify to its message: OTEL's
    ``span.record_exception`` and ``str(exc)``-based logging read the string
    form, which was empty pre-fix. The OpenAI-mapped fields must stay intact."""
    from litellm.proxy._types import ProxyException

    msg = "Authentication Error, Invalid proxy server token passed."
    exc = ProxyException(message=msg, type="auth_error", param="key", code=401)

    assert str(exc) == msg
    assert exc.message == msg
    assert exc.to_dict() == {
        "message": msg,
        "type": "auth_error",
        "param": "key",
        "code": "401",
    }
