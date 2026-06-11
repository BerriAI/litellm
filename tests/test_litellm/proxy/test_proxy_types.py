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


def test_litellm_usertable_validate_from_pydantic_model():
    """
    LiteLLM_UserTable.model_validate must accept another BaseModel instance, not
    only a dict.

    Regression test for the SSO first-login 500. On the first login the user does
    not exist yet, so new_user() returns a NewUserResponse object which is then
    cached with model_type=LiteLLM_UserTable (see _sync_user_role_from_jwt_role_map
    in litellm/proxy/management_endpoints/ui_sso.py). Pydantic runs the
    set_model_info (mode="before") validator with that model instance, which used
    to call .get() on it and raise `AttributeError: 'NewUserResponse' object has
    no attribute 'get'`. The second login worked only because the user was then
    fetched from the DB as a real LiteLLM_UserTable, which Pydantic returns as-is
    (revalidate_instances="never"), skipping the validator.
    """
    from litellm.proxy._types import LiteLLM_UserTable, NewUserResponse

    new_user_response = NewUserResponse(user_id="test-user-id", key="sk-test")

    validated = LiteLLM_UserTable.model_validate(new_user_response)

    assert validated.user_id == "test-user-id"
    assert validated.spend == 0.0
    assert validated.models == []
    assert validated.teams == []
