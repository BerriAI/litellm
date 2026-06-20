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


# === Regression tests for LIT-3094: ProxyException must populate Exception.args
# so logging integrations using str(exc) record a non-empty error_message. ===


def test_proxy_exception_str_returns_message():
    """str(ProxyException) must return the stored message, not '' (LIT-3094)."""
    from litellm.proxy._types import ProxyException

    msg = "key not allowed to access model"
    exc = ProxyException(message=msg, type="auth_error", param=None, code=401)
    assert str(exc) == msg
    assert exc.args == (msg,)
    assert exc.message == msg


def test_proxy_exception_populates_standard_logging_error_message():
    """The full logging path used by proxy callbacks must capture the message
    instead of recording an empty error_message (LIT-3094 report)."""
    from litellm.litellm_core_utils.litellm_logging import (
        StandardLoggingPayloadSetup,
    )
    from litellm.proxy._types import ProxyException

    msg = "Authentication Error, Invalid proxy server token passed."
    exc = ProxyException(message=msg, type="auth_error", param=None, code=401)
    info = StandardLoggingPayloadSetup.get_error_information(original_exception=exc)
    assert info["error_message"] == msg
    assert info["error_class"] == "ProxyException"
    assert info["error_code"] == "401"


def test_proxy_exception_to_dict_unchanged():
    """to_dict() shape must remain backwards-compatible after the fix."""
    from litellm.proxy._types import ProxyException

    exc = ProxyException(
        message="boom", type="invalid_request_error", param="model", code=400
    )
    d = exc.to_dict()
    assert d == {
        "message": "boom",
        "type": "invalid_request_error",
        "param": "model",
        "code": "400",
    }


def test_proxy_exception_routing_code_override_still_works():
    """The 'No healthy deployment available' -> 429 remapping must survive
    the super().__init__() addition."""
    from litellm.proxy._types import ProxyException

    exc = ProxyException(
        message="No healthy deployment available for model=foo",
        type="router_error",
        param=None,
        code=500,
    )
    assert exc.code == "429"
    assert str(exc) == "No healthy deployment available for model=foo"


def test_proxy_exception_non_string_message_coerced():
    """Non-string `message` must still be coerced to str via self.message =
    str(message), and Exception.args must reflect the coerced value."""
    from litellm.proxy._types import ProxyException

    exc = ProxyException(message=42, type="x", param=None, code=400)
    assert exc.message == "42"
    assert str(exc) == "42"
    assert exc.args == ("42",)
