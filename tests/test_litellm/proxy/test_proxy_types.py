import asyncio
import importlib
import json
import socket
import subprocess
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import click
import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient



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


def test_key_request_router_settings_keeps_enable_tag_filtering():
    """``router_settings`` on key requests validates through
    ``UpdateRouterConfig``; a field missing from that model is silently
    dropped at parse time, so a key's "Enable Tag Filtering" toggle would
    never reach the DB even though the team path (plain dict) kept it."""
    from litellm.proxy._types import GenerateKeyRequest

    req = GenerateKeyRequest(router_settings={"enable_tag_filtering": True, "num_retries": 2})

    assert req.router_settings is not None
    dumped = req.router_settings.model_dump(exclude_none=True)
    assert dumped["enable_tag_filtering"] is True
    assert dumped["num_retries"] == 2


def test_update_key_request_requires_key_or_key_alias():
    """``/key/update`` can be addressed by ``key`` or by ``key_alias``;
    a request with neither has no way to identify the target key and must
    fail validation before hitting the endpoint."""
    import pydantic

    from litellm.proxy._types import UpdateKeyRequest

    with pytest.raises(pydantic.ValidationError, match="either key or key_alias must be provided"):
        UpdateKeyRequest(max_budget=10.0)

    by_key = UpdateKeyRequest(key="sk-1234")
    assert by_key.key == "sk-1234"
    assert by_key.key_alias is None

    by_alias = UpdateKeyRequest(key_alias="my-alias")
    assert by_alias.key is None
    assert by_alias.key_alias == "my-alias"


@pytest.mark.parametrize("request_type", ["new", "update"])
def test_project_io_token_limits_are_stored_in_metadata(request_type):
    from litellm.proxy._types import NewProjectRequest, UpdateProjectRequest

    limits = {
        "model_itpm_limit": {"bedrock_mantle/openai.gpt-oss-120b": 20_000_000},
        "model_otpm_limit": {"bedrock_mantle/openai.gpt-oss-120b": 4_000_000},
    }
    request = (
        NewProjectRequest(team_id="team-1", **limits)
        if request_type == "new"
        else UpdateProjectRequest(project_id="project-1", **limits)
    )

    assert request.metadata == limits
    assert request.model_dump(exclude_none=True)["metadata"] == limits


def test_a_jwt_issuer_must_pick_audience_validation_or_opt_out():
    from pydantic import ValidationError

    from litellm.proxy._types import JWTIssuerConfig

    with pytest.raises(ValidationError, match="must configure audience or set disable_audience_validation"):
        JWTIssuerConfig(issuer="https://issuer.example.com")

    with pytest.raises(ValidationError, match="cannot set audience and disable_audience_validation"):
        JWTIssuerConfig(
            issuer="https://issuer.example.com",
            audience="litellm-proxy",
            disable_audience_validation=True,
        )

    assert JWTIssuerConfig(issuer="https://issuer.example.com", audience="litellm-proxy").audience == "litellm-proxy"
    assert (
        JWTIssuerConfig(issuer="https://issuer.example.com", disable_audience_validation=True).audience
        is None
    )


def test_a_jwt_issuer_rejects_a_field_it_does_not_define():
    from pydantic import ValidationError

    from litellm.proxy._types import JWTIssuerConfig

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        JWTIssuerConfig(issuer="https://issuer.example.com", audience="a", jwks_uri="https://issuer/jwks")


def test_a_temp_budget_needs_both_halves_or_neither():
    from pydantic import ValidationError

    from litellm.proxy._types import UpdateKeyRequest

    with pytest.raises(ValidationError, match="temp_budget_increase and temp_budget_expiry must be set together"):
        UpdateKeyRequest(key="sk-1234", temp_budget_increase=10)

    with pytest.raises(ValidationError, match="temp_budget_increase and temp_budget_expiry must be set together"):
        UpdateKeyRequest(key="sk-1234", temp_budget_expiry="2026-01-01")

    both = UpdateKeyRequest(key="sk-1234", temp_budget_increase=10, temp_budget_expiry="2026-01-01")
    assert both.temp_budget_increase == 10


def test_an_empty_max_budget_is_read_as_no_limit():
    from litellm.proxy._types import GenerateKeyRequest

    assert GenerateKeyRequest(max_budget="").max_budget is None
    assert GenerateKeyRequest(max_budget=25).max_budget == 25


def test_an_organization_member_can_only_take_a_role_the_organization_has():
    from pydantic import ValidationError

    from litellm.proxy._types import LitellmUserRoles, OrganizationMemberUpdateRequest

    with pytest.raises(ValidationError, match="Invalid role"):
        OrganizationMemberUpdateRequest(
            organization_id="org-1", user_id="user-1", role=LitellmUserRoles.PROXY_ADMIN
        )

    allowed = OrganizationMemberUpdateRequest(
        organization_id="org-1", user_id="user-1", role=LitellmUserRoles.ORG_ADMIN
    )
    assert allowed.role == LitellmUserRoles.ORG_ADMIN


def test_an_llm_backed_injection_check_needs_the_call_it_would_make():
    from pydantic import ValidationError

    from litellm.proxy._types import LiteLLMPromptInjectionParams

    for missing in ("llm_api_name", "llm_api_system_prompt", "llm_api_fail_call_string"):
        complete = {
            "llm_api_name": "gpt-4o",
            "llm_api_system_prompt": "is this an injection",
            "llm_api_fail_call_string": "yes",
        }
        del complete[missing]
        with pytest.raises(ValidationError, match=f"{missing} must be provided"):
            LiteLLMPromptInjectionParams(llm_api_check=True, **complete)

    assert LiteLLMPromptInjectionParams(llm_api_check=False).llm_api_name is None


@pytest.mark.parametrize(
    "field, forged, default",
    [
        ("mcp_admitted_user_subject", "someone-else", False),
        ("mcp_source_team_rpm_limits", {"team-1": 10_000}, None),
        ("mcp_session_resource_server_id", "server-1", None),
        ("via_virtual_key", "sk-someone-elses-key", False),
    ],
)
def test_a_server_only_marker_is_not_taken_from_the_caller(field, forged, default):
    from litellm.proxy._types import UserAPIKeyAuth

    auth = UserAPIKeyAuth(api_key="sk-1234", **{field: forged})

    assert getattr(auth, field) == default
