from dataclasses import FrozenInstanceError

import pytest

from litellm.proxy._experimental.mcp_server.operations import prepare_context
from litellm.proxy._types import UserAPIKeyAuth


def test_operation_context_isolates_nested_headers_and_caller_permissions():
    caller = UserAPIKeyAuth(user_id="alpha", models=["allowed"])
    caller.mcp_admitted_user_subject = True
    caller.mcp_session_resource_server_id = "alpha-server"
    caller.mcp_toolset_id = "toolset-alpha"
    caller.mcp_source_team_rpm_limits = {"team": {"alpha-server": 2}}
    headers = {"x-caller": "alpha"}
    server_headers = {"alpha-server": {"authorization": "alpha-token"}}
    context = prepare_context(caller, raw_headers=headers, mcp_server_auth_headers=server_headers)

    caller.models.append("forbidden")
    caller.mcp_source_team_rpm_limits["team"]["alpha-server"] = 999
    headers["x-caller"] = "bravo"
    server_headers["alpha-server"]["authorization"] = "bravo-token"
    captured = context.user_api_key_auth
    assert captured is not None
    assert captured.models == ["allowed"]
    assert captured.mcp_admitted_user_subject is True
    assert captured.mcp_session_resource_server_id == "alpha-server"
    assert captured.mcp_toolset_id == "toolset-alpha"
    assert captured.mcp_source_team_rpm_limits == {"team": {"alpha-server": 2}}
    captured.models.append("also-forbidden")
    assert context.user_api_key_auth.models == ["allowed"]
    assert context.raw_headers == {"x-caller": "alpha"}
    assert context.mcp_server_auth_headers == {"alpha-server": {"authorization": "alpha-token"}}
    with pytest.raises(TypeError):
        context.raw_headers["x-caller"] = "changed"
    with pytest.raises(TypeError):
        context.mcp_server_auth_headers["alpha-server"]["authorization"] = "changed"
    with pytest.raises(FrozenInstanceError):
        context.client_ip = "untrusted"


def test_operation_context_preserves_missing_and_empty_inputs():
    missing = prepare_context()
    empty = prepare_context(mcp_servers=[], raw_headers={}, oauth2_headers={}, mcp_server_auth_headers={})
    assert missing.user_api_key_auth is None
    assert missing.mcp_servers is None
    assert missing.raw_headers is None
    assert missing.oauth2_headers is None
    assert missing.mcp_server_auth_headers is None
    assert empty.mcp_servers == ()
    assert empty.raw_headers == {}
    assert empty.oauth2_headers == {}
    assert empty.mcp_server_auth_headers == {}


def test_toolset_request_marker_cannot_be_supplied_by_caller_or_serialized():
    auth = UserAPIKeyAuth.model_validate({"user_id": "alpha", "mcp_toolset_id": "forged"})
    assert auth.mcp_toolset_id is None
    auth.mcp_toolset_id = "server-resolved"
    assert "mcp_toolset_id" not in auth.model_dump()
