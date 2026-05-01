import asyncio
import copy
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from pydantic import ValidationError as PydanticValidationError
from starlette.datastructures import Headers

import litellm
from litellm.proxy._types import AddTeamCallback, TeamCallbackMetadata, UserAPIKeyAuth
from litellm.proxy.litellm_pre_call_utils import (
    KeyAndTeamLoggingSettings,
    LiteLLMProxyRequestSetup,
    _apply_credential_overrides_from_model_config,
    _extract_credential_from_entry,
    _get_dynamic_logging_metadata,
    _get_enforced_params,
    _get_metadata_variable_name,
    _resolve_credential_from_model_config,
    _update_model_if_key_alias_exists,
    add_guardrails_from_policy_engine,
    add_litellm_data_to_request,
    check_if_token_is_service_account,
    clean_headers,
)
from litellm.types.utils import CredentialItem

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


def test_check_if_token_is_service_account():
    """
    Test that only keys with `service_account_id` in metadata are considered service accounts
    """
    # Test case 1: Service account token
    service_account_token = UserAPIKeyAuth(
        api_key="test-key", metadata={"service_account_id": "test-service-account"}
    )
    assert check_if_token_is_service_account(service_account_token) == True

    # Test case 2: Regular user token
    regular_token = UserAPIKeyAuth(api_key="test-key", metadata={})
    assert check_if_token_is_service_account(regular_token) == False

    # Test case 3: Token with other metadata
    other_metadata_token = UserAPIKeyAuth(
        api_key="test-key", metadata={"user_id": "test-user"}
    )
    assert check_if_token_is_service_account(other_metadata_token) == False


class TestGetMetadataVariableName:
    """Tests for _get_metadata_variable_name()"""

    def _make_request(self, path: str) -> MagicMock:
        request = MagicMock(spec=Request)
        request.url.path = path
        return request

    def test_returns_litellm_metadata_for_thread_routes(self):
        request = self._make_request("/v1/threads/thread_123/messages")
        assert _get_metadata_variable_name(request) == "litellm_metadata"

    def test_returns_litellm_metadata_for_assistant_routes(self):
        request = self._make_request("/v1/assistants/asst_123")
        assert _get_metadata_variable_name(request) == "litellm_metadata"

    def test_returns_litellm_metadata_for_batches_route(self):
        request = self._make_request("/v1/batches")
        assert _get_metadata_variable_name(request) == "litellm_metadata"

    def test_returns_litellm_metadata_for_messages_route(self):
        request = self._make_request("/v1/messages")
        assert _get_metadata_variable_name(request) == "litellm_metadata"

    def test_returns_litellm_metadata_for_files_route(self):
        request = self._make_request("/v1/files")
        assert _get_metadata_variable_name(request) == "litellm_metadata"

    def test_returns_metadata_for_chat_completions(self):
        request = self._make_request("/chat/completions")
        assert _get_metadata_variable_name(request) == "metadata"

    def test_returns_metadata_for_completions(self):
        request = self._make_request("/v1/completions")
        assert _get_metadata_variable_name(request) == "metadata"

    def test_returns_metadata_for_embeddings(self):
        request = self._make_request("/v1/embeddings")
        assert _get_metadata_variable_name(request) == "metadata"


def test_get_enforced_params_for_service_account_settings():
    """
    Test that service account enforced params are only added to service account keys
    """
    service_account_token = UserAPIKeyAuth(
        api_key="test-key", metadata={"service_account_id": "test-service-account"}
    )
    general_settings_with_service_account_settings = {
        "service_account_settings": {"enforced_params": ["metadata.service"]},
    }
    result = _get_enforced_params(
        general_settings=general_settings_with_service_account_settings,
        user_api_key_dict=service_account_token,
    )
    assert result == ["metadata.service"]

    regular_token = UserAPIKeyAuth(
        api_key="test-key", metadata={"enforced_params": ["user"]}
    )
    result = _get_enforced_params(
        general_settings=general_settings_with_service_account_settings,
        user_api_key_dict=regular_token,
    )
    assert result == ["user"]


@pytest.mark.parametrize(
    "general_settings, user_api_key_dict, expected_enforced_params",
    [
        (
            {"enforced_params": ["param1", "param2"]},
            UserAPIKeyAuth(
                api_key="test_api_key", user_id="test_user_id", org_id="test_org_id"
            ),
            ["param1", "param2"],
        ),
        (
            {"service_account_settings": {"enforced_params": ["param1", "param2"]}},
            UserAPIKeyAuth(
                api_key="test_api_key",
                user_id="test_user_id",
                org_id="test_org_id",
                metadata={"service_account_id": "test_service_account_id"},
            ),
            ["param1", "param2"],
        ),
        (
            {"service_account_settings": {"enforced_params": ["param1", "param2"]}},
            UserAPIKeyAuth(
                api_key="test_api_key",
                metadata={
                    "enforced_params": ["param3", "param4"],
                    "service_account_id": "test_service_account_id",
                },
            ),
            ["param1", "param2", "param3", "param4"],
        ),
    ],
)
def test_get_enforced_params(
    general_settings, user_api_key_dict, expected_enforced_params
):
    from litellm.proxy.litellm_pre_call_utils import _get_enforced_params

    enforced_params = _get_enforced_params(general_settings, user_api_key_dict)
    assert enforced_params == expected_enforced_params


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_parses_string_metadata():
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Simulate data with stringified metadata
    fake_metadata = {"generation_name": "gen123"}
    data = {"metadata": json.dumps(fake_metadata), "model": "gpt-3.5-turbo"}

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},  # this one can be a dict
        team_spend=0.0,
        team_max_budget=200.0,
    )

    # Call
    updated_data = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    # Assert
    litellm_metadata = updated_data.get("metadata", {})
    assert isinstance(litellm_metadata, dict)
    assert updated_data["metadata"]["generation_name"] == "gen123"


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_admin_injection_slots():
    """User-supplied user_api_key_metadata / user_api_key_team_metadata /
    _pipeline_managed_guardrails must be stripped from both metadata keys
    before the proxy writes its own admin-populated values. Otherwise a
    caller can shadow admin config via the non-`_metadata_variable_name`
    metadata key (e.g. litellm_metadata while the proxy writes to metadata).
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Caller tries to inject admin config into BOTH metadata keys
    attacker_admin_payload = {"disable_global_guardrails": True}
    data = {
        "model": "gpt-3.5-turbo",
        "metadata": {
            "user_api_key_metadata": attacker_admin_payload,
            "user_api_key_team_metadata": attacker_admin_payload,
            "_pipeline_managed_guardrails": ["evaded"],
        },
        "litellm_metadata": {
            "user_api_key_metadata": attacker_admin_payload,
            "user_api_key_team_metadata": attacker_admin_payload,
            "_pipeline_managed_guardrails": ["evaded"],
        },
    }

    real_admin_metadata = {"admin_flag": "from_proxy"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata=real_admin_metadata,
        team_metadata=real_admin_metadata,
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    # The key that matches `_metadata_variable_name` gets proxy-populated
    # with the real admin payload; the OTHER key must not retain the
    # attacker's injection.
    populated = updated["metadata"]
    assert populated["user_api_key_metadata"] == real_admin_metadata
    assert populated["user_api_key_team_metadata"] == real_admin_metadata
    assert "_pipeline_managed_guardrails" not in populated or populated[
        "_pipeline_managed_guardrails"
    ] != ["evaded"]

    other = updated.get("litellm_metadata") or {}
    assert other.get("user_api_key_metadata") in (None, {}, real_admin_metadata)
    assert other.get("user_api_key_team_metadata") in (None, {}, real_admin_metadata)
    assert "_pipeline_managed_guardrails" not in other


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_all_user_api_key_prefix_keys():
    """Strip must cover the full user_api_key_* family, not a hand-maintained
    list of 2-3 names. Proxy writes a dozen such fields (user_id, alias,
    spend, team_id, request_route, …) and an attacker populating any of them
    in the non-authoritative metadata key would otherwise forge identity /
    spend in audit logs and guardrails."""
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    attacker_injected = {
        "user_api_key_user_id": "victim",
        "user_api_key_alias": "admin-key",
        "user_api_key_spend": 0.0,
        "user_api_key_team_id": "victim-team",
        "user_api_key_end_user_id": "victim-user",
        "user_api_key_request_route": "/fake/route",
        "user_api_key_hash": "fake-hash",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "metadata": {**attacker_injected},
        "litellm_metadata": {**attacker_injected},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        user_id="real-user",
        metadata={},
        team_metadata={},
        spend=42.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    # The non-authoritative metadata dict must not retain ANY attacker-injected
    # user_api_key_* key.
    other = updated.get("litellm_metadata") or {}
    attacker_leaks = [k for k in other if k.startswith("user_api_key_")]
    assert attacker_leaks == [], f"Unexpected leaked keys: {attacker_leaks}"


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_string_metadata_does_not_crash():
    """Regression: pre-strip code that pre-populated data['metadata'][k]=v
    before the string-to-dict parse would crash on JSON-string metadata.
    The snapshot / strip / admin-population pipeline must survive metadata
    arriving as a string."""
    import json as _json

    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "multipart/form-data"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "metadata": _json.dumps({"generation_name": "test"}),
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    # Must not raise TypeError / AttributeError.
    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    # The parsed metadata should be a dict and the proxy snapshot body
    # should have been taken AFTER the strip (so no leaked user_api_key_*
    # from a raw string snapshot).
    assert isinstance(updated["metadata"], dict)
    assert updated["metadata"].get("generation_name") == "test"


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_proxy_server_request_body_is_post_strip():
    """Regression: proxy_server_request['body'] used to be snapshotted before
    the admin-slot strip, so standard_logging_object and spend-tracking
    readers saw attacker-injected payload. Snapshot must now be post-strip."""
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "metadata": {"user_api_key_user_id": "victim"},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        user_id="real-user",
        metadata={},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    snapshot_body = updated["proxy_server_request"]["body"]
    assert snapshot_body is not None
    snapshot_metadata = snapshot_body.get("metadata") or {}
    assert "user_api_key_user_id" not in snapshot_metadata or (
        snapshot_metadata["user_api_key_user_id"] != "victim"
    )


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_string_encoded_admin_injection():
    """Regression: metadata arriving as a JSON string (multipart/form-data or
    extra_body) must not bypass the admin-injection strip. The parse happens
    AFTER receipt, so the strip has to run after the parse, not before.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "multipart/form-data"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Attacker encodes an admin-injection payload inside a JSON string.
    attacker_payload = {
        "user_api_key_metadata": {"disable_global_guardrails": True},
        "user_api_key_team_metadata": {"disable_global_guardrails": True},
        "_pipeline_managed_guardrails": ["evaded"],
    }
    data = {
        "model": "gpt-3.5-turbo",
        "metadata": json.dumps(attacker_payload),
        "litellm_metadata": json.dumps(attacker_payload),
    }

    real_admin_metadata = {"admin_flag": "from_proxy"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata=real_admin_metadata,
        team_metadata=real_admin_metadata,
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    populated = updated["metadata"]
    # The real admin payload from user_api_key_dict wins.
    assert populated["user_api_key_metadata"] == real_admin_metadata
    assert populated["user_api_key_team_metadata"] == real_admin_metadata
    assert populated.get("_pipeline_managed_guardrails") != ["evaded"]

    other = updated.get("litellm_metadata") or {}
    # After the strip, litellm_metadata has no admin-injection slots.
    assert "user_api_key_metadata" not in other
    assert "user_api_key_team_metadata" not in other
    assert "_pipeline_managed_guardrails" not in other


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_user_control_fields():
    """Strip untrusted proxy-control fields before guardrails, logging, and headers read metadata."""
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    malicious_metadata = {
        "disable_global_guardrails": True,
        "opted_out_global_guardrails": ["pii"],
        "pillar_response_headers": {"set-cookie": "session=evil"},
        "_pillar_response_headers_trusted": True,
        "pillar_flagged": True,
        "pillar_scanners": {"jailbreak": True},
        "pillar_evidence": [{"evidence": "spoofed"}],
        "pillar_session_id_response": "spoofed-session",
        "applied_guardrails": ["spoofed"],
        "applied_policies": ["spoofed-policy"],
        "policy_sources": {"spoofed-policy": "request"},
        "_guardrail_pipelines": [{"name": "spoofed"}],
        "_pipeline_managed_guardrails": ["evaded"],
        "safe_user_metadata": "kept",
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hello"}],
        "mock_response": "free response",
        "mock_tool_calls": [{"id": "call_1"}],
        "disable_global_guardrails": True,
        "metadata": copy.deepcopy(malicious_metadata),
        "litellm_metadata": copy.deepcopy(malicious_metadata),
    }

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "mock_response" not in updated
    assert "mock_tool_calls" not in updated
    assert "disable_global_guardrails" not in updated

    stripped_keys = {
        "disable_global_guardrails",
        "opted_out_global_guardrails",
        "pillar_response_headers",
        "_pillar_response_headers_trusted",
        "pillar_flagged",
        "pillar_scanners",
        "pillar_evidence",
        "pillar_session_id_response",
        "applied_guardrails",
        "applied_policies",
        "policy_sources",
        "_guardrail_pipelines",
        "_pipeline_managed_guardrails",
    }
    for metadata_key in ("metadata", "litellm_metadata"):
        cleaned_metadata = updated.get(metadata_key) or {}
        for stripped_key in stripped_keys:
            assert stripped_key not in cleaned_metadata
        assert cleaned_metadata.get("safe_user_metadata") == "kept"

    requester_metadata = updated["metadata"]["requester_metadata"]
    for stripped_key in stripped_keys:
        assert stripped_key not in requester_metadata

    snapshot_body = updated["proxy_server_request"]["body"]
    assert "mock_response" not in snapshot_body
    assert "mock_tool_calls" not in snapshot_body
    assert "pillar_response_headers" not in snapshot_body["metadata"]


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_allows_client_mock_response_with_admin_opt_in():
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    updated = await add_litellm_data_to_request(
        data={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hello"}],
            "mock_response": "allowed mock",
            "mock_tool_calls": [{"id": "call_1"}],
        },
        request=request_mock,
        user_api_key_dict=UserAPIKeyAuth(
            api_key="hashed-key",
            metadata={"allow_client_mock_response": True},
        ),
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["mock_response"] == "allowed mock"
    assert updated["mock_tool_calls"] == [{"id": "call_1"}]


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_client_redaction_bypass_controls():
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "application/json",
        "litellm-disable-message-redaction": "true",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    original_turn_off_message_logging = litellm.turn_off_message_logging
    litellm.turn_off_message_logging = True
    try:
        updated = await add_litellm_data_to_request(
            data={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "hello"}],
                "turn_off_message_logging": False,
                "metadata": {"headers": {"litellm-disable-message-redaction": "true"}},
                "litellm_metadata": json.dumps(
                    {"headers": {"LiteLLM-Disable-Message-Redaction": "true"}}
                ),
            },
            request=request_mock,
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key"),
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )
    finally:
        litellm.turn_off_message_logging = original_turn_off_message_logging

    assert "turn_off_message_logging" not in updated
    assert "litellm-disable-message-redaction" not in {
        header.lower() for header in updated["metadata"]["headers"]
    }
    assert "litellm-disable-message-redaction" not in {
        header.lower()
        for header in updated["metadata"]["requester_metadata"].get("headers", {})
    }
    assert "litellm-disable-message-redaction" not in {
        header.lower() for header in updated["proxy_server_request"]["headers"]
    }
    assert "litellm-disable-message-redaction" not in {
        header.lower()
        for header in updated["proxy_server_request"]["body"]["metadata"]["headers"]
    }
    assert "litellm-disable-message-redaction" not in {
        header.lower()
        for header in (updated.get("litellm_metadata") or {}).get("headers", {})
    }


@pytest.mark.parametrize(
    "auth_kwargs",
    [
        {"metadata": {"allow_client_message_redaction_opt_out": True}},
        {"team_metadata": {"allow_client_message_redaction_opt_out": True}},
    ],
)
@pytest.mark.asyncio
async def test_add_litellm_data_to_request_allows_redaction_opt_out_with_admin_opt_in(
    auth_kwargs,
):
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "application/json",
        "litellm-disable-message-redaction": "true",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    original_turn_off_message_logging = litellm.turn_off_message_logging
    litellm.turn_off_message_logging = True
    try:
        updated = await add_litellm_data_to_request(
            data={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "hello"}],
                "turn_off_message_logging": False,
                "metadata": {"headers": {"litellm-disable-message-redaction": "true"}},
                "litellm_metadata": json.dumps(
                    {"headers": {"LiteLLM-Disable-Message-Redaction": "true"}}
                ),
            },
            request=request_mock,
            user_api_key_dict=UserAPIKeyAuth(api_key="hashed-key", **auth_kwargs),
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )
    finally:
        litellm.turn_off_message_logging = original_turn_off_message_logging

    assert updated["turn_off_message_logging"] is False
    assert "litellm-disable-message-redaction" in {
        header.lower() for header in updated["metadata"]["headers"]
    }
    assert "litellm-disable-message-redaction" in {
        header.lower()
        for header in updated["metadata"]["requester_metadata"].get("headers", {})
    }
    assert "litellm-disable-message-redaction" in {
        header.lower() for header in updated["proxy_server_request"]["headers"]
    }
    assert "litellm-disable-message-redaction" in {
        header.lower()
        for header in updated["proxy_server_request"]["body"]["metadata"]["headers"]
    }
    assert "litellm-disable-message-redaction" in {
        header.lower()
        for header in (updated.get("litellm_metadata") or {}).get("headers", {})
    }


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_ignores_x_litellm_tags_header_without_permission():
    """Regression: the `x-litellm-tags` header bypassed the body-metadata
    tag strip. Header tags must also be gated by `allow_client_tags`."""
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "application/json",
        "x-litellm-tags": "restricted-tier,victim-team",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {"model": "gpt-3.5-turbo"}

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "tags" not in (updated.get("metadata") or {})


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_ignores_root_level_tags_without_permission():
    """Regression: root-level `data["tags"]` bypassed the body-metadata
    tag strip. Root-level tags must also be gated by `allow_client_tags`."""
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "tags": ["restricted-tier", "victim-team"],
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "tags" not in (updated.get("metadata") or {})
    # Also ensure the root-level tags are removed. get_tags_from_request_body
    # reads request_body["tags"] directly, so leaving it in place would let
    # the policy engine see caller-supplied tags even after the metadata
    # strip.
    assert "tags" not in updated


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_honors_header_tags_when_opted_in():
    """When allow_client_tags=True, header-supplied tags flow through."""
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "application/json",
        "x-litellm-tags": "production,ab-test",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {"model": "gpt-3.5-turbo"}

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={"allow_client_tags": True},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["metadata"].get("tags") == ["production", "ab-test"]


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_strips_user_tags_without_permission():
    """Caller-supplied metadata.tags must be stripped when the key/team
    metadata does not opt in via allow_client_tags=True. Otherwise an
    attacker can reach restricted tag-routed deployments or attribute
    spend to a victim team's tag."""
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "metadata": {"tags": ["restricted-tier", "victim-team"]},
        "litellm_metadata": {"tags": ["also-stripped"]},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert "tags" not in (updated.get("metadata") or {})
    assert "tags" not in (updated.get("litellm_metadata") or {})


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_preserves_user_tags_when_key_opts_in():
    """When key.metadata.allow_client_tags=True, caller-supplied tags are
    preserved and reach the router."""
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "metadata": {"tags": ["opted-in-tag"]},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={"allow_client_tags": True},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["metadata"].get("tags") == ["opted-in-tag"]


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_preserves_user_tags_when_team_opts_in():
    """Team-level allow_client_tags is also honored (not just key-level)."""
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "metadata": {"tags": ["team-allowed"]},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={"allow_client_tags": True},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    assert updated["metadata"].get("tags") == ["team-allowed"]


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_user_spend_and_budget():
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={},
        team_metadata={},
        user_spend=150.0,
        user_max_budget=500.0,
    )

    updated_data = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    metadata = updated_data.get("metadata", {})
    assert metadata["user_api_key_user_spend"] == 150.0
    assert metadata["user_api_key_user_max_budget"] == 500.0


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_audio_transcription_multipart():
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup request mock for /v1/audio/transcriptions
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/audio/transcriptions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/audio/transcriptions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "multipart/form-data",
        "Authorization": "Bearer sk-1234",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Simulate multipart data (metadata as string)
    metadata_dict = {
        "tags": ["jobID:214590dsff09fds", "taskName:run_page_classification"]
    }
    stringified_metadata = json.dumps(metadata_dict)

    data = {
        "model": "fake-openai-endpoint",
        "metadata": stringified_metadata,  # Simulating multipart-form field
        "file": b"Fake audio bytes",
    }

    # Opt the key in to client-supplied tags so the parsed tags from the
    # JSON-string multipart body aren't stripped by the admin-injection strip.
    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-key",
        metadata={"allow_client_tags": True},
        team_metadata={},
        spend=0.0,
        max_budget=100.0,
        model_max_budget={},
        team_spend=0.0,
        team_max_budget=200.0,
    )

    updated_data = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=MagicMock(),
        general_settings={},
        version="test-version",
    )

    # Assert metadata was parsed correctly
    metadata_field = updated_data.get("metadata", {})
    litellm_metadata = updated_data.get("litellm_metadata", {})

    assert isinstance(metadata_field, dict)
    assert "tags" in metadata_field
    assert metadata_field["tags"] == [
        "jobID:214590dsff09fds",
        "taskName:run_page_classification",
    ]


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks():
    """
    Test that litellm_disabled_callbacks from key metadata is properly added to the request data.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key with disabled callbacks in metadata
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={"litellm_disabled_callbacks": ["langfuse", "langsmith", "datadog"]},
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that litellm_disabled_callbacks was added to the request data
    assert "litellm_disabled_callbacks" in result
    assert result["litellm_disabled_callbacks"] == ["langfuse", "langsmith", "datadog"]

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks_empty():
    """
    Test that litellm_disabled_callbacks is not added when it's empty.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key with empty disabled callbacks
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={"litellm_disabled_callbacks": []},
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that litellm_disabled_callbacks is not added when empty
    assert "litellm_disabled_callbacks" not in result

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks_not_present():
    """
    Test that litellm_disabled_callbacks is not added when it's not present in metadata.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key without disabled callbacks in metadata
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={},  # No litellm_disabled_callbacks
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that litellm_disabled_callbacks is not added when not present
    assert "litellm_disabled_callbacks" not in result

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks_invalid_type():
    """
    Test that litellm_disabled_callbacks is not added when it's not a list.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key with invalid disabled callbacks type
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={"litellm_disabled_callbacks": "not_a_list"},  # Should be a list
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that litellm_disabled_callbacks is not added when invalid type
    assert "litellm_disabled_callbacks" not in result

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


@pytest.mark.asyncio
async def test_add_litellm_data_to_request_disabled_callbacks_with_logging_settings():
    """
    Test that litellm_disabled_callbacks works correctly alongside logging settings.
    """
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup user API key with both logging settings and disabled callbacks
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test_api_key",
        user_id="test_user_id",
        org_id="test_org_id",
        metadata={
            "logging": [
                {
                    "callback_name": "langfuse",
                    "callback_type": "success",
                    "callback_vars": {},
                }
            ],
            "litellm_disabled_callbacks": ["langsmith", "datadog"],
        },
    )

    # Setup request data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    # Setup proxy config
    proxy_config = MagicMock()

    # Call add_litellm_data_to_request
    result = await add_litellm_data_to_request(
        data=data,
        request=request_mock,
        user_api_key_dict=user_api_key_dict,
        proxy_config=proxy_config,
    )

    # Verify that both logging settings and disabled callbacks are handled correctly
    assert "litellm_disabled_callbacks" in result
    assert result["litellm_disabled_callbacks"] == ["langsmith", "datadog"]

    # Verify that other data is still present
    assert "model" in result
    assert result["model"] == "gpt-3.5-turbo"
    assert "messages" in result


def test_key_dynamic_logging_settings():
    """
    Test KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings method with arize and langfuse callbacks
    """
    # Test with arize logging
    key_with_arize = UserAPIKeyAuth(
        api_key="test-key",
        metadata={"logging": [{"callback_name": "arize", "callback_type": "success"}]},
        team_metadata={},
    )
    result = KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings(key_with_arize)
    assert result == [{"callback_name": "arize", "callback_type": "success"}]

    # Test with langfuse logging
    key_with_langfuse = UserAPIKeyAuth(
        api_key="test-key",
        metadata={
            "logging": [{"callback_name": "langfuse", "callback_type": "success"}]
        },
        team_metadata={},
    )
    result = KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings(
        key_with_langfuse
    )
    assert result == [{"callback_name": "langfuse", "callback_type": "success"}]

    # Test with no logging metadata
    key_without_logging = UserAPIKeyAuth(
        api_key="test-key", metadata={}, team_metadata={}
    )
    result = KeyAndTeamLoggingSettings.get_key_dynamic_logging_settings(
        key_without_logging
    )
    assert result is None


def test_team_dynamic_logging_settings():
    """
    Test KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings method with arize and langfuse callbacks
    """
    # Test with arize team logging
    key_with_team_arize = UserAPIKeyAuth(
        api_key="test-key",
        metadata={},
        team_metadata={
            "logging": [{"callback_name": "arize", "callback_type": "failure"}]
        },
    )
    result = KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings(
        key_with_team_arize
    )
    assert result == [{"callback_name": "arize", "callback_type": "failure"}]

    # Test with langfuse team logging
    key_with_team_langfuse = UserAPIKeyAuth(
        api_key="test-key",
        metadata={},
        team_metadata={
            "logging": [{"callback_name": "langfuse", "callback_type": "success"}]
        },
    )
    result = KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings(
        key_with_team_langfuse
    )
    assert result == [{"callback_name": "langfuse", "callback_type": "success"}]

    # Test with no team logging metadata
    key_without_team_logging = UserAPIKeyAuth(
        api_key="test-key", metadata={}, team_metadata={}
    )
    result = KeyAndTeamLoggingSettings.get_team_dynamic_logging_settings(
        key_without_team_logging
    )
    assert result is None


def test_get_dynamic_logging_metadata_with_arize_team_logging():
    """
    Test _get_dynamic_logging_metadata function with arize team logging and dynamic parameters
    """
    # Setup user with arize team logging including callback_vars
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={},
        team_metadata={
            "logging": [
                {
                    "callback_name": "arize",
                    "callback_type": "success",
                    "callback_vars": {
                        "arize_api_key": "test_arize_api_key",
                        "arize_space_id": "test_arize_space_id",
                    },
                }
            ]
        },
    )

    # Mock proxy_config (not used in this test path since we have team dynamic logging)
    mock_proxy_config = MagicMock()

    # Call the function
    result = _get_dynamic_logging_metadata(
        user_api_key_dict=user_api_key_dict, proxy_config=mock_proxy_config
    )

    # Verify the result
    assert result is not None
    assert isinstance(result, TeamCallbackMetadata)
    assert result.success_callback == ["arize"]
    assert result.callback_vars is not None
    assert result.callback_vars["arize_api_key"] == "test_arize_api_key"
    assert result.callback_vars["arize_space_id"] == "test_arize_space_id"


def test_add_team_callback_rejects_env_reference():
    with pytest.raises(PydanticValidationError) as exc_info:
        AddTeamCallback(
            callback_name="langfuse",
            callback_type="success",
            callback_vars={
                "langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY_TEMP"
            },
        )

    assert "os.environ/" in str(exc_info.value)


def test_get_dynamic_logging_metadata_ignores_env_reference_from_key_metadata(
    monkeypatch,
):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY_TEMP", "server-side-secret")
    monkeypatch.setattr(
        litellm.utils,
        "get_secret",
        lambda *args, **kwargs: pytest.fail("get_secret should not be called"),
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={
            "logging": [
                {
                    "callback_name": "langfuse",
                    "callback_type": "success",
                    "callback_vars": {
                        "langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY_TEMP",
                    },
                }
            ]
        },
        team_metadata={},
    )

    result = _get_dynamic_logging_metadata(
        user_api_key_dict=user_api_key_dict, proxy_config=MagicMock()
    )

    assert result is None


def test_get_num_retries_from_request():
    """
    Test LiteLLMProxyRequestSetup._get_num_retries_from_request method
    """
    # Test case 1: Header is present with valid integer string
    headers_with_retries = {"x-litellm-num-retries": "3"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(
        headers_with_retries
    )
    assert result == 3

    # Test case 2: Header is not present
    headers_without_retries = {"Content-Type": "application/json"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(
        headers_without_retries
    )
    assert result is None

    # Test case 3: Empty headers dictionary
    empty_headers = {}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(empty_headers)
    assert result is None

    # Test case 4: Header present with zero value
    headers_with_zero = {"x-litellm-num-retries": "0"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(headers_with_zero)
    assert result == 0

    # Test case 5: Header present with large number
    headers_with_large_number = {"x-litellm-num-retries": "100"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(
        headers_with_large_number
    )
    assert result == 100

    # Test case 6: Multiple headers with num retries header
    headers_multiple = {
        "Content-Type": "application/json",
        "x-litellm-num-retries": "5",
        "Authorization": "Bearer token",
    }
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(headers_multiple)
    assert result == 5

    # Test case 7: Header present with invalid value (should raise ValueError when int() is called)
    headers_with_invalid = {"x-litellm-num-retries": "invalid"}
    with pytest.raises(ValueError):
        LiteLLMProxyRequestSetup._get_num_retries_from_request(headers_with_invalid)

    # Test case 8: Header present with float string (should raise ValueError when int() is called)
    headers_with_float = {"x-litellm-num-retries": "3.5"}
    with pytest.raises(ValueError):
        LiteLLMProxyRequestSetup._get_num_retries_from_request(headers_with_float)

    # Test case 9: Header present with negative number
    headers_with_negative = {"x-litellm-num-retries": "-1"}
    result = LiteLLMProxyRequestSetup._get_num_retries_from_request(
        headers_with_negative
    )
    assert result == -1


def test_add_user_api_key_auth_to_request_metadata():
    """
    Test that add_user_api_key_auth_to_request_metadata properly adds user API key authentication data to request metadata
    """
    # Setup test data
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hello"}],
        "litellm_metadata": {},  # This will be the metadata variable name
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="hashed-test-key-123",
        user_id="test-user-123",
        org_id="test-org-456",
        team_id="test-team-789",
        key_alias="test-key-alias",
        user_email="test@example.com",
        team_alias="test-team-alias",
        end_user_id="test-end-user-123",
        request_route="/chat/completions",
        end_user_max_budget=500.0,
    )

    metadata_variable_name = "litellm_metadata"

    # Call the function
    result = LiteLLMProxyRequestSetup.add_user_api_key_auth_to_request_metadata(
        data=data,
        user_api_key_dict=user_api_key_dict,
        _metadata_variable_name=metadata_variable_name,
    )

    # Verify the metadata was properly added
    metadata = result[metadata_variable_name]

    # Check that user API key information was added
    assert metadata["user_api_key_hash"] == "hashed-test-key-123"
    assert metadata["user_api_key_alias"] == "test-key-alias"
    assert metadata["user_api_key_team_id"] == "test-team-789"
    assert metadata["user_api_key_user_id"] == "test-user-123"
    assert metadata["user_api_key_org_id"] == "test-org-456"
    assert metadata["user_api_key_team_alias"] == "test-team-alias"
    assert metadata["user_api_key_end_user_id"] == "test-end-user-123"
    assert metadata["user_api_key_user_email"] == "test@example.com"
    assert metadata["user_api_key_request_route"] == "/chat/completions"

    # Check that the hashed API key was added
    assert metadata["user_api_key"] == "hashed-test-key-123"

    # Check that end user max budget was added
    assert metadata["user_api_end_user_max_budget"] == 500.0

    # Verify original data is preserved
    assert result["model"] == "gpt-3.5-turbo"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.parametrize(
    "data, model_group_settings, expected_headers_added",
    [
        # Test case 1: Model is in forward_client_headers_to_llm_api list
        (
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=["gpt-4"]),
            True,
        ),
        # Test case 2: Model is not in forward_client_headers_to_llm_api list
        (
            {"model": "claude-3", "messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=["gpt-4"]),
            False,
        ),
        # Test case 3: Model group settings is None
        (
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            None,
            False,
        ),
        # Test case 4: forward_client_headers_to_llm_api is None
        (
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=None),
            False,
        ),
        # Test case 5: Data has no model
        (
            {"messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=["gpt-4"]),
            False,
        ),
        # Test case 6: Model is None
        (
            {"model": None, "messages": [{"role": "user", "content": "Hello"}]},
            MagicMock(forward_client_headers_to_llm_api=["gpt-4"]),
            False,
        ),
    ],
)
def test_add_headers_to_llm_call_by_model_group(
    data, model_group_settings, expected_headers_added
):
    """
    Test LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group method

    This tests various scenarios:
    1. When model is in the forward_client_headers_to_llm_api list
    2. When model is not in the list
    3. When model_group_settings is None
    4. When forward_client_headers_to_llm_api is None
    5. When data has no model
    6. When model is None
    """
    import litellm

    # Setup test headers and user API key
    headers = {
        "Authorization": "Bearer token123",
        "User-Agent": "test-client/1.0",
        "X-Custom-Header": "custom-value",
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key", user_id="test-user", org_id="test-org"
    )

    # Mock the model_group_settings
    original_model_group_settings = getattr(litellm, "model_group_settings", None)
    litellm.model_group_settings = model_group_settings

    try:
        # Mock the add_headers_to_llm_call method to return expected headers
        expected_returned_headers = {
            "X-LiteLLM-User": "test-user",
            "X-LiteLLM-Org": "test-org",
        }

        with patch.object(
            LiteLLMProxyRequestSetup,
            "add_headers_to_llm_call",
            return_value=expected_returned_headers if expected_headers_added else {},
        ) as mock_add_headers:

            # Make a copy of original data to verify it's not mutated unexpectedly
            original_data = copy.deepcopy(data)

            # Call the method under test
            result = LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group(
                data=data, headers=headers, user_api_key_dict=user_api_key_dict
            )

            # Verify the result
            assert result is not None
            assert isinstance(result, dict)

            if expected_headers_added:
                # Verify that add_headers_to_llm_call was called
                mock_add_headers.assert_called_once_with(headers, user_api_key_dict)
                # Verify that headers were added to the data
                assert "headers" in result
                assert result["headers"] == expected_returned_headers
            else:
                # Verify that add_headers_to_llm_call was not called
                mock_add_headers.assert_not_called()
                # Verify that no headers were added
                assert "headers" not in result or result.get("headers") is None

            # Verify that original data fields are preserved
            for key, value in original_data.items():
                if key != "headers":  # headers might be added
                    assert result[key] == value

    finally:
        # Restore original model_group_settings
        litellm.model_group_settings = original_model_group_settings


def test_add_headers_to_llm_call_by_model_group_empty_headers_returned():
    """
    Test that when add_headers_to_llm_call returns empty dict, no headers are added to data
    """
    import litellm

    # Setup test data
    data = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
    headers = {"Authorization": "Bearer token123"}
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

    # Mock model_group_settings with model in the list
    mock_settings = MagicMock(forward_client_headers_to_llm_api=["gpt-4"])
    original_model_group_settings = getattr(litellm, "model_group_settings", None)
    litellm.model_group_settings = mock_settings

    try:
        with patch.object(
            LiteLLMProxyRequestSetup,
            "add_headers_to_llm_call",
            return_value={},  # Return empty dict
        ) as mock_add_headers:

            result = LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group(
                data=data, headers=headers, user_api_key_dict=user_api_key_dict
            )

            # Verify that add_headers_to_llm_call was called
            mock_add_headers.assert_called_once_with(headers, user_api_key_dict)

            # Verify that no headers were added since returned headers were empty
            assert "headers" not in result

            # Verify original data is preserved
            assert result["model"] == "gpt-4"
            assert result["messages"] == [{"role": "user", "content": "Hello"}]

    finally:
        # Restore original model_group_settings
        litellm.model_group_settings = original_model_group_settings


def test_add_headers_to_llm_call_by_model_group_existing_headers_in_data():
    """
    Test that existing headers in data are overwritten when new headers are added
    """
    import litellm

    # Setup test data with existing headers
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "headers": {"Existing-Header": "existing-value"},
    }
    headers = {"Authorization": "Bearer token123"}
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

    # Mock model_group_settings with model in the list
    mock_settings = MagicMock(forward_client_headers_to_llm_api=["gpt-4"])
    original_model_group_settings = getattr(litellm, "model_group_settings", None)
    litellm.model_group_settings = mock_settings

    try:
        new_headers = {"X-LiteLLM-User": "test-user"}

        with patch.object(
            LiteLLMProxyRequestSetup,
            "add_headers_to_llm_call",
            return_value=new_headers,
        ) as mock_add_headers:

            result = LiteLLMProxyRequestSetup.add_headers_to_llm_call_by_model_group(
                data=data, headers=headers, user_api_key_dict=user_api_key_dict
            )

            # Verify that add_headers_to_llm_call was called
            mock_add_headers.assert_called_once_with(headers, user_api_key_dict)

            # Verify that headers were overwritten
            assert "headers" in result
            assert result["headers"] == new_headers
            assert result["headers"] != {"Existing-Header": "existing-value"}

            # Verify original data is preserved
            assert result["model"] == "gpt-4"
            assert result["messages"] == [{"role": "user", "content": "Hello"}]

    finally:
        # Restore original model_group_settings
        litellm.model_group_settings = original_model_group_settings


import json
import time
from typing import Optional
from unittest.mock import AsyncMock

from fastapi.responses import Response

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.utils import ProxyLogging
from litellm.types.utils import StandardLoggingPayload


class TestCustomLogger(CustomLogger):
    def __init__(self):
        self.standard_logging_object: Optional[StandardLoggingPayload] = None
        super().__init__()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        print(f"SUCCESS CALLBACK CALLED! kwargs keys: {list(kwargs.keys())}")
        self.standard_logging_object = kwargs.get("standard_logging_object")
        print(f"Captured standard_logging_object: {self.standard_logging_object}")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        print(f"FAILURE CALLBACK CALLED! kwargs keys: {list(kwargs.keys())}")


@pytest.mark.asyncio
async def test_add_litellm_metadata_from_request_headers():
    """
    Test that add_litellm_metadata_from_request_headers properly adds litellm metadata from request headers,
    makes an LLM request using base_process_llm_request, sleeps for 3 seconds, and checks standard_logging_payload has spend_logs_metadata from headers

    Relevant issue: https://github.com/BerriAI/litellm/issues/14008
    """
    # Set up test logger
    litellm._turn_on_debug()
    test_logger = TestCustomLogger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [test_logger]

    try:
        # Prepare test data (ensure no streaming, add mock_response and api_key to route to litellm.acompletion)
        headers = {
            "x-litellm-spend-logs-metadata": '{"user_id": "12345", "project_id": "proj_abc", "request_type": "chat_completion", "timestamp": "2025-09-02T10:30:00Z"}'
        }
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "mock_response": "Hi",
            "api_key": "fake-key",
        }

        # Create mock request with headers
        mock_request = MagicMock(spec=Request)
        mock_request.headers = headers
        mock_request.url.path = "/chat/completions"

        # Create mock response
        mock_fastapi_response = MagicMock(spec=Response)

        # Create mock user API key dict
        mock_user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            org_id="test-org",
            metadata={"allow_client_mock_response": True},
        )

        # Create mock proxy logging object
        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)

        # Create async functions for the hooks
        async def mock_during_call_hook(*args, **kwargs):
            return None

        async def mock_pre_call_hook(*args, **kwargs):
            return data

        async def mock_post_call_success_hook(*args, **kwargs):
            # Return the response unchanged
            return kwargs.get("response", args[2] if len(args) > 2 else None)

        mock_proxy_logging_obj.during_call_hook = mock_during_call_hook
        mock_proxy_logging_obj.pre_call_hook = mock_pre_call_hook
        mock_proxy_logging_obj.post_call_success_hook = mock_post_call_success_hook

        # Create mock proxy config
        mock_proxy_config = MagicMock()

        # Create mock general settings
        general_settings = {}

        # Create mock select_data_generator with correct signature
        def mock_select_data_generator(
            response=None, user_api_key_dict=None, request_data=None
        ):
            async def mock_generator():
                yield "data: " + json.dumps(
                    {"choices": [{"delta": {"content": "Hello"}}]}
                ) + "\n\n"
                yield "data: [DONE]\n\n"

            return mock_generator()

        # Create the processor
        processor = ProxyBaseLLMRequestProcessing(data=data)

        # Call base_process_llm_request (it will use the mock_response="Hi" parameter)
        result = await processor.base_process_llm_request(
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
            route_type="acompletion",
            proxy_logging_obj=mock_proxy_logging_obj,
            general_settings=general_settings,
            proxy_config=mock_proxy_config,
            select_data_generator=mock_select_data_generator,
            llm_router=None,
            model="gpt-4",
            is_streaming_request=False,
        )

        # Sleep for 3 seconds to allow logging to complete
        await asyncio.sleep(3)

        # Check if standard_logging_object was set
        assert (
            test_logger.standard_logging_object is not None
        ), "standard_logging_object should be populated after LLM request"

        # Verify the logging object contains expected metadata
        standard_logging_obj = test_logger.standard_logging_object

        print(
            f"Standard logging object captured: {json.dumps(standard_logging_obj, indent=4, default=str)}"
        )

        SPEND_LOGS_METADATA = standard_logging_obj["metadata"]["spend_logs_metadata"]
        assert SPEND_LOGS_METADATA == dict(
            json.loads(headers["x-litellm-spend-logs-metadata"])
        ), "spend_logs_metadata should be the same as the headers"
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_anthropic_messages_standard_logging_object_matches_fixture():
    """
    Regression: /v1/messages calls routed to non-Anthropic providers should keep
    call_type=anthropic_messages in standard logging payloads.
    """
    litellm._turn_on_debug()
    test_logger = TestCustomLogger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [test_logger]

    try:
        data = {
            "model": "gemini/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "Hi."}],
            "stream": False,
            "mock_response": "Hello! How can I help you today?",
            "api_key": "fake-key",
            "max_tokens": 4096,
        }

        mock_request = MagicMock(spec=Request)
        mock_request.headers = {"user-agent": "PostmanRuntime/7.53.0"}
        mock_request.url.path = "/v1/messages"
        mock_request.url = MagicMock()
        mock_request.url.__str__.return_value = "http://localhost/v1/messages"
        mock_request.method = "POST"
        mock_request.query_params = {}
        mock_request.client = MagicMock()
        mock_request.client.host = "127.0.0.1"

        mock_fastapi_response = MagicMock(spec=Response)
        mock_user_api_key_dict = UserAPIKeyAuth(
            api_key="test-key",
            user_id="default_user_id",
            metadata={"allow_client_mock_response": True},
        )

        mock_proxy_logging_obj = MagicMock(spec=ProxyLogging)

        async def mock_during_call_hook(*args, **kwargs):
            return None

        async def mock_pre_call_hook(*args, **kwargs):
            return data

        async def mock_post_call_success_hook(*args, **kwargs):
            return kwargs.get("response", args[2] if len(args) > 2 else None)

        mock_proxy_logging_obj.during_call_hook = mock_during_call_hook
        mock_proxy_logging_obj.pre_call_hook = mock_pre_call_hook
        mock_proxy_logging_obj.post_call_success_hook = mock_post_call_success_hook

        processor = ProxyBaseLLMRequestProcessing(data=data)
        await processor.base_process_llm_request(
            request=mock_request,
            fastapi_response=mock_fastapi_response,
            user_api_key_dict=mock_user_api_key_dict,
            route_type="anthropic_messages",
            proxy_logging_obj=mock_proxy_logging_obj,
            general_settings={},
            proxy_config=MagicMock(),
            select_data_generator=None,
            llm_router=None,
            model="gemini/gemini-2.5-flash",
            is_streaming_request=False,
        )

        await asyncio.sleep(3)

        assert test_logger.standard_logging_object is not None
        actual = test_logger.standard_logging_object

        expected = {
            "call_type": "anthropic_messages",
            "status": "success",
            "model": "gemini/gemini-2.5-flash",
        }

        # Compare only stable fields from the saved proxy log snapshot.
        actual_projection = {
            "call_type": actual.get("call_type"),
            "status": actual.get("status"),
            "model": actual.get("model"),
        }
        assert actual_projection == expected
        assert actual.get("call_type") == "anthropic_messages"
    finally:
        litellm.callbacks = original_callbacks


def test_add_litellm_metadata_from_request_headers_x_litellm_trace_id_sets_chain_id():
    """x-litellm-trace-id sets both metadata and top-level litellm_session_id/litellm_trace_id for call chaining."""
    headers = {"x-litellm-trace-id": "foo"}
    data = {"metadata": {}}
    LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
        headers=headers, data=data, _metadata_variable_name="metadata"
    )
    assert data["metadata"]["trace_id"] == "foo"
    assert data["metadata"]["session_id"] == "foo"
    assert data["litellm_session_id"] == "foo"
    assert data["litellm_trace_id"] == "foo"


def test_add_litellm_metadata_from_request_headers_x_litellm_session_id_sets_chain_id():
    """x-litellm-session-id sets both metadata and top-level litellm_session_id/litellm_trace_id for call chaining."""
    headers = {"x-litellm-session-id": "bar"}
    data = {"metadata": {}}
    LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
        headers=headers, data=data, _metadata_variable_name="metadata"
    )
    assert data["metadata"]["trace_id"] == "bar"
    assert data["metadata"]["session_id"] == "bar"
    assert data["litellm_session_id"] == "bar"
    assert data["litellm_trace_id"] == "bar"


def test_add_litellm_metadata_from_request_headers_both_headers_trace_id_precedence():
    """When both x-litellm-trace-id and x-litellm-session-id are present, trace-id takes precedence for chain_id."""
    headers = {
        "x-litellm-trace-id": "trace-value",
        "x-litellm-session-id": "session-value",
    }
    data = {"metadata": {}}
    LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
        headers=headers, data=data, _metadata_variable_name="metadata"
    )
    assert data["metadata"]["trace_id"] == "trace-value"
    assert data["metadata"]["session_id"] == "trace-value"
    assert data["litellm_session_id"] == "trace-value"
    assert data["litellm_trace_id"] == "trace-value"


def test_add_litellm_metadata_from_request_headers_generic_session_id_header():
    """A generic x-<vendor>-session-id header is used when no explicit litellm header is set."""
    headers = {"x-claude-code-session-id": "e96634a3-fa28-4083-b354-55542e2dca01"}
    data = {"metadata": {}}
    LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
        headers=headers, data=data, _metadata_variable_name="metadata"
    )
    assert data["metadata"]["session_id"] == "e96634a3-fa28-4083-b354-55542e2dca01"
    assert data["litellm_session_id"] == "e96634a3-fa28-4083-b354-55542e2dca01"
    assert data["litellm_trace_id"] == "e96634a3-fa28-4083-b354-55542e2dca01"


def test_add_litellm_metadata_from_request_headers_explicit_header_beats_generic():
    """Explicit x-litellm-trace-id wins over a generic x-*-session-id header."""
    headers = {
        "x-litellm-trace-id": "explicit-trace-id-value",
        "x-claude-code-session-id": "e96634a3-fa28-4083-b354-55542e2dca01",
    }
    data = {"metadata": {}}
    LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
        headers=headers, data=data, _metadata_variable_name="metadata"
    )
    assert data["litellm_session_id"] == "explicit-trace-id-value"
    assert data["litellm_trace_id"] == "explicit-trace-id-value"


def test_get_chain_id_from_headers_generic_vendor_session_id():
    """get_chain_id_from_headers picks up any x-<vendor>-session-id with a valid value."""
    from litellm.proxy.litellm_pre_call_utils import get_chain_id_from_headers

    assert (
        get_chain_id_from_headers(
            {"x-claude-code-session-id": "e96634a3-fa28-4083-b354-55542e2dca01"}
        )
        == "e96634a3-fa28-4083-b354-55542e2dca01"
    )
    # Short / non-alphanumeric values should be ignored
    assert get_chain_id_from_headers({"x-foo-session-id": "short"}) is None
    assert get_chain_id_from_headers({"x-foo-session-id": "has spaces!!"}) is None
    # Explicit headers still take precedence
    assert (
        get_chain_id_from_headers(
            {
                "x-litellm-trace-id": "explicit-id-value",
                "x-claude-code-session-id": "e96634a3-fa28-4083-b354-55542e2dca01",
            }
        )
        == "explicit-id-value"
    )


def test_get_internal_user_header_from_mapping_returns_expected_header():
    mappings = [
        {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"},
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
    ]

    header_name = LiteLLMProxyRequestSetup.get_internal_user_header_from_mapping(
        mappings
    )
    assert header_name == "X-OpenWebUI-User-Id"


def test_get_internal_user_header_from_mapping_none_when_absent():
    mappings = [
        {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"}
    ]
    header_name = LiteLLMProxyRequestSetup.get_internal_user_header_from_mapping(
        mappings
    )
    assert header_name is None

    single = {"header_name": "X-Only-Customer", "litellm_user_role": "customer"}
    header_name = LiteLLMProxyRequestSetup.get_internal_user_header_from_mapping(single)
    assert header_name is None


def test_add_internal_user_from_user_mapping_sets_user_id_when_header_present():
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")
    headers = {"X-OpenWebUI-User-Id": "internal-user-123"}
    general_settings = {
        "user_header_mappings": [
            {
                "header_name": "X-OpenWebUI-User-Id",
                "litellm_user_role": "internal_user",
            },
            {"header_name": "X-OpenWebUI-User-Email", "litellm_user_role": "customer"},
        ]
    }

    result = LiteLLMProxyRequestSetup.add_internal_user_from_user_mapping(
        general_settings, user_api_key_dict, headers
    )

    assert result is user_api_key_dict
    assert user_api_key_dict.user_id == "internal-user-123"


def test_add_internal_user_from_user_mapping_no_header_or_mapping_returns_unchanged():
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key")

    result = LiteLLMProxyRequestSetup.add_internal_user_from_user_mapping(
        None, user_api_key_dict, {"X-OpenWebUI-User-Id": "abc"}
    )
    assert result is user_api_key_dict
    assert user_api_key_dict.user_id is None

    general_settings = {
        "user_header_mappings": [
            {"header_name": "X-OpenWebUI-User-Id", "litellm_user_role": "internal_user"}
        ]
    }
    result = LiteLLMProxyRequestSetup.add_internal_user_from_user_mapping(
        general_settings, user_api_key_dict, {"Other": "value"}
    )
    assert result is user_api_key_dict
    assert user_api_key_dict.user_id is None


def test_get_sanitized_user_information_from_key_includes_guardrails_metadata():
    """
    Test that get_sanitized_user_information_from_key includes guardrails field from key metadata in the returned payload
    """
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key-hash",
        key_alias="test-alias",
        user_id="test-user",
        metadata={"guardrails": ["presidio", "aporia"], "other_field": "value"},
    )

    result = LiteLLMProxyRequestSetup.get_sanitized_user_information_from_key(
        user_api_key_dict=user_api_key_dict
    )

    assert result["user_api_key_auth_metadata"] is not None
    assert "guardrails" in result["user_api_key_auth_metadata"]
    assert result["user_api_key_auth_metadata"]["guardrails"] == ["presidio", "aporia"]
    assert result["user_api_key_auth_metadata"]["other_field"] == "value"


@pytest.mark.asyncio
async def test_team_guardrails_append_to_key_guardrails():
    """
    Test that team guardrails are appended to key guardrails instead of overriding them.
    Team guardrails should only be added if they are not already present in key guardrails.
    """
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={"guardrails": ["key-guardrail-1", "key-guardrail-2"]},
        team_metadata={"guardrails": ["team-guardrail-1", "key-guardrail-1"]},
    )

    with patch("litellm.proxy.utils._premium_user_check"):
        updated_data = await add_litellm_data_to_request(
            data=data,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

    metadata = updated_data.get("metadata", {})
    guardrails = metadata.get("guardrails", [])

    assert "key-guardrail-1" in guardrails
    assert "key-guardrail-2" in guardrails
    assert "team-guardrail-1" in guardrails
    assert guardrails.count("key-guardrail-1") == 1


@pytest.mark.asyncio
async def test_request_guardrails_do_not_override_key_guardrails():
    """
    Test that request-level guardrails do not override key-level guardrails.

    Key guardrails should be preserved when request contains guardrails (including empty array).
    """
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={"guardrails": ["key-guardrail-1"]},
        team_metadata={},
    )

    # Test case: Request with empty guardrails should not result in empty guardrails
    data_with_empty = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
        "guardrails": [],
    }

    with patch("litellm.proxy.utils._premium_user_check"):
        updated_data_empty = await add_litellm_data_to_request(
            data=data_with_empty,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

    _metadata = updated_data_empty.get("metadata", {})
    requested_guardrails = _metadata.get("guardrails", [])

    assert "guardrails" not in updated_data_empty
    assert "key-guardrail-1" in requested_guardrails
    assert len(requested_guardrails) == 1


@pytest.mark.asyncio
async def test_project_guardrails_merge_with_key_and_team():
    """
    Test that project guardrails are merged with key and team guardrails (union semantics).
    All three levels should contribute to the final guardrails list without duplicates.
    """
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={"guardrails": ["key-guardrail-1"]},
        team_metadata={"guardrails": ["team-guardrail-1", "key-guardrail-1"]},
        project_metadata={"guardrails": ["project-guardrail-1", "team-guardrail-1"]},
    )

    with patch("litellm.proxy.utils._premium_user_check"):
        updated_data = await add_litellm_data_to_request(
            data=data,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

    metadata = updated_data.get("metadata", {})
    guardrails = metadata.get("guardrails", [])

    # All three sources contribute
    assert "key-guardrail-1" in guardrails
    assert "team-guardrail-1" in guardrails
    assert "project-guardrail-1" in guardrails
    # No duplicates
    assert guardrails.count("key-guardrail-1") == 1
    assert guardrails.count("team-guardrail-1") == 1


@pytest.mark.asyncio
async def test_project_guardrails_only():
    """
    Test that project guardrails work when key and team have no guardrails configured.
    """
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/chat/completions"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/chat/completions"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {"Content-Type": "application/json"}
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test"}],
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        metadata={},
        team_metadata={},
        project_metadata={"guardrails": ["project-guardrail-1", "project-guardrail-2"]},
    )

    with patch("litellm.proxy.utils._premium_user_check"):
        updated_data = await add_litellm_data_to_request(
            data=data,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

    metadata = updated_data.get("metadata", {})
    guardrails = metadata.get("guardrails", [])

    assert "project-guardrail-1" in guardrails
    assert "project-guardrail-2" in guardrails
    assert len(guardrails) == 2


def test_update_model_if_key_alias_exists():
    """
    Test that _update_model_if_key_alias_exists properly updates the model when a key alias exists.
    """
    # Test case 1: Key alias exists and matches model
    data = {"model": "modelAlias", "messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={"modelAlias": "xai/grok-4-fast-non-reasoning"},
    )
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] == "xai/grok-4-fast-non-reasoning"

    # Test case 2: Key alias doesn't exist
    data = {
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={"modelAlias": "xai/grok-4-fast-non-reasoning"},
    )
    original_model = data["model"]
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] == original_model  # Should remain unchanged

    # Test case 3: Model is None
    data = {"model": None, "messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={"modelAlias": "xai/grok-4-fast-non-reasoning"},
    )
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] is None  # Should remain None

    # Test case 4: Model key doesn't exist in data
    data = {"messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={"modelAlias": "xai/grok-4-fast-non-reasoning"},
    )
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert "model" not in data  # Should not add model if it doesn't exist

    # Test case 5: Multiple aliases, matching one
    data = {"model": "alias1", "messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        aliases={
            "alias1": "model1",
            "alias2": "model2",
            "alias3": "model3",
        },
    )
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] == "model1"

    # Test case 6: Empty aliases dict
    data = {"model": "modelAlias", "messages": [{"role": "user", "content": "Hello"}]}
    user_api_key_dict = UserAPIKeyAuth(api_key="test-key", aliases={})
    original_model = data["model"]
    _update_model_if_key_alias_exists(data=data, user_api_key_dict=user_api_key_dict)
    assert data["model"] == original_model  # Should remain unchanged


@pytest.mark.asyncio
async def test_embedding_header_forwarding_with_model_group():
    """
    Test that headers are properly forwarded for embedding requests when
    forward_client_headers_to_llm_api is configured for the model group.

    This test verifies the fix for embedding endpoints not forwarding headers
    similar to how chat completion endpoints do.
    """
    import importlib

    import litellm.proxy.litellm_pre_call_utils as pre_call_utils_module

    # Reload the module to ensure it has a fresh reference to litellm
    # This is necessary because conftest.py reloads litellm at module scope,
    # which can cause the module's litellm reference to become stale
    importlib.reload(pre_call_utils_module)

    # Re-import the function after reload to get the fresh version
    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request

    # Setup mock request for embeddings
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/embeddings"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/embeddings"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "application/json",
        "X-Custom-Header": "custom-value",
        "X-Request-ID": "test-request-123",
        "Authorization": "Bearer sk-test-key",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup embedding request data
    data = {
        "model": "local-openai/text-embedding-3-small",
        "input": ["Text to embed"],
    }

    # Setup user API key
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        org_id="test-org",
    )

    # Mock model_group_settings to enable header forwarding for the model
    # Use string-based patch to ensure we patch the current sys.modules['litellm']
    # This avoids issues with module reloading during parallel test execution
    mock_settings = MagicMock(forward_client_headers_to_llm_api=["local-openai/*"])
    with patch("litellm.model_group_settings", mock_settings):
        # Call add_litellm_data_to_request which includes header forwarding logic
        updated_data = await add_litellm_data_to_request(
            data=data,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

        # Verify that headers were added to the request data
        assert "headers" in updated_data, "Headers should be added to embedding request"

        # Verify that only x- prefixed headers (except x-stainless) were forwarded
        forwarded_headers = updated_data["headers"]
        assert (
            "X-Custom-Header" in forwarded_headers
        ), "X-Custom-Header should be forwarded"
        assert forwarded_headers["X-Custom-Header"] == "custom-value"
        assert "X-Request-ID" in forwarded_headers, "X-Request-ID should be forwarded"
        assert forwarded_headers["X-Request-ID"] == "test-request-123"

        # Verify that authorization header was NOT forwarded (sensitive header)
        assert (
            "Authorization" not in forwarded_headers
        ), "Authorization header should not be forwarded"

        # Verify that Content-Type was NOT forwarded (doesn't start with x-)
        assert (
            "Content-Type" not in forwarded_headers
        ), "Content-Type should not be forwarded"

        # Verify original data fields are preserved
        assert updated_data["model"] == "local-openai/text-embedding-3-small"
        assert updated_data["input"] == ["Text to embed"]


@pytest.mark.asyncio
async def test_embedding_header_forwarding_without_model_group_config():
    """
    Test that headers are NOT forwarded for embedding requests when
    the model is not in the forward_client_headers_to_llm_api list.
    """
    import litellm

    # Setup mock request for embeddings
    request_mock = MagicMock(spec=Request)
    request_mock.url.path = "/v1/embeddings"
    request_mock.url = MagicMock()
    request_mock.url.__str__.return_value = "http://localhost/v1/embeddings"
    request_mock.method = "POST"
    request_mock.query_params = {}
    request_mock.headers = {
        "Content-Type": "application/json",
        "X-Custom-Header": "custom-value",
    }
    request_mock.client = MagicMock()
    request_mock.client.host = "127.0.0.1"

    # Setup embedding request data with a model NOT in the forward list
    data = {
        "model": "text-embedding-ada-002",
        "input": ["Text to embed"],
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
    )

    # Mock model_group_settings with a different model in the forward list
    mock_settings = MagicMock(forward_client_headers_to_llm_api=["gpt-4", "claude-*"])
    original_model_group_settings = getattr(litellm, "model_group_settings", None)
    litellm.model_group_settings = mock_settings

    try:
        updated_data = await add_litellm_data_to_request(
            data=data,
            request=request_mock,
            user_api_key_dict=user_api_key_dict,
            proxy_config=MagicMock(),
            general_settings={},
            version="test-version",
        )

        # Verify that headers were NOT added since model is not in forward list
        assert (
            "headers" not in updated_data or updated_data.get("headers") is None
        ), "Headers should not be forwarded for models not in forward_client_headers_to_llm_api list"

        # Verify original data fields are preserved
        assert updated_data["model"] == "text-embedding-ada-002"
        assert updated_data["input"] == ["Text to embed"]

    finally:
        # Restore original model_group_settings
        litellm.model_group_settings = original_model_group_settings


@pytest.mark.asyncio
async def test_add_guardrails_from_policy_engine():
    """
    Test that add_guardrails_from_policy_engine adds guardrails from matching policies
    and tracks applied policies in metadata.
    """
    from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.types.proxy.policy_engine import (
        Policy,
        PolicyAttachment,
        PolicyGuardrails,
    )

    # Setup test data
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "metadata": {},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_alias="healthcare-team",
        key_alias="my-key",
    )

    # Setup mock policies in the registry (policies define WHAT guardrails to apply)
    policy_registry = get_policy_registry()
    policy_registry._policies = {
        "global-baseline": Policy(
            guardrails=PolicyGuardrails(add=["pii_blocker"]),
        ),
        "healthcare": Policy(
            guardrails=PolicyGuardrails(add=["hipaa_audit"]),
        ),
    }
    policy_registry._initialized = True

    # Setup attachments in the attachment registry (attachments define WHERE policies apply)
    attachment_registry = get_attachment_registry()
    attachment_registry._attachments = [
        PolicyAttachment(policy="global-baseline", scope="*"),  # applies to all
        PolicyAttachment(
            policy="healthcare", teams=["healthcare-team"]
        ),  # applies to healthcare team
    ]
    attachment_registry._initialized = True

    # Call the function
    await add_guardrails_from_policy_engine(
        data=data,
        metadata_variable_name="metadata",
        user_api_key_dict=user_api_key_dict,
    )

    # Verify guardrails were added
    assert "guardrails" in data["metadata"]
    assert "pii_blocker" in data["metadata"]["guardrails"]
    assert "hipaa_audit" in data["metadata"]["guardrails"]

    # Verify applied policies were tracked
    assert "applied_policies" in data["metadata"]
    assert "global-baseline" in data["metadata"]["applied_policies"]
    assert "healthcare" in data["metadata"]["applied_policies"]

    # Clean up registries
    policy_registry._policies = {}
    policy_registry._initialized = False
    attachment_registry._attachments = []
    attachment_registry._initialized = False


@pytest.mark.asyncio
async def test_add_guardrails_from_policy_engine_accepts_dynamic_policies_and_pops_from_data():
    """
    Test that add_guardrails_from_policy_engine accepts dynamic 'policies' from the request body
    and removes them to prevent forwarding to the LLM provider.

    This is critical because 'policies' is a LiteLLM proxy-specific parameter that should
    not be sent to the actual LLM API (e.g., OpenAI, Anthropic, etc.).
    """
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry

    # Setup test data with 'policies' in the request body
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "policies": [
            "PII-POLICY-GLOBAL",
            "HIPAA-POLICY",
        ],  # Dynamic policies - should be accepted and removed
        "metadata": {},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_alias="test-team",
        key_alias="test-key",
    )

    # Initialize empty policy registry (we're just testing the accept and pop behavior)
    policy_registry = get_policy_registry()
    policy_registry._policies = {}
    policy_registry._initialized = False

    # Call the function - should accept dynamic policies and not raise an error
    await add_guardrails_from_policy_engine(
        data=data,
        metadata_variable_name="metadata",
        user_api_key_dict=user_api_key_dict,
    )

    # Verify that 'policies' was removed from the request body
    assert (
        "policies" not in data
    ), "'policies' should be removed from request body to prevent forwarding to LLM provider"

    # Verify that other fields are preserved
    assert "model" in data
    assert data["model"] == "gpt-4"
    assert "messages" in data
    assert data["messages"] == [{"role": "user", "content": "Hello"}]
    assert "metadata" in data


@pytest.mark.asyncio
async def test_add_guardrails_from_policy_engine_policy_version_by_id():
    """
    Test that add_guardrails_from_policy_engine executes a specific policy version
    when policy_<uuid> is passed in the request body.
    """
    from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.types.proxy.policy_engine import Policy, PolicyGuardrails

    policy_version_uuid = "12345678-1234-5678-1234-567812345678"
    policy_version_ref = f"policy_{policy_version_uuid}"

    # Policy from the specific version (e.g. published) - different guardrail than production
    published_version_policy = Policy(
        guardrails=PolicyGuardrails(add=["published_version_guardrail"]),
    )

    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "policies": [policy_version_ref],
        "metadata": {},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_alias="test-team",
        key_alias="test-key",
    )

    policy_registry = get_policy_registry()
    policy_registry._policies = {}
    policy_registry._initialized = True

    attachment_registry = get_attachment_registry()
    attachment_registry._attachments = []
    attachment_registry._initialized = True

    with patch.object(
        policy_registry,
        "get_policy_by_id_for_request",
        return_value=("test-policy-from-version", published_version_policy),
    ):
        await add_guardrails_from_policy_engine(
            data=data,
            metadata_variable_name="metadata",
            user_api_key_dict=user_api_key_dict,
        )

    # Verify guardrails from the specific version were applied
    assert "metadata" in data
    assert "guardrails" in data["metadata"]
    assert "published_version_guardrail" in data["metadata"]["guardrails"]
    assert "policies" not in data

    # Clean up
    policy_registry._policies = {}
    policy_registry._initialized = False


@pytest.mark.asyncio
async def test_bearer_token_not_in_debug_logs():
    """
    E2E regression test for the client-reported JWT leak.

    Calls add_litellm_data_to_request with a Bearer token in the request
    headers and captures all debug log output. Asserts the raw token never
    appears in any log message — covering the exact paths the client reported:
      - "Request Headers: ..."
      - "receiving data: ..."
      - "[PROXY] returned data from litellm_pre_call_utils: ..."
    """
    import logging
    from io import StringIO

    from litellm.proxy.litellm_pre_call_utils import add_litellm_data_to_request
    from litellm.proxy.proxy_server import ProxyConfig

    secret_token = (
        "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.fakesignature"
    )

    mock_request = MagicMock(spec=Request)
    mock_request.headers = {
        "authorization": f"Bearer {secret_token}",
        "content-type": "application/json",
    }
    mock_request.url = MagicMock()
    mock_request.url.__str__ = lambda self: "http://localhost:4000/v1/chat/completions"
    mock_request.method = "POST"
    mock_request.query_params = {}

    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "hi"}],
    }

    user_api_key_dict = UserAPIKeyAuth(api_key="sk-1234")

    # Capture all debug log output from the proxy logger
    log_capture = StringIO()
    log_handler = logging.StreamHandler(log_capture)
    log_handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("LiteLLM Proxy")
    logger.addHandler(log_handler)
    original_level = logger.level
    logger.setLevel(logging.DEBUG)

    try:
        with (
            patch("litellm.proxy.proxy_server.llm_router", None),
            patch("litellm.proxy.proxy_server.premium_user", True),
        ):
            await add_litellm_data_to_request(
                data=data,
                request=mock_request,
                user_api_key_dict=user_api_key_dict,
                proxy_config=ProxyConfig(),
                general_settings={},
            )
    finally:
        logger.removeHandler(log_handler)
        logger.setLevel(original_level)

    log_output = log_capture.getvalue()
    assert secret_token not in log_output, (
        f"Bearer token leaked in debug logs. "
        f"Found token in log output:\n{log_output[:500]}"
    )


# ============================================================================
# Tests for credential overrides from model_config (team/project metadata)
# ============================================================================


@pytest.fixture()
def setup_test_credentials():
    """Populate litellm.credential_list with test credentials and enable feature flag, clean up after."""
    original = litellm.credential_list[:]
    original_flag = litellm.enable_model_config_credential_overrides
    litellm.enable_model_config_credential_overrides = True
    litellm.credential_list.extend(
        [
            CredentialItem(
                credential_name="hotel-azure-eastus",
                credential_info={},
                credential_values={
                    "api_base": "https://hotel-eastus.openai.azure.com/",
                    "api_key": "key-hotel-eastus",
                },
            ),
            CredentialItem(
                credential_name="hotel-azure-westus",
                credential_info={},
                credential_values={
                    "api_base": "https://hotel-westus.openai.azure.com/",
                    "api_key": "key-hotel-westus",
                },
            ),
            CredentialItem(
                credential_name="hotel-rec-azure",
                credential_info={},
                credential_values={
                    "api_base": "https://hotel-rec-app.openai.azure.com/",
                    "api_key": "key-hotel-rec",
                },
            ),
            CredentialItem(
                credential_name="hotel-rec-vision",
                credential_info={},
                credential_values={
                    "api_base": "https://hotel-rec-vision.openai.azure.com/",
                    "api_key": "key-hotel-rec-vision",
                    "api_version": "2024-06-01",
                },
            ),
            CredentialItem(
                credential_name="flight-azure-centralus",
                credential_info={},
                credential_values={
                    "api_base": "https://flight-centralus.openai.azure.com/",
                    "api_key": "key-flight-centralus",
                },
            ),
        ]
    )
    yield
    litellm.credential_list[:] = original
    litellm.enable_model_config_credential_overrides = original_flag


# --- Unit tests for _extract_credential_from_entry ---


def test_extract_credential_from_entry_azure():
    entry = {"azure": {"litellm_credentials": "my-cred"}}
    assert _extract_credential_from_entry(entry) == "my-cred"


def test_extract_credential_from_entry_no_credential():
    entry = {"azure": {"some_other_key": "value"}}
    assert _extract_credential_from_entry(entry) is None


def test_extract_credential_from_entry_empty():
    assert _extract_credential_from_entry({}) is None


def test_extract_credential_from_entry_non_dict_value():
    entry = {"azure": "not-a-dict"}
    assert _extract_credential_from_entry(entry) is None


def test_extract_credential_from_entry_non_dict_entry():
    """Non-dict entry (e.g. string) should return None, not crash."""
    assert _extract_credential_from_entry("my-cred-name") is None
    assert _extract_credential_from_entry(["a", "list"]) is None
    assert _extract_credential_from_entry(42) is None


# --- Unit tests for _resolve_credential_from_model_config ---


def test_resolve_project_model_specific_wins():
    project_config = {
        "gpt-4": {"azure": {"litellm_credentials": "proj-gpt4"}},
        "defaultconfig": {"azure": {"litellm_credentials": "proj-default"}},
    }
    team_config = {
        "gpt-4": {"azure": {"litellm_credentials": "team-gpt4"}},
        "defaultconfig": {"azure": {"litellm_credentials": "team-default"}},
    }
    result = _resolve_credential_from_model_config("gpt-4", project_config, team_config)
    assert result == "proj-gpt4"


def test_resolve_project_default_wins_over_team():
    project_config = {
        "defaultconfig": {"azure": {"litellm_credentials": "proj-default"}},
    }
    team_config = {
        "gpt-4": {"azure": {"litellm_credentials": "team-gpt4"}},
        "defaultconfig": {"azure": {"litellm_credentials": "team-default"}},
    }
    result = _resolve_credential_from_model_config("gpt-4", project_config, team_config)
    assert result == "proj-default"


def test_resolve_team_model_specific_wins_over_team_default():
    team_config = {
        "gpt-4": {"azure": {"litellm_credentials": "team-gpt4"}},
        "defaultconfig": {"azure": {"litellm_credentials": "team-default"}},
    }
    result = _resolve_credential_from_model_config("gpt-4", None, team_config)
    assert result == "team-gpt4"


def test_resolve_team_default_used_as_fallback():
    team_config = {
        "defaultconfig": {"azure": {"litellm_credentials": "team-default"}},
    }
    result = _resolve_credential_from_model_config("gpt-3.5", None, team_config)
    assert result == "team-default"


def test_resolve_no_match_returns_none():
    result = _resolve_credential_from_model_config("gpt-4", None, None)
    assert result is None


def test_resolve_empty_configs_returns_none():
    result = _resolve_credential_from_model_config("gpt-4", {}, {})
    assert result is None


def test_resolve_model_not_in_any_config():
    project_config = {"gpt-4": {"azure": {"litellm_credentials": "x"}}}
    result = _resolve_credential_from_model_config("gpt-3.5", project_config, None)
    assert result is None


# --- Integration tests for _apply_credential_overrides_from_model_config ---


def test_apply_overrides_project_model_specific(setup_test_credentials):
    """Scenario 2: Hotel Rec App -> gpt-4-vision -> project model-specific."""
    data = {"model": "gpt-4-vision"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "defaultconfig": {
                    "azure": {"litellm_credentials": "hotel-azure-eastus"}
                },
                "gpt-4": {"azure": {"litellm_credentials": "hotel-azure-westus"}},
            }
        },
        project_metadata={
            "model_config": {
                "defaultconfig": {"azure": {"litellm_credentials": "hotel-rec-azure"}},
                "gpt-4-vision": {"azure": {"litellm_credentials": "hotel-rec-vision"}},
            }
        },
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert data["api_base"] == "https://hotel-rec-vision.openai.azure.com/"
    assert data["api_key"] == "key-hotel-rec-vision"
    assert data["api_version"] == "2024-06-01"


def test_apply_overrides_project_default(setup_test_credentials):
    """Scenario 1: Hotel Rec App -> gpt-4 -> project default."""
    data = {"model": "gpt-4"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "defaultconfig": {
                    "azure": {"litellm_credentials": "hotel-azure-eastus"}
                },
                "gpt-4": {"azure": {"litellm_credentials": "hotel-azure-westus"}},
            }
        },
        project_metadata={
            "model_config": {
                "defaultconfig": {"azure": {"litellm_credentials": "hotel-rec-azure"}},
                "gpt-4-vision": {"azure": {"litellm_credentials": "hotel-rec-vision"}},
            }
        },
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert data["api_base"] == "https://hotel-rec-app.openai.azure.com/"
    assert data["api_key"] == "key-hotel-rec"


def test_apply_overrides_team_model_specific(setup_test_credentials):
    """Scenario 4: Hotel Review App -> gpt-4 -> team model-specific."""
    data = {"model": "gpt-4"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "defaultconfig": {
                    "azure": {"litellm_credentials": "hotel-azure-eastus"}
                },
                "gpt-4": {"azure": {"litellm_credentials": "hotel-azure-westus"}},
            }
        },
        project_metadata={},
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert data["api_base"] == "https://hotel-westus.openai.azure.com/"
    assert data["api_key"] == "key-hotel-westus"


def test_apply_overrides_team_default(setup_test_credentials):
    """Scenario 3: Hotel Review App -> gpt-3.5 -> team default."""
    data = {"model": "gpt-3.5"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "defaultconfig": {
                    "azure": {"litellm_credentials": "hotel-azure-eastus"}
                },
                "gpt-4": {"azure": {"litellm_credentials": "hotel-azure-westus"}},
            }
        },
        project_metadata={},
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert data["api_base"] == "https://hotel-eastus.openai.azure.com/"
    assert data["api_key"] == "key-hotel-eastus"


def test_apply_overrides_no_config(setup_test_credentials):
    """Scenario 6: No model_config anywhere -> data unchanged."""
    data = {"model": "gpt-4"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={},
        project_metadata={},
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert "api_base" not in data
    assert "api_key" not in data


def test_apply_overrides_clientside_credentials_take_precedence(
    setup_test_credentials,
):
    """Clientside api_base/api_key in data should block model_config override."""
    data = {
        "model": "gpt-4",
        "api_base": "https://my-custom-endpoint.openai.azure.com/",
        "api_key": "my-custom-key",
    }
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "defaultconfig": {
                    "azure": {"litellm_credentials": "hotel-azure-eastus"}
                }
            }
        },
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert data["api_base"] == "https://my-custom-endpoint.openai.azure.com/"
    assert data["api_key"] == "my-custom-key"


def test_apply_overrides_missing_credential_name(setup_test_credentials):
    """model_config references a credential that doesn't exist -> no override."""
    data = {"model": "gpt-4"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "gpt-4": {"azure": {"litellm_credentials": "nonexistent-credential"}}
            }
        },
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert "api_base" not in data
    assert "api_key" not in data


def test_apply_overrides_api_version_only_if_present(setup_test_credentials):
    """api_version should only be set if the credential contains it."""
    data = {"model": "gpt-3.5"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "defaultconfig": {
                    "azure": {"litellm_credentials": "hotel-azure-eastus"}
                }
            }
        },
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert data["api_base"] == "https://hotel-eastus.openai.azure.com/"
    assert data["api_key"] == "key-hotel-eastus"
    assert "api_version" not in data


def test_apply_overrides_no_model_in_data(setup_test_credentials):
    """No model in request data -> skip override."""
    data = {"messages": [{"role": "user", "content": "hello"}]}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "defaultconfig": {"azure": {"litellm_credentials": "some-cred"}}
            }
        },
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert "api_base" not in data


def test_apply_overrides_none_metadata(setup_test_credentials):
    """None metadata on both team and project -> skip override."""
    data = {"model": "gpt-4"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata=None,
        project_metadata=None,
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert "api_base" not in data


def test_apply_overrides_clientside_api_version_preserved(setup_test_credentials):
    """Clientside api_version should not be overwritten by credential."""
    data = {"model": "gpt-4-vision", "api_version": "2025-01-01"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "gpt-4-vision": {"azure": {"litellm_credentials": "hotel-rec-vision"}}
            }
        },
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    # api_base and api_key should be set from credential
    assert data["api_base"] == "https://hotel-rec-vision.openai.azure.com/"
    assert data["api_key"] == "key-hotel-rec-vision"
    # api_version should be preserved from the request, not overwritten
    assert data["api_version"] == "2025-01-01"


def test_resolve_non_dict_model_config_ignored():
    """Non-dict model_config (e.g. string) should be safely skipped."""
    result = _resolve_credential_from_model_config("gpt-4", "not-a-dict", None)
    assert result is None

    result = _resolve_credential_from_model_config(
        "gpt-4", None, ["also", "not", "a", "dict"]
    )
    assert result is None

    # Valid config still works alongside invalid one
    result = _resolve_credential_from_model_config(
        "gpt-4",
        "invalid",
        {"gpt-4": {"azure": {"litellm_credentials": "valid-cred"}}},
    )
    assert result == "valid-cred"


def test_resolve_pre_alias_model_name_fallback():
    """model_config keyed on pre-alias name should match after alias resolution."""
    team_config = {
        "gpt-4": {"azure": {"litellm_credentials": "team-gpt4"}},
    }
    # Post-alias name doesn't match, but pre-alias does (team scope)
    result = _resolve_credential_from_model_config(
        "azure/gpt-4-0613", None, team_config, pre_alias_model_name="gpt-4"
    )
    assert result == "team-gpt4"

    # Same test for project scope
    project_config = {
        "gpt-4": {"azure": {"litellm_credentials": "proj-gpt4"}},
    }
    result = _resolve_credential_from_model_config(
        "azure/gpt-4-0613", project_config, None, pre_alias_model_name="gpt-4"
    )
    assert result == "proj-gpt4"


def test_resolve_post_alias_name_takes_priority():
    """Post-alias (resolved) name should be tried before pre-alias name."""
    team_config = {
        "gpt-4": {"azure": {"litellm_credentials": "pre-alias-cred"}},
        "gpt-4o-team-1": {"azure": {"litellm_credentials": "post-alias-cred"}},
    }
    # Team scope
    result = _resolve_credential_from_model_config(
        "gpt-4o-team-1", None, team_config, pre_alias_model_name="gpt-4"
    )
    assert result == "post-alias-cred"

    # Project scope
    result = _resolve_credential_from_model_config(
        "gpt-4o-team-1", team_config, None, pre_alias_model_name="gpt-4"
    )
    assert result == "post-alias-cred"


def test_apply_overrides_with_alias(setup_test_credentials):
    """Credential override should work when model name was changed by alias."""
    # Simulate: user called "my-gpt4", alias resolved to "azure/gpt-4-custom"
    # model_config is keyed on "my-gpt4" (the pre-alias name)
    data = {"model": "azure/gpt-4-custom"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "my-gpt4": {"azure": {"litellm_credentials": "hotel-azure-eastus"}},
            }
        },
    )
    _apply_credential_overrides_from_model_config(
        data=data,
        user_api_key_dict=user_api_key_dict,
        pre_alias_model_name="my-gpt4",
    )
    assert data["api_base"] == "https://hotel-eastus.openai.azure.com/"
    assert data["api_key"] == "key-hotel-eastus"


def test_apply_overrides_feature_flag_disabled_by_default():
    """Feature flag defaults to False — credential overrides are inert until explicitly enabled."""
    assert litellm.enable_model_config_credential_overrides is False
    data = {"model": "gpt-4"}
    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={
            "model_config": {
                "gpt-4": {"azure": {"litellm_credentials": "hotel-azure-eastus"}}
            }
        },
    )
    _apply_credential_overrides_from_model_config(
        data=data, user_api_key_dict=user_api_key_dict
    )
    assert "api_base" not in data
    assert "api_key" not in data


def test_extract_credential_provider_hint_prefers_exact_match():
    """Provider hint selects the correct provider in a multi-provider entry."""
    entry = {
        "openai": {"litellm_credentials": "openai-cred"},
        "azure": {"litellm_credentials": "azure-cred"},
    }
    # With provider hint, should pick the exact match
    assert _extract_credential_from_entry(entry, provider="azure") == "azure-cred"
    assert _extract_credential_from_entry(entry, provider="openai") == "openai-cred"

    # Without provider hint, falls back to first key (insertion order)
    result = _extract_credential_from_entry(entry)
    assert result in ("openai-cred", "azure-cred")

    # Unknown provider falls back to first available
    result = _extract_credential_from_entry(entry, provider="bedrock")
    assert result in ("openai-cred", "azure-cred")


def test_resolve_provider_hint_from_model_name():
    """Provider prefix in model name (e.g. azure/gpt-4) threads through to entry extraction."""
    config = {
        "gpt-4": {
            "openai": {"litellm_credentials": "openai-cred"},
            "azure": {"litellm_credentials": "azure-cred"},
        },
    }
    # Model name "azure/gpt-4" -> provider="azure" -> should prefer azure-cred
    # But _resolve_credential_from_model_config tries "azure/gpt-4" first (no match),
    # then falls to defaultconfig (no match). So we need to use pre_alias_model_name.
    result = _resolve_credential_from_model_config(
        "azure/gpt-4", config, None, pre_alias_model_name="gpt-4", provider="azure"
    )
    assert result == "azure-cred"


def test_clean_headers_preserves_x_api_key_when_byok_enabled():
    """
    Regression test: when forward_llm_provider_auth_headers=True,
    clean_headers() must preserve the client-supplied x-api-key header
    so it can be forwarded to the upstream Anthropic API (BYOK flow).
    """
    headers = Headers(
        {
            "x-api-key": "sk-ant-api03-client-key",
            "x-litellm-api-key": "sk-proxy-virtual-key",
            "content-type": "application/json",
        }
    )

    result = clean_headers(
        headers=headers,
        litellm_key_header_name="x-litellm-api-key",
        forward_llm_provider_auth_headers=True,
        authenticated_with_header="x-litellm-api-key",
    )

    # x-api-key must be preserved for BYOK
    assert result.get("x-api-key") == "sk-ant-api03-client-key"
    # x-litellm-api-key must NOT leak to the upstream
    assert "x-litellm-api-key" not in result


def test_clean_headers_strips_x_api_key_when_byok_disabled():
    """
    Regression test: with forward_llm_provider_auth_headers=False (default),
    x-api-key must be stripped so proxy-configured keys are not overridden
    by a client-supplied one.
    """
    headers = Headers(
        {
            "x-api-key": "sk-ant-api03-client-key",
            "x-litellm-api-key": "sk-proxy-virtual-key",
        }
    )

    result = clean_headers(
        headers=headers,
        litellm_key_header_name="x-litellm-api-key",
        forward_llm_provider_auth_headers=False,
        authenticated_with_header="x-litellm-api-key",
    )

    assert "x-api-key" not in result


def test_clean_headers_strips_x_api_key_when_byok_enabled_but_x_api_key_was_auth_header():
    """
    Anti-replay regression: even when forward_llm_provider_auth_headers=True,
    if the client authenticated TO the proxy using x-api-key (i.e., the proxy
    key arrived as x-api-key), clean_headers() must NOT forward that header
    upstream. Otherwise a proxy-auth key would leak to the LLM provider.
    """
    headers = Headers(
        {
            "x-api-key": "sk-proxy-auth-key-masquerading-as-anthropic-key",
            "content-type": "application/json",
        }
    )

    result = clean_headers(
        headers=headers,
        litellm_key_header_name="x-litellm-api-key",
        forward_llm_provider_auth_headers=True,
        authenticated_with_header="x-api-key",
    )

    # Even with BYOK enabled, x-api-key must be stripped when it was used
    # as the LiteLLM auth header (anti-replay guard).
    assert "x-api-key" not in result


# ---------------------------------------------------------------------------
# Team guardrail + global policy regression tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_guardrail_merges_with_global_policy():
    """
    Regression: team's direct guardrail must be present alongside guardrails
    resolved from a global policy (scope='*') configured by the admin.

    The bug: get_guardrail_from_metadata checked litellm_metadata before
    metadata. When the request contained a non-empty litellm_metadata field
    (without a 'guardrails' key), the merged list in data["metadata"] was
    shadowed and non-default guardrails silently received an empty
    requested_guardrails list.
    """
    from litellm.proxy.policy_engine.attachment_registry import get_attachment_registry
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.proxy.litellm_pre_call_utils import move_guardrails_to_metadata
    from litellm.types.proxy.policy_engine import (
        Policy,
        PolicyAttachment,
        PolicyGuardrails,
    )

    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        # Simulate a request that carries litellm_metadata (without guardrails)
        # which previously shadowed data["metadata"]["guardrails"].
        "litellm_metadata": {"some_user_field": "some_value"},
        "metadata": {},
    }

    user_api_key_dict = UserAPIKeyAuth(
        api_key="test-key",
        team_metadata={"guardrails": ["team-direct-guardrail"]},
    )

    policy_registry = get_policy_registry()
    policy_registry._policies = {
        "global-policy": Policy(
            guardrails=PolicyGuardrails(
                add=["policy-guardrail-1", "policy-guardrail-2"]
            ),
        ),
    }
    policy_registry._initialized = True

    attachment_registry = get_attachment_registry()
    attachment_registry._attachments = [
        PolicyAttachment(policy="global-policy", scope="*"),
    ]
    attachment_registry._initialized = True

    try:
        with patch("litellm.proxy.utils._premium_user_check"):
            await move_guardrails_to_metadata(
                data=data,
                _metadata_variable_name="metadata",
                user_api_key_dict=user_api_key_dict,
            )

        guardrails = data["metadata"].get("guardrails", [])

        assert (
            "team-direct-guardrail" in guardrails
        ), f"Team guardrail missing from merged list: {guardrails}"
        assert (
            "policy-guardrail-1" in guardrails
        ), f"policy-guardrail-1 missing: {guardrails}"
        assert (
            "policy-guardrail-2" in guardrails
        ), f"policy-guardrail-2 missing: {guardrails}"
        assert len(guardrails) == len(
            set(guardrails)
        ), f"Duplicates in guardrails list: {guardrails}"

        # Verify get_guardrail_from_metadata returns the merged list even
        # when litellm_metadata is present (the bug: it returned [] before fix)
        from litellm.integrations.custom_guardrail import CustomGuardrail

        class _DummyGuardrail(CustomGuardrail):
            pass

        dummy = _DummyGuardrail(guardrail_name="team-direct-guardrail")
        returned = dummy.get_guardrail_from_metadata(data)
        assert (
            "team-direct-guardrail" in returned
        ), f"get_guardrail_from_metadata shadowed by litellm_metadata; got: {returned}"

    finally:
        policy_registry._policies = {}
        policy_registry._initialized = False
        attachment_registry._attachments = []
        attachment_registry._initialized = False


@pytest.mark.asyncio
async def test_get_guardrail_from_metadata_prefers_metadata_over_litellm_metadata():
    """
    Unit test: get_guardrail_from_metadata must read from data["metadata"] first.
    A non-empty data["litellm_metadata"] without a 'guardrails' key must not
    shadow data["metadata"]["guardrails"].
    """
    from litellm.integrations.custom_guardrail import CustomGuardrail

    class _DummyGuardrail(CustomGuardrail):
        pass

    dummy = _DummyGuardrail(guardrail_name="my-guardrail")

    data = {
        "metadata": {"guardrails": ["my-guardrail", "other-guardrail"]},
        "litellm_metadata": {"some_field": "some_value"},  # no 'guardrails' key
    }

    result = dummy.get_guardrail_from_metadata(data)
    assert result == [
        "my-guardrail",
        "other-guardrail",
    ], f"Expected guardrails from metadata, got: {result}"


def test_get_guardrail_from_metadata_reads_litellm_metadata_when_no_metadata():
    """
    get_guardrail_from_metadata must still read from litellm_metadata when
    data["metadata"] has no 'guardrails' key (thread/assistant endpoint path).
    """
    from litellm.integrations.custom_guardrail import CustomGuardrail

    class _DummyGuardrail(CustomGuardrail):
        pass

    dummy = _DummyGuardrail(guardrail_name="my-guardrail")

    data = {
        "metadata": {"requester_metadata": {"user": "alice"}},  # no guardrails key
        "litellm_metadata": {"guardrails": ["my-guardrail"]},
    }

    result = dummy.get_guardrail_from_metadata(data)
    assert result == [
        "my-guardrail"
    ], f"Expected guardrails from litellm_metadata fallback, got: {result}"
