"""
Unit tests for Zscaler AI Guard guardrail

Tests covering:
- API call handling (ALLOW, BLOCK, errors)
- Policy ID precedence (metadata > user_api_key > team > init)
- Boolean config parameter handling (send_user_api_key_*)
- Metadata resolution (pre-call vs post-call)
- Header preparation with kwargs
- resolve-and-execute-policy endpoint (policyId omission)
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import HTTPException
from litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard import ZscalerAIGuard
import asyncio


@pytest.mark.asyncio
async def test_make_zscaler_ai_guard_api_call_allow():
    """Test Zscaler AI Guard API call when response action is 'ALLOW'."""
    # Mock the Zscaler AI Guard API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "statusCode": 200,
        "action": "ALLOW",
        "zscaler_ai_guard_response": {},
    }

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.return_value = mock_response
        result = await guardrail.make_zscaler_ai_guard_api_call(
            guardrail.zscaler_ai_guard_url,
            guardrail.api_key,
            guardrail.policy_id,
            "IN",
            "Test content",
        )

        assert result["action"] == "ALLOW"
        assert (
            result["zscaler_ai_guard_response"]["zscaler_ai_guard_response"] == {}
        )  # Validating response structure
        assert result["direction"] == "IN"  # Check additional fields returned


@pytest.mark.asyncio
async def test_make_zscaler_ai_guard_api_call_block():
    """Test Zscaler AI Guard API call when response action is 'BLOCK'."""
    # Mock the Zscaler AI Guard API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "statusCode": 200,
        "action": "BLOCK",
        "transactionId": "12345",
        "detectorResponses": {"detector-1": {"triggered": True, "action": "BLOCK"}},
    }

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.return_value = mock_response
        result = await guardrail.make_zscaler_ai_guard_api_call(
            guardrail.zscaler_ai_guard_url,
            guardrail.api_key,
            guardrail.policy_id,
            "IN",
            "Blocked content",
        )

        assert result["action"] == "BLOCK"
        assert result["zscaler_ai_guard_response"]["transactionId"] == "12345"
        assert (
            result["zscaler_ai_guard_response"]["detectorResponses"]["detector-1"][
                "action"
            ]
            == "BLOCK"
        )

@pytest.mark.asyncio
async def test_make_zscaler_ai_guard_api_call_request_exception():
    """Test Zscaler AI Guard API call where an exception in the request occurs."""
    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.side_effect = Exception("Connection error")

        with pytest.raises(HTTPException) as e:
            await guardrail.make_zscaler_ai_guard_api_call(
                guardrail.zscaler_ai_guard_url,
                guardrail.api_key,
                guardrail.policy_id,
                "IN",
                "Error content",
            )

        assert e.value.status_code == 500
        assert "Connection error" in e.value.detail["reason"]

def test_extract_blocking_info():
    """Test extract_blocking_info method."""
    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    response = {
        "transactionId": "12345",
        "detectorResponses": {
            "detector1": {"triggered": True, "action": "BLOCK"},
            "detector2": {"triggered": False, "action": "ALLOW"},
        },
    }

    blocking_info = guardrail.extract_blocking_info(response)

    assert blocking_info["transactionId"] == "12345"
    assert blocking_info["blockingDetectors"] == ["detector1"]


@pytest.mark.asyncio
@patch(
    "litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard.ZscalerAIGuard.make_zscaler_ai_guard_api_call",
    new_callable=AsyncMock,
)
async def test_apply_guardrail_text_concatenation(mock_api_call):
    """
    Test that `apply_guardrail` correctly concatenates texts.
    """
    guardrail = ZscalerAIGuard(policy_id=100)
    inputs = {"texts": ["Hello", "world"]}
    request_data = {}

    await guardrail.apply_guardrail(inputs, request_data, "request")

    mock_api_call.assert_called_once()
    call_args = mock_api_call.call_args
    assert call_args.kwargs["content"] == "Hello world"


@pytest.mark.asyncio
@patch(
    "litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard.ZscalerAIGuard.make_zscaler_ai_guard_api_call",
    new_callable=AsyncMock,
)
async def test_policy_id_from_request_metadata(mock_api_call):
    """
    Test policy_id is picked from request metadata (highest precedence).
    """
    guardrail = ZscalerAIGuard(policy_id=100)
    inputs = {"texts": ["test"]}
    request_data = {
        "metadata": {
            "zguard_policy_id": 1,
            "user_api_key_metadata": {"zguard_policy_id": 2},
            "team_metadata": {"zguard_policy_id": 3},
        }
    }

    await guardrail.apply_guardrail(inputs, request_data, "request")

    mock_api_call.assert_called_once()
    assert mock_api_call.call_args.kwargs["policy_id"] == 1


@pytest.mark.asyncio
@patch(
    "litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard.ZscalerAIGuard.make_zscaler_ai_guard_api_call",
    new_callable=AsyncMock,
)
async def test_policy_id_from_user_api_key_metadata(mock_api_call):
    """
    Test policy_id is picked from user_api_key_metadata (2nd precedence).
    """
    guardrail = ZscalerAIGuard(policy_id=100)
    inputs = {"texts": ["test"]}
    request_data = {
        "metadata": {
            "user_api_key_metadata": {"zguard_policy_id": 2},
            "team_metadata": {"zguard_policy_id": 3},
        }
    }

    await guardrail.apply_guardrail(inputs, request_data, "request")

    mock_api_call.assert_called_once()
    assert mock_api_call.call_args.kwargs["policy_id"] == 2


@pytest.mark.asyncio
@patch(
    "litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard.ZscalerAIGuard.make_zscaler_ai_guard_api_call",
    new_callable=AsyncMock,
)
async def test_policy_id_from_team_metadata(mock_api_call):
    """
    Test policy_id is picked from team_metadata (3rd precedence).
    """
    guardrail = ZscalerAIGuard(policy_id=100)
    inputs = {"texts": ["test"]}
    request_data = {"metadata": {"team_metadata": {"zguard_policy_id": 3}}}

    await guardrail.apply_guardrail(inputs, request_data, "request")

    mock_api_call.assert_called_once()
    assert mock_api_call.call_args.kwargs["policy_id"] == 3


@pytest.mark.asyncio
@patch(
    "litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard.ZscalerAIGuard.make_zscaler_ai_guard_api_call",
    new_callable=AsyncMock,
)
async def test_policy_id_from_init(mock_api_call):
    """
    Test policy_id is picked from guardrail initialization (lowest precedence).
    """
    guardrail = ZscalerAIGuard(policy_id=100)
    inputs = {"texts": ["test"]}
    request_data = {"metadata": {}}

    await guardrail.apply_guardrail(inputs, request_data, "request")

    mock_api_call.assert_called_once()
    assert mock_api_call.call_args.kwargs["policy_id"] == 100

@pytest.mark.asyncio
@patch(
    "litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard.ZscalerAIGuard.make_zscaler_ai_guard_api_call",
    new_callable=AsyncMock,
)
async def test_policy_id_zero_from_request_metadata(mock_api_call):
    """
    Test policy_id=0 is correctly picked. Make sure pick exact policy_id which users set
    """
    guardrail = ZscalerAIGuard(policy_id=100)
    inputs = {"texts": ["test"]}
    request_data = {
        "metadata": {
            "zguard_policy_id": 0,
        }
    }
    await guardrail.apply_guardrail(inputs, request_data, "request")
    mock_api_call.assert_called_once()
    assert mock_api_call.call_args.kwargs["policy_id"] == 0

@pytest.mark.asyncio
async def test_should_use_config_send_user_api_key_alias_when_true():
    """Test that send_user_api_key_alias=True from config is used (not overridden by env)"""
    guardrail = ZscalerAIGuard(
        api_key="test_key",
        send_user_api_key_alias=True,
    )
    assert guardrail.send_user_api_key_alias is True


@pytest.mark.asyncio
async def test_should_preserve_policy_id_zero_in_init():
    """Test that policy_id=0 is preserved (not treated as falsy and overridden by env)"""
    guardrail = ZscalerAIGuard(
        api_key="test_key",
        policy_id=0,
    )
    assert guardrail.policy_id == 0


@pytest.mark.asyncio
async def test_should_resolve_from_litellm_metadata_during_post_call():
    """Test that user_api_key_alias is resolved from litellm_metadata during post-call"""
    request_data = {
        "litellm_metadata": {
            "user_api_key_alias": "test-alias-post-call"
        }
    }
    result = ZscalerAIGuard._resolve_metadata_value(request_data, "user_api_key_alias")
    assert result == "test-alias-post-call"


@pytest.mark.asyncio
async def test_should_resolve_user_api_key_key_alias_mapping():
    """Test key_alias -> user_api_key_key_alias mapping in litellm_metadata"""
    request_data = {
        "litellm_metadata": {
            "user_api_key_key_alias": "test-key-alias"
        }
    }
    result = ZscalerAIGuard._resolve_metadata_value(request_data, "user_api_key_alias")
    assert result == "test-key-alias"


@pytest.mark.asyncio
async def test_should_include_user_api_key_alias_header():
    """Test that user-api-key-alias header is included when send_user_api_key_alias is True"""
    guardrail = ZscalerAIGuard(
        api_key="test_key",
        send_user_api_key_alias=True,
    )
    headers = guardrail._prepare_headers("test_key", user_api_key_alias="test-alias")
    assert headers.get("user-api-key-alias") == "test-alias"


@pytest.mark.asyncio
async def test_should_omit_policy_id_when_zero_or_negative():
    """Test that policyId is omitted from request body when policy_id <= 0 (for resolve-and-execute-policy)"""
    guardrail = ZscalerAIGuard(
        api_key="test_key",
        policy_id=-1,
    )

    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "statusCode": 200,
        "action": "ALLOW",
    }

    with patch.object(guardrail, "_send_request", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = mock_response

        await guardrail.make_zscaler_ai_guard_api_call(
            zscaler_ai_guard_url="http://example.com",
            api_key="test_key",
            policy_id=-1,
            direction="OUT",
            content="test content",
        )

        call_args = mock_send.call_args
        data = call_args[0][2]  # Third positional arg is data
        assert "policyId" not in data
