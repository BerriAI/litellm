"""Tests for Alice WonderFence guardrail integration."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.alice.alice_wonderfence import (
    WonderFenceGuardrail,
    WonderFenceMissingSecrets,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import ModelResponse, Choices, Message


@pytest.fixture
def mock_wonderfence_client():
    """Create a mock WonderFence client."""
    mock_client = Mock()
    mock_client.evaluate_prompt = AsyncMock()
    mock_client.evaluate_response = AsyncMock()
    return mock_client


@pytest.fixture
def wonderfence_guardrail(monkeypatch, mock_wonderfence_client):
    """Create a WonderFence guardrail with mocked client."""
    # Mock the WonderFence SDK import
    mock_sdk = Mock()
    mock_sdk.WonderFenceClient = Mock(return_value=mock_wonderfence_client)
    mock_sdk.models = Mock()
    mock_sdk.models.AnalysisContext = Mock(return_value=Mock())

    monkeypatch.setitem(__import__("sys").modules, "wonderfence_sdk", mock_sdk)
    monkeypatch.setitem(__import__("sys").modules, "wonderfence_sdk.client", mock_sdk)
    monkeypatch.setitem(__import__("sys").modules, "wonderfence_sdk.models", mock_sdk.models)

    guardrail = WonderFenceGuardrail(
        guardrail_name="wonderfence-test",
        api_key="test-api-key",
        app_name="test-app",
        event_hook=[GuardrailEventHooks.pre_call, GuardrailEventHooks.post_call],
        default_on=True,
    )

    # Replace the client with our mock
    guardrail.client = mock_wonderfence_client

    return guardrail


@pytest.fixture
def user_api_key_dict():
    """Create a mock user API key dict."""
    return UserAPIKeyAuth(
        user_id="test-user",
        team_id="test-team",
        end_user_id="end-user-123",
        api_key="test-key",
    )


def test_initialization_without_api_key(monkeypatch):
    """Test that initialization fails without API key."""
    monkeypatch.delenv("WONDERFENCE_API_KEY", raising=False)

    with pytest.raises(WonderFenceMissingSecrets) as exc:
        WonderFenceGuardrail(
            guardrail_name="test",
            event_hook=GuardrailEventHooks.pre_call,
        )

    assert "WonderFence API key not found" in str(exc.value)


def test_initialization_with_env_var(monkeypatch):
    """Test that initialization uses environment variable."""
    # Mock the SDK
    mock_sdk = Mock()
    mock_client = Mock()
    mock_sdk.WonderFenceClient = Mock(return_value=mock_client)
    mock_sdk.models = Mock()
    mock_sdk.models.AnalysisContext = Mock()

    monkeypatch.setitem(__import__("sys").modules, "wonderfence_sdk", mock_sdk)
    monkeypatch.setitem(__import__("sys").modules, "wonderfence_sdk.client", mock_sdk)
    monkeypatch.setitem(__import__("sys").modules, "wonderfence_sdk.models", mock_sdk.models)

    monkeypatch.setenv("WONDERFENCE_API_KEY", "env-api-key")
    monkeypatch.setenv("WONDERFENCE_APP_NAME", "env-app-name")

    guardrail = WonderFenceGuardrail(
        guardrail_name="test",
        event_hook=GuardrailEventHooks.pre_call,
    )

    assert guardrail.api_key == "env-api-key"
    assert guardrail.app_name == "env-app-name"


@pytest.mark.asyncio
async def test_pre_call_hook_block_action(
    wonderfence_guardrail, mock_wonderfence_client, user_api_key_dict
):
    """Test that BLOCK action raises HTTPException in pre-call hook."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "BLOCK"
    mock_result.detections = [
        Mock(
            policy_name="test-policy",
            confidence=0.95,
            message="Violation detected",
        )
    ]
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    data = {
        "messages": [{"role": "user", "content": "test message"}],
        "model": "gpt-4",
        "metadata": {},
    }

    with pytest.raises(HTTPException) as exc:
        await wonderfence_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=Mock(),
            data=data,
            call_type="acompletion",
        )

    assert exc.value.status_code == 400
    assert "Blocked by WonderFence guardrail" in exc.value.detail["error"]
    assert exc.value.detail["action"] == "BLOCK"
    assert len(exc.value.detail["detections"]) == 1


@pytest.mark.asyncio
async def test_pre_call_hook_mask_action(
    wonderfence_guardrail, mock_wonderfence_client, user_api_key_dict
):
    """Test that MASK action replaces content in pre-call hook."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "MASK"
    mock_result.action_text = "[REDACTED]"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    data = {
        "messages": [{"role": "user", "content": "sensitive content"}],
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=Mock(),
        data=data,
        call_type="acompletion",
    )

    # Check that the content was masked
    assert result["messages"][0]["content"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_pre_call_hook_detect_action(
    wonderfence_guardrail, mock_wonderfence_client, user_api_key_dict
):
    """Test that DETECT action logs but continues in pre-call hook."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "DETECT"
    mock_result.detections = [Mock(policy_name="test-policy")]
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    data = {
        "messages": [{"role": "user", "content": "test message"}],
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=Mock(),
        data=data,
        call_type="acompletion",
    )

    # Should return data unchanged
    assert result["messages"][0]["content"] == "test message"


@pytest.mark.asyncio
async def test_pre_call_hook_no_action(
    wonderfence_guardrail, mock_wonderfence_client, user_api_key_dict
):
    """Test that NO_ACTION passes through unchanged."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "NO_ACTION"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    data = {
        "messages": [{"role": "user", "content": "safe content"}],
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=Mock(),
        data=data,
        call_type="acompletion",
    )

    assert result["messages"][0]["content"] == "safe content"


@pytest.mark.asyncio
async def test_post_call_hook_block_action(
    wonderfence_guardrail, mock_wonderfence_client, user_api_key_dict
):
    """Test that BLOCK action raises HTTPException in post-call hook."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "BLOCK"
    mock_result.detections = [Mock(policy_name="test-policy", confidence=0.9, message="Bad response")]
    mock_wonderfence_client.evaluate_response.return_value = mock_result

    # Create a mock response
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="test response", role="assistant"),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
    )

    data = {
        "messages": [{"role": "user", "content": "test"}],
        "model": "gpt-4",
        "metadata": {},
    }

    with pytest.raises(HTTPException) as exc:
        await wonderfence_guardrail.async_post_call_success_hook(
            data=data,
            user_api_key_dict=user_api_key_dict,
            response=response,
        )

    assert exc.value.status_code == 400
    assert "Blocked by WonderFence guardrail" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_post_call_hook_mask_action(
    wonderfence_guardrail, mock_wonderfence_client, user_api_key_dict
):
    """Test that MASK action replaces response content in post-call hook."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "MASK"
    mock_result.action_text = "[RESPONSE BLOCKED]"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_response.return_value = mock_result

    # Create a mock response
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content="sensitive response", role="assistant"),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
    )

    data = {
        "messages": [{"role": "user", "content": "test"}],
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.async_post_call_success_hook(
        data=data,
        user_api_key_dict=user_api_key_dict,
        response=response,
    )

    # Check that the content was masked
    assert result.choices[0].message.content == "[RESPONSE BLOCKED]"


@pytest.mark.asyncio
async def test_error_propagation(
    wonderfence_guardrail, mock_wonderfence_client, user_api_key_dict
):
    """Test that errors are propagated and block requests."""
    # Mock the evaluation to raise an exception
    mock_wonderfence_client.evaluate_prompt.side_effect = Exception("API Error")

    data = {
        "messages": [{"role": "user", "content": "test"}],
        "model": "gpt-4",
        "metadata": {},
    }

    # Should raise HTTPException
    with pytest.raises(HTTPException) as exc:
        await wonderfence_guardrail.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=Mock(),
            data=data,
            call_type="acompletion",
        )

    assert exc.value.status_code == 500
    assert "Error in WonderFence Guardrail" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_should_run_guardrail_disabled(
    wonderfence_guardrail, mock_wonderfence_client, user_api_key_dict
):
    """Test that guardrail doesn't run when disabled."""
    wonderfence_guardrail.default_on = False

    data = {
        "messages": [{"role": "user", "content": "test"}],
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=Mock(),
        data=data,
        call_type="acompletion",
    )

    # Should return data unchanged without calling the API
    assert result == data
    mock_wonderfence_client.evaluate_prompt.assert_not_called()


@pytest.mark.asyncio
async def test_skip_non_user_messages(
    wonderfence_guardrail, mock_wonderfence_client, user_api_key_dict
):
    """Test that only user messages are evaluated in pre-call."""
    mock_result = Mock()
    mock_result.action = "NO_ACTION"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    data = {
        "messages": [
            {"role": "system", "content": "system message"},
            {"role": "assistant", "content": "assistant message"},
            {"role": "user", "content": "user message"},
        ],
        "model": "gpt-4",
        "metadata": {},
    }

    await wonderfence_guardrail.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=Mock(),
        data=data,
        call_type="acompletion",
    )

    # Should only be called once for the user message
    assert mock_wonderfence_client.evaluate_prompt.call_count == 1


def test_get_config_model():
    """Test that get_config_model returns the correct model."""
    from litellm.types.proxy.guardrails.guardrail_hooks.alice import (
        WonderFenceGuardrailConfigModel,
    )

    config_model = WonderFenceGuardrail.get_config_model()
    assert config_model == WonderFenceGuardrailConfigModel
