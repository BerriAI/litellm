"""Tests for Alice WonderFence guardrail integration."""

from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.alice_wonderfence.alice_wonderfence import (
    WonderFenceGuardrail,
    WonderFenceMissingSecrets,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.alice_wonderfence import (
    WonderFenceGuardrailConfigModel,
)


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
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that BLOCK action raises HTTPException in apply_guardrail."""
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

    inputs = {"texts": ["test message"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    with pytest.raises(HTTPException) as exc:
        await wonderfence_guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

    assert exc.value.status_code == 400
    assert "Blocked by WonderFence guardrail" in exc.value.detail["error"]
    assert exc.value.detail["action"] == "BLOCK"
    assert len(exc.value.detail["detections"]) == 1


@pytest.mark.asyncio
async def test_pre_call_hook_mask_action(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that MASK action replaces content in apply_guardrail."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "MASK"
    mock_result.action_text = "[REDACTED]"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    inputs = {"texts": ["sensitive content"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )

    # Check that the content was masked
    assert result["texts"][0] == "[REDACTED]"


@pytest.mark.asyncio
async def test_pre_call_hook_detect_action(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that DETECT action logs but continues in apply_guardrail."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "DETECT"
    mock_result.detections = [Mock(policy_name="test-policy")]
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    inputs = {"texts": ["test message"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )

    # Should return inputs unchanged
    assert result["texts"][0] == "test message"


@pytest.mark.asyncio
async def test_pre_call_hook_no_action(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that NO_ACTION passes through unchanged."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "NO_ACTION"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    inputs = {"texts": ["safe content"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )

    assert result["texts"][0] == "safe content"


@pytest.mark.asyncio
async def test_post_call_hook_block_action(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that BLOCK action raises HTTPException in apply_guardrail for response."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "BLOCK"
    mock_result.detections = [Mock(policy_name="test-policy", confidence=0.9, message="Bad response")]
    mock_wonderfence_client.evaluate_response.return_value = mock_result

    inputs = {"texts": ["test response"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    with pytest.raises(HTTPException) as exc:
        await wonderfence_guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="response",
        )

    assert exc.value.status_code == 400
    assert "Blocked by WonderFence guardrail" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_post_call_hook_mask_action(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that MASK action replaces response content in apply_guardrail."""
    # Mock the evaluation result
    mock_result = Mock()
    mock_result.action = "MASK"
    mock_result.action_text = "[RESPONSE BLOCKED]"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_response.return_value = mock_result

    inputs = {"texts": ["sensitive response"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="response",
    )

    # Check that the content was masked
    assert result["texts"][0] == "[RESPONSE BLOCKED]"


@pytest.mark.asyncio
async def test_error_propagation(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that errors are propagated and block requests."""
    # Mock the evaluation to raise an exception
    mock_wonderfence_client.evaluate_prompt.side_effect = Exception("API Error")

    inputs = {"texts": ["test"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    # Should raise HTTPException
    with pytest.raises(HTTPException) as exc:
        await wonderfence_guardrail.apply_guardrail(
            inputs=inputs,
            request_data=request_data,
            input_type="request",
        )

    assert exc.value.status_code == 500
    assert "Error in WonderFence Guardrail" in exc.value.detail["error"]


@pytest.mark.asyncio
async def test_should_run_guardrail_disabled(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that guardrail processes texts even when default_on is false (apply_guardrail always runs)."""
    # Note: With apply_guardrail, the should_run_guardrail check happens at the framework level
    # So this test now just verifies normal operation
    mock_result = Mock()
    mock_result.action = "NO_ACTION"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    inputs = {"texts": ["test"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )

    # Should process normally
    assert result["texts"][0] == "test"
    mock_wonderfence_client.evaluate_prompt.assert_called_once()


@pytest.mark.asyncio
async def test_multiple_texts(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that apply_guardrail processes only the last text (latest message)."""
    mock_result = Mock()
    mock_result.action = "NO_ACTION"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    inputs = {"texts": ["text1", "text2", "text3"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )

    # Should only be called once for the last text (latest message)
    assert mock_wonderfence_client.evaluate_prompt.call_count == 1
    # Should evaluate only the last text
    mock_wonderfence_client.evaluate_prompt.assert_called_once()
    call_args = mock_wonderfence_client.evaluate_prompt.call_args
    assert call_args.kwargs["prompt"] == "text3"
    assert result["texts"] == ["text1", "text2", "text3"]


@pytest.mark.asyncio
async def test_mask_multiple_texts_request(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that MASK action only masks the last text in request (latest user message)."""
    mock_result = Mock()
    mock_result.action = "MASK"
    mock_result.action_text = "[USER MESSAGE REDACTED]"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    # Simulate multiple messages in conversation history
    inputs = {"texts": ["Hello", "How are you?", "Tell me a secret"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )

    # Should only evaluate the last text (most recent user message)
    assert mock_wonderfence_client.evaluate_prompt.call_count == 1
    call_args = mock_wonderfence_client.evaluate_prompt.call_args
    assert call_args.kwargs["prompt"] == "Tell me a secret"

    # Should only mask the last text, leaving others unchanged
    assert result["texts"] == ["Hello", "How are you?", "[USER MESSAGE REDACTED]"]


@pytest.mark.asyncio
async def test_mask_multiple_texts_response(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that MASK action only masks the last text in response (latest assistant message)."""
    mock_result = Mock()
    mock_result.action = "MASK"
    mock_result.action_text = "[ASSISTANT RESPONSE REDACTED]"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_response.return_value = mock_result

    # Simulate multiple response texts
    inputs = {"texts": ["Here's some info", "Let me add more", "Here's sensitive data"]}
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="response",
    )

    # Should only evaluate the last text (most recent assistant response)
    assert mock_wonderfence_client.evaluate_response.call_count == 1
    call_args = mock_wonderfence_client.evaluate_response.call_args
    assert call_args.kwargs["response"] == "Here's sensitive data"

    # Should only mask the last text, leaving others unchanged
    assert result["texts"] == ["Here's some info", "Let me add more", "[ASSISTANT RESPONSE REDACTED]"]


@pytest.mark.asyncio
async def test_mask_with_structured_messages_request(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that MASK action works correctly with structured_messages in request."""
    mock_result = Mock()
    mock_result.action = "MASK"
    mock_result.action_text = "[REDACTED]"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    # Simulate conversation with structured messages
    structured_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What's my password?"},
    ]
    inputs = {
        "texts": ["Hello", "Hi there!", "What's my password?"],
        "structured_messages": structured_messages,
    }
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )

    # Should evaluate only the last user message
    assert mock_wonderfence_client.evaluate_prompt.call_count == 1
    call_args = mock_wonderfence_client.evaluate_prompt.call_args
    assert call_args.kwargs["prompt"] == "What's my password?"

    # Should mask only the last text
    assert result["texts"] == ["Hello", "Hi there!", "[REDACTED]"]


@pytest.mark.asyncio
async def test_mask_with_multipart_content_request(
    wonderfence_guardrail, mock_wonderfence_client
):
    """Test that MASK action works with multi-part content (text + images)."""
    mock_result = Mock()
    mock_result.action = "MASK"
    mock_result.action_text = "[REDACTED MULTIPART]"
    mock_result.detections = []
    mock_wonderfence_client.evaluate_prompt.return_value = mock_result

    # Simulate multi-part content message with assistant message to separate user messages
    structured_messages = [
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "I understand"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            ],
        },
    ]
    inputs = {
        "texts": ["Previous message", "I understand", "Describe this image"],
        "structured_messages": structured_messages,
    }
    request_data = {
        "model": "gpt-4",
        "metadata": {},
    }

    result = await wonderfence_guardrail.apply_guardrail(
        inputs=inputs,
        request_data=request_data,
        input_type="request",
    )

    # Should evaluate only the last user message (text part extracted from multipart content)
    assert mock_wonderfence_client.evaluate_prompt.call_count == 1
    call_args = mock_wonderfence_client.evaluate_prompt.call_args
    assert call_args.kwargs["prompt"] == "Describe this image"

    # Should mask only the last text
    assert result["texts"] == ["Previous message", "I understand", "[REDACTED MULTIPART]"]


def test_get_config_model():
    """Test that get_config_model returns the correct model."""
    config_model = WonderFenceGuardrail.get_config_model()
    assert config_model == WonderFenceGuardrailConfigModel
