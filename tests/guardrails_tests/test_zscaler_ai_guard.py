import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi import HTTPException
from litellm import TextCompletionResponse, TextChoices
from litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard import ZscalerAIGuard
import asyncio
import os


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
async def test_make_zscaler_ai_guard_api_call_timeout():
    """Test Zscaler AI Guard API call where a timeout occurs."""
    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.side_effect = asyncio.TimeoutError

        with pytest.raises(HTTPException) as e:
            await guardrail.make_zscaler_ai_guard_api_call(
                guardrail.zscaler_ai_guard_url,
                guardrail.api_key,
                guardrail.policy_id,
                "IN",
                "Timeout content",
            )

        assert e.value.status_code == 500
        assert "Hit exception when request Zscaler AI Guard" in e.value.detail["reason"]


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
        assert "Hit exception when request Zscaler AI Guard" in e.value.detail["reason"]


@pytest.mark.asyncio
async def test_async_moderation_hook_block():
    """Test moderation hook where response action is 'BLOCK'."""
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

        # Properly structured data for moderation hook
        data = {"messages": [{"content": "Blocked content"}]}

        with pytest.raises(HTTPException) as e:
            await guardrail.async_moderation_hook(
                user_api_key_dict=Mock(), data=data, call_type="completion"
            )

        assert e.value.status_code == 400
        assert "Guardrail Policy Violation" in e.value.detail["error_type"]


@pytest.mark.asyncio
async def test_async_moderation_hook_allow():
    """Test moderation hook where response action is 'ALLOW'."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"action": "ALLOW", "statusCode": 200}

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.return_value = mock_response

        data = {"prompt": "Allowed content"}

        result = await guardrail.async_moderation_hook(
            user_api_key_dict=Mock(), data=data, call_type="completion"
        )

        assert result == data


@pytest.mark.asyncio
async def test_async_post_call_success_hook_block():
    """Test post-call success hook where response is blocked."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "BLOCK"}

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    # Simulate a valid LiteLLM response object
    from litellm import TextCompletionResponse, TextChoices

    response = TextCompletionResponse(choices=[TextChoices(text="Blocked response")])

    data = {}
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.return_value = mock_response

        with pytest.raises(HTTPException) as e:
            await guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=Mock(), response=response
            )

        assert e.value.status_code == 400
        assert "Guardrail Policy Violation" in e.value.detail["error_type"]


@pytest.mark.asyncio
async def test_async_post_call_success_hook_allow():
    """Test post-call success hook where response is allowed."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "ALLOW"}

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.return_value = mock_response

        response = Mock()
        response.choices = [Mock(message=Mock(content="Allowed response"))]

        data = {}

        await guardrail.async_post_call_success_hook(
            data=data, user_api_key_dict=Mock(), response=response
        )


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


@patch.dict(
    {
        "ZSCALER_AI_GUARD_URL": "http://example.com",
        "ZSCALER_AI_GUARD_POLICY_ID": "47",
        "ZSCALER_AI_GUARD_API_KEY": "test_api_key",
    },
)
def test_convert_litellm_response_object_to_str_text_completion():
    """Test successful conversion of TextCompletionResponse to string."""
    from litellm import TextCompletionResponse, TextChoices

    response_obj = TextCompletionResponse(
        choices=[TextChoices(text="Text result 1"), TextChoices(text="Text result 2")]
    )

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    result = guardrail.convert_litellm_response_object_to_str(response_obj)

    assert result == "Text result 1Text result 2"


@patch.dict(
    {
        "ZSCALER_AI_GUARD_URL": "http://example.com",
        "ZSCALER_AI_GUARD_POLICY_ID": "47",
        "ZSCALER_AI_GUARD_API_KEY": "test_api_key",
    },
)
def test_convert_litellm_response_object_to_str_model_response():
    """Test successful conversion of ModelResponse to string."""
    from litellm.types.utils import ModelResponse, Choices, Message

    # Use an instance of the Message class with valid content
    message_obj = Message(content="Message result 1")
    response_obj = ModelResponse(choices=[Choices(message=message_obj)])

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    result = guardrail.convert_litellm_response_object_to_str(response_obj)

    # Verify the result matches the expected string
    assert result == "Message result 1"


@patch.dict(
    {
        "ZSCALER_AI_GUARD_URL": "http://example.com",
        "ZSCALER_AI_GUARD_POLICY_ID": "47",
        "ZSCALER_AI_GUARD_API_KEY": "test_api_key",
    },
)
def test_convert_litellm_response_object_to_str_empty_choices():
    """Test conversion of response object with empty choices."""
    from litellm import TextCompletionResponse

    response_obj = TextCompletionResponse(choices=[])
    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    result = guardrail.convert_litellm_response_object_to_str(response_obj)

    assert result is None


@patch.dict(
    {
        "ZSCALER_AI_GUARD_URL": "http://example.com",
        "ZSCALER_AI_GUARD_POLICY_ID": "47",
        "ZSCALER_AI_GUARD_API_KEY": "test_api_key",
    },
)
def test_convert_litellm_response_object_to_str_invalid_type():
    """Test conversion with invalid types."""
    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    result = guardrail.convert_litellm_response_object_to_str("Invalid type")
    assert result is None


@pytest.mark.asyncio
async def test_async_moderation_hook_messages():
    """Test moderation hook with various message formats."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "ALLOW"}

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.return_value = mock_response
        data = {
            "messages": [
                {"content": "Message 1"},
                {"content": ["Message 2.1", {"text": "Message 2.2"}]},
            ]
        }

        result = await guardrail.async_moderation_hook(
            user_api_key_dict=Mock(), data=data, call_type="completion"
        )
        assert result == data


@pytest.mark.asyncio
async def test_async_moderation_hook_prompt():
    """Test moderation hook with prompt field."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "ALLOW"}

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.return_value = mock_response
        data = {"prompt": "Single prompt content"}

        result = await guardrail.async_moderation_hook(
            user_api_key_dict=Mock(), data=data, call_type="completion"
        )
        assert result == data


@pytest.mark.asyncio
async def test_async_moderation_hook_inputs():
    """Test moderation hook with inputs field."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "ALLOW"}

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    with patch.object(
        guardrail, "_send_request", new_callable=AsyncMock
    ) as mock_send_request:
        mock_send_request.return_value = mock_response
        data = {"inputs": "Input content"}

        result = await guardrail.async_moderation_hook(
            user_api_key_dict=Mock(), data=data, call_type="completion"
        )
        assert result == data


@pytest.mark.asyncio
async def test_async_moderation_hook_empty_input():
    """Test moderation hook with empty input."""
    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    data = {}

    result = await guardrail.async_moderation_hook(
        user_api_key_dict=Mock(), data=data, call_type="completion"
    )
    assert result == data  # Ensure it bypasses moderation hook
