import pytest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard import ZscalerAIGuard
from requests.exceptions import Timeout, RequestException
import asyncio
import os


@patch("requests.Session.post")
def test_make_zscaler_ai_guard_api_call_allow(mock_post):
    """Test Zscaler AI Guard API call when response action is 'ALLOW'."""
    # Mock the Zscaler AI Guard API response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "statusCode": 200,
        "action": "ALLOW",
        "zscaler_ai_guard_response": {},
    }
    mock_post.return_value = mock_response

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    result = guardrail.make_zscaler_ai_guard_api_call(
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


@patch("requests.Session.post")
def test_make_zscaler_ai_guard_api_call_block(mock_post):
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
    mock_post.return_value = mock_response

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    result = guardrail.make_zscaler_ai_guard_api_call(
        guardrail.zscaler_ai_guard_url,
        guardrail.api_key,
        guardrail.policy_id,
        "IN",
        "Blocked content",
    )

    assert result["action"] == "BLOCK"
    assert result["zscaler_ai_guard_response"]["transactionId"] == "12345"
    assert (
        result["zscaler_ai_guard_response"]["detectorResponses"]["detector-1"]["action"]
        == "BLOCK"
    )


@patch("requests.Session.post")
def test_make_zscaler_ai_guard_api_call_timeout(mock_post):
    """Test Zscaler AI Guard API call where a timeout occurs."""
    mock_post.side_effect = Timeout

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    with pytest.raises(HTTPException) as e:
        guardrail.make_zscaler_ai_guard_api_call(
            guardrail.zscaler_ai_guard_url,
            guardrail.api_key,
            guardrail.policy_id,
            "IN",
            "Timeout content",
        )

    assert e.value.status_code == 500
    assert e.value.detail["reason"] == "Service timed out"


@patch("requests.Session.post")
def test_make_zscaler_ai_guard_api_call_request_exception(mock_post):
    """Test Zscaler AI Guard API call where an exception in the request occurs."""
    mock_post.side_effect = RequestException("Connection error")

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    with pytest.raises(HTTPException) as e:
        guardrail.make_zscaler_ai_guard_api_call(
            guardrail.zscaler_ai_guard_url,
            guardrail.api_key,
            guardrail.policy_id,
            "IN",
            "Error content",
        )

    assert e.value.status_code == 500
    assert "Hit exception when request Zscaler AI Guard" in e.value.detail["reason"]


@patch("requests.Session.post")
def test_async_moderation_hook_block(mock_post):
    """Test moderation hook where response action is 'BLOCK'."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "statusCode": 200,
        "action": "BLOCK",
        "zscaler_ai_guard_response": {
            "transactionId": "12345",
            "detectorResponses": {"detector-1": {"triggered": True, "action": "BLOCK"}},
        },
    }
    mock_post.return_value = mock_response

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    # Properly structured data for moderation hook
    data = {"messages": [{"content": "Blocked content"}]}

    with pytest.raises(HTTPException) as e:
        asyncio.run(
            guardrail.async_moderation_hook(
                user_api_key_dict=Mock(), data=data, call_type="completion"
            )
        )

    assert e.value.status_code == 400
    assert "Guardrail Policy Violation" in e.value.detail["error_type"]


@patch("requests.Session.post")
def test_async_moderation_hook_allow(mock_post):
    """Test moderation hook where response action is 'ALLOW'."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"action": "ALLOW", "statusCode":200}
    mock_post.return_value = mock_response

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    data = {"prompt": "Allowed content"}

    result = asyncio.run(
        guardrail.async_moderation_hook(
            user_api_key_dict=Mock(), data=data, call_type="completion"
        )
    )

    assert result == data


@patch("requests.Session.post")
def test_async_post_call_success_hook_block(mock_post):
    """Test post-call success hook where response is blocked."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "BLOCK"}
    mock_post.return_value = mock_response

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    # Simulate a valid LiteLLM response object
    from litellm import TextCompletionResponse, TextChoices

    response = TextCompletionResponse(choices=[TextChoices(text="Blocked response")])

    data = {}

    with pytest.raises(HTTPException) as e:
        asyncio.run(
            guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=Mock(), response=response
            )
        )

    assert e.value.status_code == 400
    assert "Guardrail Policy Violation" in e.value.detail["error_type"]


@patch("requests.Session.post")
def test_async_post_call_success_hook_allow(mock_post):
    """Test post-call success hook where response is allowed."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "ALLOW"}
    mock_post.return_value = mock_response

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    response = Mock()
    response.choices = [Mock(message=Mock(content="Allowed response"))]

    data = {}

    asyncio.run(
        guardrail.async_post_call_success_hook(
            data=data, user_api_key_dict=Mock(), response=response
        )
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
    os.environ,
    {
        "ZSCALER_AI_GAURD_URL": "http://example.com",
        "ZSCALER_AI_GAURD_POLICY_ID": "47",
        "ZSCALER_AI_GAURD_API_KEY": "test_api_key",
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
    os.environ,
    {
        "ZSCALER_AI_GAURD_URL": "http://example.com",
        "ZSCALER_AI_GAURD_POLICY_ID": "47",
        "ZSCALER_AI_GAURD_API_KEY": "test_api_key",
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
    os.environ,
    {
        "ZSCALER_AI_GAURD_URL": "http://example.com",
        "ZSCALER_AI_GAURD_POLICY_ID": "47",
        "ZSCALER_AI_GAURD_API_KEY": "test_api_key",
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
    os.environ,
    {
        "ZSCALER_AI_GAURD_URL": "http://example.com",
        "ZSCALER_AI_GAURD_POLICY_ID": "47",
        "ZSCALER_AI_GAURD_API_KEY": "test_api_key",
    },
)
def test_convert_litellm_response_object_to_str_invalid_type():
    """Test conversion with invalid types."""
    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )

    result = guardrail.convert_litellm_response_object_to_str("Invalid type")
    assert result is None


@patch("requests.Session.post")
@pytest.mark.asyncio
async def test_async_moderation_hook_messages(mock_post):
    """Test moderation hook with various message formats."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "ALLOW"}
    mock_post.return_value = mock_response

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
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


@patch("requests.Session.post")
@pytest.mark.asyncio
async def test_async_moderation_hook_prompt(mock_post):
    """Test moderation hook with prompt field."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "ALLOW"}
    mock_post.return_value = mock_response

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
    data = {"prompt": "Single prompt content"}

    result = await guardrail.async_moderation_hook(
        user_api_key_dict=Mock(), data=data, call_type="completion"
    )
    assert result == data


@patch("requests.Session.post")
@pytest.mark.asyncio
async def test_async_moderation_hook_inputs(mock_post):
    """Test moderation hook with inputs field."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"statusCode": 200, "action": "ALLOW"}
    mock_post.return_value = mock_response

    guardrail = ZscalerAIGuard(
        api_key="test_api_key", api_base="http://example.com", policy_id=1
    )
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
