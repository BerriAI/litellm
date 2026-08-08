import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from litellm.llms.azure_ai.azure_model_router.transformation import (
    AzureModelRouterConfig,
)
from litellm.llms.azure_ai.chat.transformation import AzureAIStudioConfig


@pytest.mark.asyncio
async def test_get_openai_compatible_provider_info():
    """
    Test that Azure AI requests are formatted correctly with the proper endpoint and parameters
    for both synchronous and asynchronous calls
    """
    config = AzureAIStudioConfig()

    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model="azure_ai/gpt-4o-mini",
        api_base="https://my-base",
        api_key="my-key",
        custom_llm_provider="azure_ai",
    )

    assert custom_llm_provider == "azure"


def test_azure_ai_validate_environment():
    config = AzureAIStudioConfig()
    headers = config.validate_environment(
        headers={},
        model="azure_ai/gpt-4o-mini",
        messages=[],
        optional_params={},
        litellm_params={},
    )
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_validate_environment_with_api_key():
    """
    Test that when api_key is provided, it is set in the api-key header
    for Azure Foundry endpoints (.services.ai.azure.com).
    """
    config = AzureAIStudioConfig()
    headers = config.validate_environment(
        headers={},
        model="Kimi-K2.5",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-api-key",
        api_base="https://my-endpoint.services.ai.azure.com",
    )
    assert headers["api-key"] == "test-api-key"
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_validate_environment_with_azure_ad_token():
    """
    Test that when no api_key is provided but Azure AD credentials are available,
    the Authorization header is set with a Bearer token.

    Regression test for https://github.com/BerriAI/litellm/issues/20759
    """
    import litellm

    config = AzureAIStudioConfig()
    with (
        patch(
            "litellm.llms.azure.common_utils.get_azure_ad_token",
            return_value="fake-azure-ad-token",
        ),
        patch(
            "litellm.llms.azure.common_utils.get_secret_str",
            return_value=None,
        ),
        patch.object(litellm, "api_key", None),
        patch.object(litellm, "azure_key", None),
    ):
        headers = config.validate_environment(
            headers={},
            model="Kimi-K2.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="https://my-endpoint.services.ai.azure.com",
        )
    assert headers.get("Authorization") == "Bearer fake-azure-ad-token"
    assert "api-key" not in headers
    assert headers["Content-Type"] == "application/json"


def test_azure_ai_grok_stop_parameter_handling():
    """
    Test that Grok models properly handle stop parameter filtering in Azure AI Studio.
    """
    config = AzureAIStudioConfig()

    # Test Grok model detection
    assert config._supports_stop_reason("grok-4-fast") == False
    assert config._supports_stop_reason("grok-4") == False
    assert config._supports_stop_reason("grok-3-mini") == False
    assert config._supports_stop_reason("grok-code-fast") == False
    assert config._supports_stop_reason("gpt-4") == True

    # Test supported parameters for Grok models
    grok_params = config.get_supported_openai_params("grok-4-fast")
    assert "stop" not in grok_params, "Grok models should not support stop parameter"

    # Test supported parameters for non-Grok models
    gpt_params = config.get_supported_openai_params("gpt-4")
    assert "stop" in gpt_params, "GPT models should support stop parameter"


def test_azure_model_router_response_shows_actual_model():
    """
    Test that Azure Model Router returns the actual model used in the response,
    not the router model.

    According to the documentation, when using Azure Model Router, the response
    should show the actual model that handled the request (e.g., gpt-5-nano-2025-08-07)
    rather than the router model (e.g., model-router).

    Regression test for: Azure Model Router should show actual model in response
    """
    from httpx import Response

    from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse

    config = AzureModelRouterConfig()

    # Mock raw response from Azure that includes the actual model used
    raw_response_json = {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-5-nano-2025-08-07",  # Actual model used by the router
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    # Create mock Response object
    mock_response = MagicMock(spec=Response)
    mock_response.json.return_value = raw_response_json
    mock_response.text = json.dumps(raw_response_json)
    mock_response.headers = {}

    # Create ModelResponse object
    model_response = ModelResponse()

    # Create mock logging object with required methods
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.post_call = MagicMock()
    logging_obj.model_call_details = {}

    # Call transform_response with router model
    result = config.transform_response(
        model="model-router",  # This is the router model (without prefix)
        raw_response=mock_response,
        model_response=model_response,
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={"model": "azure_ai/model-router"},  # Original request model
        encoding=None,
        api_key="test-key",
        json_mode=False,
    )

    # Verify that the response contains the actual model used, not the router model
    assert result.model == "azure_ai/gpt-5-nano-2025-08-07", (
        f"Expected model to be 'azure_ai/gpt-5-nano-2025-08-07' (actual model used), "
        f"but got '{result.model}'"
    )


def test_drop_tool_level_extra_fields_strips_copilot_mcp_server_name():
    """
    Regression test: Azure AI returns 400 when tools contain copilot_mcp_server_name.
    LiteLLM should strip the field and retry automatically.
    """
    import httpx

    config = AzureAIStudioConfig()

    error_text = json.dumps(
        {
            "error": {
                "message": "2 request validation errors: Extra inputs are not permitted, field: 'tools[0].copilot_mcp_server_name', value: 'github-mcp-server'; Extra inputs are not permitted, field: 'tools[1].copilot_mcp_server_name', value: 'ide'"
            }
        }
    )
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.text = error_text
    mock_response.json.return_value = json.loads(error_text)
    mock_response.status_code = 400
    e = httpx.HTTPStatusError(
        message="400", request=MagicMock(), response=mock_response
    )

    assert config._error_has_tool_level_extra_fields(error_text) is True
    assert (
        config.should_retry_llm_api_inside_llm_translation_on_http_error(e, {}) is True
    )

    request_data = {
        "model": "FW-Kimi-K2.6",
        "messages": [{"role": "user", "content": "Say hi."}],
        "tools": [
            {
                "type": "function",
                "copilot_mcp_server_name": "github-mcp-server",
                "function": {
                    "name": "github_search_code",
                    "description": "Search code",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "copilot_mcp_server_name": "ide",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    }

    result = config.transform_request_on_unprocessable_entity_error(e, request_data)

    for tool in result["tools"]:
        assert "copilot_mcp_server_name" not in tool
    assert result["tools"][0]["type"] == "function"
    assert result["tools"][1]["function"]["name"] == "read_file"


def test_transform_messages_strips_anthropic_replay_fields():
    """Regression: Anthropic /v1/messages adapter attaches ``thinking_blocks``
    (and ``provider_specific_fields`` / ``cache_control``) to the OpenAI-format
    assistant messages it builds when replaying Claude Code turns. Azure AI
    Foundry hosted model backends with strict chat-message schemas (e.g.
    Fireworks GLM) reject them with
    ``Extra inputs are not permitted, field: 'messages[n].thinking_blocks'``.
    ``_transform_messages`` must drop them before the request is sent.
    """
    config = AzureAIStudioConfig()
    messages = [
        {"role": "user", "content": "Read a file."},
        {
            "role": "assistant",
            "content": "I can help.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "The user wants me to read a file.",
                    "signature": "",
                    "cache_control": {},
                }
            ],
            "reasoning_content": "internal reasoning",
            "provider_specific_fields": {"thought_signature": "sig"},
            "cache_control": {"type": "ephemeral"},
        },
        {"role": "user", "content": "go ahead"},
    ]

    out = config._transform_messages(messages, model="glm-5.2")

    assert "thinking_blocks" not in out[1]
    assert "reasoning_content" not in out[1]
    assert "provider_specific_fields" not in out[1]
    assert "cache_control" not in out[1]
    assert out[1]["content"] == "I can help."


def test_transform_request_drops_thinking_blocks_for_fireworks_model():
    """End-to-end of the reported scenario: a Fireworks model deployed on Azure
    AI Foundry receives a replayed assistant turn carrying ``thinking_blocks``.
    The outgoing request payload built by ``transform_request`` must not carry
    it, or the Fireworks backend 400s with
    ``Extra inputs are not permitted, field: 'messages[2].thinking_blocks'``.
    Also covers the nested ``tool_calls[].function.provider_specific_fields``
    signature the Anthropic ``/v1/messages`` adapter attaches on tool-use
    replays, which otherwise 400s with
    ``Extra inputs are not permitted, field:
    'messages[n].tool_calls[m].function.provider_specific_fields'``.
    """
    config = AzureAIStudioConfig()
    messages = [
        {"role": "user", "content": "Read a file."},
        {
            "role": "assistant",
            "content": "I can help.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "The user wants me to read a file.",
                    "signature": "",
                    "cache_control": {},
                }
            ],
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "foo.txt"}',
                        "provider_specific_fields": {"thought_signature": "sig"},
                    },
                }
            ],
        },
        {"role": "user", "content": "go ahead"},
    ]

    payload = config.transform_request(
        model="glm-5.2",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    # Match the production invariant exactly: ``_strip_unsupported_message_fields``
    # pops ``thinking_blocks`` / ``reasoning_content`` at the message top level only,
    # and strips ``provider_specific_fields`` / ``cache_control`` recursively (they
    # nest inside ``tool_calls[].function`` / content blocks). Asserting the
    # top-level-only fields are absent from every nested node would document a
    # stronger invariant than the code enforces and give a false sense of security.
    top_level_only_fields = ("thinking_blocks", "reasoning_content")
    recursively_stripped_fields = ("provider_specific_fields", "cache_control")

    def _assert_no_recursively_stripped_fields(node):
        if isinstance(node, dict):
            for field in recursively_stripped_fields:
                assert field not in node
            for value in node.values():
                _assert_no_recursively_stripped_fields(value)
        elif isinstance(node, list):
            for item in node:
                _assert_no_recursively_stripped_fields(item)

    for message in payload["messages"]:
        for field in top_level_only_fields:
            assert field not in message
        _assert_no_recursively_stripped_fields(message)


def test_transform_messages_strips_nested_tool_call_provider_specific_fields():
    """Regression: the Anthropic ``/v1/messages`` adapter attaches the thought
    signature at ``tool_calls[].function.provider_specific_fields``
    (``adapters/transformation.py``). A top-level ``pop`` of
    ``provider_specific_fields`` from the message misses the nested copy, so
    strict upstreams (Fireworks-on-Azure) still 400 with
    ``Extra inputs are not permitted, field:
    'messages[1].tool_calls[0].function.provider_specific_fields'``.
    ``_transform_messages`` must strip it recursively.
    """
    config = AzureAIStudioConfig()
    messages = [
        {"role": "user", "content": "Run the tool."},
        {
            "role": "assistant",
            "content": "Calling the tool now.",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "foo.txt"}',
                        "provider_specific_fields": {"thought_signature": "sig"},
                    },
                }
            ],
        },
        {"role": "user", "content": "show me the result"},
    ]

    out = config._transform_messages(messages, model="glm-5.2")

    tool_call = out[1]["tool_calls"][0]
    assert "provider_specific_fields" not in tool_call
    assert "provider_specific_fields" not in tool_call["function"]
    assert tool_call["function"]["name"] == "read_file"


def test_strip_unsupported_message_fields_top_level_only_fields_not_stipped_below_top_level():
    """Document the production invariant Greptile flagged: ``thinking_blocks``
    and ``reasoning_content`` are popped at the message top level only. They do
    not appear nested in practice, and the production code does not recurse into
    content blocks to remove them. A future change that nests them would need a
    matching production change; pinning the current contract here keeps the test
    suite and the code in sync rather than asserting a stronger invariant than
    the code enforces.
    """
    from litellm.llms.azure_ai.chat.transformation import (
        _strip_unsupported_message_fields,
    )

    # A message that carries a top-level-only field nested inside a content
    # block; the helper must remove the top-level copy but leave the nested one
    # (the contract is top-level-only for these fields).
    message = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "hi", "thinking_blocks": [{"type": "thinking"}]},
        ],
        "thinking_blocks": [{"type": "thinking", "thinking": "top"}],
        "reasoning_content": "top-level reasoning",
    }

    _strip_unsupported_message_fields(message)  # type: ignore[arg-type]

    assert "thinking_blocks" not in message
    assert "reasoning_content" not in message
    # The nested copy inside the content block is NOT touched (top-level-only):
    assert "thinking_blocks" in message["content"][0]
