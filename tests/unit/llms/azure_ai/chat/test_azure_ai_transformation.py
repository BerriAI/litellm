import json
import traceback
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map
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
    ) = config.get_openai_compatible_provider_info(
        model="azure_ai/gpt-4o-mini",
        api_base="https://my-base",
        api_key="my-key",
        custom_llm_provider="azure_ai",
    )

    assert custom_llm_provider == "azure"


@pytest.mark.parametrize(
    "model, api_base, expected_provider",
    [
        ("azure_ai/gpt-4o", "https://my-resource.services.ai.azure.com", "azure_ai"),
        ("azure_ai/gpt-4o", "https://my-resource.services.ai.azure.com/models", "azure_ai"),
        ("azure_ai/gpt-5.4-nano", "https://my-resource.services.ai.azure.com", "azure_ai"),
        ("azure_ai/gpt-4o", "https://my-resource.openai.azure.com", "azure"),
        (
            "azure_ai/gpt-4o",
            "https://my-resource.services.ai.azure.com/openai/deployments/gpt-4o/chat/completions"
            "?api-version=2024-08-01-preview",
            "azure",
        ),
        ("azure_ai/mistral-large-latest", "https://my-resource.services.ai.azure.com", "azure_ai"),
        ("azure_ai/mistral-large-latest", "https://my-resource.openai.azure.com", "azure_ai"),
    ],
)
def test_foundry_base_keeps_azure_ai_provider(model: str, api_base: str, expected_provider: str):
    """Regression for #38276: a Foundry .services.ai.azure.com base must not be reclassified as azure."""
    config = AzureAIStudioConfig()
    (
        _,
        _,
        custom_llm_provider,
    ) = config.get_openai_compatible_provider_info(
        model=model,
        api_base=api_base,
        api_key="my-key",
        custom_llm_provider="azure_ai",
    )
    assert custom_llm_provider == expected_provider


def test_is_azure_openai_model_without_api_base_keeps_azure_ai():
    """Metadata lookups (get_model_info, supports_* checks) carry no api_base and must not flip the provider."""
    config = AzureAIStudioConfig()
    assert config._is_azure_openai_model(model="azure_ai/gpt-4o", api_base=None) is False
    assert config._is_azure_openai_model(model="azure_ai/gpt-4o", api_base="https://my-res.openai.azure.com") is True


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


@pytest.fixture
def _local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", get_model_cost_map(url=litellm.model_cost_map_url))


def test_foundry_gpt_6_astra_keeps_sampling_params_when_reasoning_effort_is_none(_local_model_cost_map):
    optional_params = AzureAIStudioConfig().map_openai_params(
        non_default_params={"reasoning_effort": "none", "temperature": 0.2, "top_p": 0.9},
        optional_params={},
        model="gpt-6-astra",
        drop_params=False,
    )

    assert optional_params == {"reasoning_effort": "none", "temperature": 0.2, "top_p": 0.9}




def test_azure_ai_grok_stop_parameter_handling():
    """
    Test that Grok models properly handle stop parameter filtering in Azure AI Studio.
    """
    config = AzureAIStudioConfig()

    # Test Grok model detection
    assert config._supports_stop_reason("grok-4-fast") is False
    assert config._supports_stop_reason("grok-4.3") is False
    assert config._supports_stop_reason("grok-4") is False
    assert config._supports_stop_reason("grok-3-mini") is False
    assert config._supports_stop_reason("grok-code-fast") is False
    assert config._supports_stop_reason("gpt-4") is True

    # Test supported parameters for Grok models
    for model in ("grok-4-fast", "grok-4.3"):
        grok_params = config.get_supported_openai_params(model)
        assert (
            "stop" not in grok_params
        ), "Grok models should not support stop parameter"

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


def test_azure_model_router_stamps_selected_model_on_hidden_params():
    """
    The selected model must be stamped on _hidden_params, not left for downstream code to
    re-derive by looking for "model-router" in the model string. Deployments whose alias
    does not contain that text are invisible to the string check.
    """
    from httpx import Response

    from litellm.llms.azure_ai.common_utils import (
        AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY,
        AzureFoundryModelInfo,
    )
    from litellm.llms.base_llm.chat.transformation import LiteLLMLoggingObj
    from litellm.types.utils import ModelResponse

    raw_response_json = {
        "id": "chatcmpl-test456",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "grok-4-1-fast-reasoning",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "pong"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    mock_response = MagicMock(spec=Response)
    mock_response.json.return_value = raw_response_json
    mock_response.text = json.dumps(raw_response_json)
    mock_response.headers = {}

    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.post_call = MagicMock()
    logging_obj.model_call_details = {}

    result = AzureModelRouterConfig().transform_response(
        model="smart-pick",
        raw_response=mock_response,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "Reply with just pong"}],
        optional_params={},
        litellm_params={"model": "azure_ai/model_router/smart-pick"},
        encoding=None,
        api_key="test-key",
        json_mode=False,
    )

    assert result._hidden_params[AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY] == result.model
    assert (
        result._hidden_params[AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY]
        == "azure_ai/grok-4-1-fast-reasoning"
    )
    assert AzureFoundryModelInfo.get_model_router_selected_model(
        result._hidden_params
    ) == ("azure_ai/grok-4-1-fast-reasoning")
    assert (
        AzureFoundryModelInfo.is_model_router_call(
            model="smart-pick", hidden_params=result._hidden_params
        )
        is True
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


def _find_key_anywhere(obj, key: str) -> bool:
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_find_key_anywhere(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_find_key_anywhere(item, key) for item in obj)
    return False


def test_azure_ai_strips_non_openai_spec_message_fields():
    """
    Regression for https://github.com/BerriAI/litellm/issues/33961.

    Azure AI Foundry backends set additionalProperties=false, so any message
    field outside the OpenAI chat-completions schema causes a 400 "Extra inputs
    are not permitted". Anthropic-format clients (e.g. Claude Code) echo prior
    assistant turns back as history carrying thinking_blocks, a nested thought
    signature at tool_calls[].function.provider_specific_fields, and Anthropic
    cache_control annotations. transform_request must strip all of these before
    the request reaches the upstream.
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
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "reasoning_content": "The user wants me to read a file.",
            "provider_specific_fields": {"thought_signature": "sig-top"},
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                        "provider_specific_fields": {"thought_signature": "sig-nested"},
                    },
                }
            ],
        },
        {"role": "user", "content": "go ahead"},
    ]

    request = config.transform_request(
        model="fw-glm-5.2",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    transformed_messages = request["messages"]

    assert not _find_key_anywhere(transformed_messages, "thinking_blocks")
    assert not _find_key_anywhere(transformed_messages, "reasoning_content")
    assert not _find_key_anywhere(transformed_messages, "provider_specific_fields")
    assert not _find_key_anywhere(transformed_messages, "cache_control")

    assistant_message = transformed_messages[1]
    assert assistant_message["content"] == "I can help."
    assert assistant_message["tool_calls"][0]["function"]["name"] == "read_file"


def test_azure_ai_stripping_does_not_mutate_caller_messages():
    """
    The stripping must not touch the caller's messages. LiteLLM reuses the same
    message objects when falling back to another provider, so stripping in place
    would hand the fallback a conversation history with its thinking blocks and
    provider metadata already destroyed.
    """
    config = AzureAIStudioConfig()

    messages = [
        {"role": "user", "content": "Read a file."},
        {
            "role": "assistant",
            "content": "I can help.",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "Reading the file.", "signature": "sig"}
            ],
            "provider_specific_fields": {"thought_signature": "sig-top"},
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "{}",
                        "provider_specific_fields": {"thought_signature": "sig-nested"},
                    },
                }
            ],
        },
    ]

    request = config.transform_request(
        model="fw-glm-5.2",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert not _find_key_anywhere(request["messages"], "thinking_blocks")

    original_assistant = messages[1]
    assert original_assistant["thinking_blocks"][0]["thinking"] == "Reading the file."
    assert original_assistant["provider_specific_fields"] == {"thought_signature": "sig-top"}
    assert original_assistant["tool_calls"][0]["function"]["provider_specific_fields"] == {
        "thought_signature": "sig-nested"
    }


def test_azure_ai_keeps_prompt_cache_breakpoint_for_a_model_that_supports_it(_local_model_cost_map):
    """
    Foundry GPT-5.6+ deployments take OpenAI explicit prompt caching, and the breakpoint only
    counts on a content part, so a system message carrying one must reach the request body as
    the content-part list it arrived in while the Anthropic-only fields are still stripped.
    """
    request = AzureAIStudioConfig().transform_request(
        model="gpt-6-astra",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "stable instructions",
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
        optional_params={"prompt_cache_options": {"mode": "explicit"}},
        litellm_params={},
        headers={},
    )

    assert request["messages"][0]["content"] == [
        {"type": "text", "text": "stable instructions", "prompt_cache_breakpoint": {"mode": "explicit"}}
    ]
    assert request["messages"][1]["content"] == [{"type": "text", "text": "hi"}]
    assert request["prompt_cache_options"] == {"mode": "explicit"}


def test_azure_ai_still_flattens_list_content_for_a_model_without_the_breakpoint_flag(_local_model_cost_map):
    request = AzureAIStudioConfig().transform_request(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "stable ", "prompt_cache_breakpoint": {"mode": "explicit"}},
                    {"type": "text", "text": "instructions"},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert request["messages"][0]["content"] == "stable instructions"
    assert request["messages"][1]["content"] == "hi"
    assert not _find_key_anywhere(request["messages"], "prompt_cache_breakpoint")


def test_azure_ai_drops_thinking_parts_from_history_while_keeping_the_breakpoint(_local_model_cost_map):
    """
    A multi-turn client can echo an assistant turn back with thinking parts inside its content
    list. Foundry rejects those parts, and the string conversion always dropped them, so a model
    that keeps content parts must still drop them while the text parts and the breakpoint stay.
    """
    request = AzureAIStudioConfig().transform_request(
        model="gpt-6-astra",
        messages=[
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "stable instructions", "prompt_cache_breakpoint": {"mode": "explicit"}}
                ],
            },
            {"role": "user", "content": "read the file"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "The user wants me to read a file.", "signature": "sig"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "Reading it now."},
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "thanks"}]},
        ],
        optional_params={"prompt_cache_options": {"mode": "explicit"}},
        litellm_params={},
        headers={},
    )

    assert request["messages"][0]["content"] == [
        {"type": "text", "text": "stable instructions", "prompt_cache_breakpoint": {"mode": "explicit"}}
    ]
    assert request["messages"][2]["content"] == [{"type": "text", "text": "Reading it now."}]
    assert request["messages"][3]["content"] == [{"type": "text", "text": "thanks"}]
    assert not _find_key_anywhere(request["messages"], "thinking")
    assert not _find_key_anywhere(request["messages"], "data")


@pytest.mark.parametrize(
    "api_base, expected_url",
    [
        (
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
        ),
        (
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions",
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions",
        ),
        (
            "https://litellm8397336933.services.ai.azure.com/models",
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions",
        ),
        (
            "https://litellm8397336933.services.ai.azure.com",
            "https://litellm8397336933.services.ai.azure.com/models/chat/completions",
        ),
    ],
)
def test_azure_ai_services_handler(api_base, expected_url):
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm.set_verbose = True

    client = HTTPHandler()

    with patch.object(client, "post") as mock_client:
        try:
            response = litellm.completion(
                model="azure_ai/Meta-Llama-3.1-70B-Instruct",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                api_key="my-fake-api-key",
                api_base=api_base,
                client=client,
            )

            print(response)

        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["headers"]["api-key"] == "my-fake-api-key"
        assert mock_client.call_args.kwargs["url"] == expected_url


def test_azure_ai_services_with_api_version():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler, AsyncHTTPHandler

    client = HTTPHandler()

    with patch.object(client, "post") as mock_client:
        try:
            response = litellm.completion(
                model="azure_ai/Meta-Llama-3.1-70B-Instruct",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                api_key="my-fake-api-key",
                api_version="2024-05-01-preview",
                api_base="https://litellm8397336933.services.ai.azure.com/models",
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        assert mock_client.call_args.kwargs["headers"]["api-key"] == "my-fake-api-key"
        assert (
            mock_client.call_args.kwargs["url"]
            == "https://litellm8397336933.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"
        )


@pytest.mark.asyncio
async def test_azure_ai_with_image_url():
    """
    Important test:

    Test that Azure AI studio can handle image_url passed when content is a list containing both text and image_url
    """
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    litellm.set_verbose = True

    client = AsyncHTTPHandler()

    with patch.object(client, "post") as mock_client:
        try:
            await litellm.acompletion(
                model="azure_ai/Phi-3-5-vision-instruct-dcvov",
                api_base="https://Phi-3-5-vision-instruct-dcvov.eastus2.models.ai.azure.com",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is in this image?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
                                },
                            },
                        ],
                    },
                ],
                api_key="fake-api-key",
                client=client,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")

        # Verify the request was made
        mock_client.assert_called_once()

        print(f"mock_client.call_args.kwargs: {mock_client.call_args.kwargs}")
        # Check the request body
        request_body = json.loads(mock_client.call_args.kwargs["data"])
        assert request_body["model"] == "Phi-3-5-vision-instruct-dcvov"
        assert request_body["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://litellm-listing.s3.amazonaws.com/litellm_logo.png"
                        },
                    },
                ],
            }
        ]


def test_azure_deepseek_reasoning_content():
    import json

    client = HTTPHandler()

    with patch.object(client, "post") as mock_post:
        mock_response = MagicMock()

        mock_response.text = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": "<think>I am thinking here</think>\n\nThe sky is a canvas of blue",
                            "role": "assistant",
                        },
                    }
                ],
            }
        )

        mock_response.status_code = 200
        # Add required response attributes
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json = lambda: json.loads(mock_response.text)
        mock_post.return_value = mock_response

        response = litellm.completion(
            model="azure_ai/deepseek-r1",
            messages=[{"role": "user", "content": "Hello, world!"}],
            api_base="https://litellm8397336933.services.ai.azure.com/models/chat/completions",
            api_key="my-fake-api-key",
            client=client,
        )

        print(response)
        assert response.choices[0].message.reasoning_content == "I am thinking here"
        assert response.choices[0].message.content == "\n\nThe sky is a canvas of blue"


@pytest.mark.parametrize(
    "model_group_header, expected_model",
    [
        ("offer-cohere-embed-multili-paygo", "Cohere-embed-v3-multilingual"),
        ("offer-cohere-embed-english-paygo", "Cohere-embed-v3-english"),
    ],
)
def test_map_azure_model_group(model_group_header, expected_model):
    from litellm.llms.azure_ai.embed.cohere_transformation import AzureAICohereConfig

    config = AzureAICohereConfig()
    assert config._map_azure_model_group(model_group_header) == expected_model
