import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm import get_model_info, supports_reasoning
from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig
from litellm.types.llms.openai import ChatCompletionToolCallFunctionChunk
from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message


@pytest.fixture(autouse=True)
def force_local_model_cost(monkeypatch):
    """Force local model cost map usage for all tests in this file."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # Refresh model_cost from local map
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

    litellm.model_cost = get_model_cost_map(url=litellm.model_cost_map_url)


def test_handle_message_content_with_tool_calls():
    config = FireworksAIConfig()
    message = Message(
        content='{"type": "function", "name": "get_current_weather", "parameters": {"location": "Boston, MA", "unit": "fahrenheit"}}',
        role="assistant",
        tool_calls=None,
        function_call=None,
        provider_specific_fields=None,
    )
    expected_tool_call = ChatCompletionMessageToolCall(
        function=Function(**json.loads(message.content)), id=None, type=None
    )
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    updated_message = config._handle_message_content_with_tool_calls(
        message, tool_calls
    )
    assert updated_message.tool_calls is not None
    assert len(updated_message.tool_calls) == 1
    assert updated_message.tool_calls[0].function.name == "get_current_weather"
    assert (
        updated_message.tool_calls[0].function.arguments
        == expected_tool_call.function.arguments
    )


def test_supports_reasoning_effort():
    """Test that reasoning_effort is only supported for specific Fireworks AI models."""
    supported_models = [
        "fireworks_ai/accounts/fireworks/models/qwen3-8b",
        "fireworks_ai/accounts/fireworks/models/qwen3-32b",
        "fireworks_ai/accounts/fireworks/models/qwen3-coder-480b-a35b-instruct",
        "fireworks_ai/accounts/fireworks/models/deepseek-v3p1",
        "fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
        "fireworks_ai/accounts/fireworks/models/glm-4p5",
        "fireworks_ai/accounts/fireworks/models/glm-4p5-air",
        "fireworks_ai/accounts/fireworks/models/glm-4p6",
        "fireworks_ai/accounts/fireworks/models/glm-4p7",
        "fireworks_ai/accounts/fireworks/models/glm-5p1",
        "fireworks_ai/accounts/fireworks/models/gpt-oss-120b",
        "fireworks_ai/accounts/fireworks/models/gpt-oss-20b",
        "fireworks_ai/glm-5p1",
    ]

    unsupported_models = [
        "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct",
        "fireworks_ai/accounts/fireworks/models/mixtral-8x7b-instruct",
    ]

    for model in supported_models:
        assert (
            supports_reasoning(model=model, custom_llm_provider="fireworks_ai") == True
        ), f"{model} should support reasoning_effort"

    for model in unsupported_models:
        assert (
            supports_reasoning(model=model, custom_llm_provider="fireworks_ai") == False
        ), f"{model} should not support reasoning_effort"


def test_get_supported_openai_params_reasoning_effort():
    """Test that reasoning_effort is only included in supported params for models that support it."""
    config = FireworksAIConfig()

    supported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/glm-5p1"
    )
    assert "reasoning_effort" in supported_params

    unsupported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct"
    )
    assert "reasoning_effort" not in unsupported_params


def test_get_supported_openai_params_parallel_tool_calls():
    """Test that parallel_tool_calls is included for models that support function calling."""
    config = FireworksAIConfig()

    supported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/glm-5p1"
    )
    assert "parallel_tool_calls" in supported_params
    assert "tools" in supported_params
    assert "tool_choice" in supported_params

    unsupported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct"
    )
    assert "parallel_tool_calls" not in unsupported_params


def test_get_supported_openai_params_parallel_tool_calls_without_tool_choice(
    monkeypatch,
):
    """Test that parallel_tool_calls is gated on tools, not tool_choice."""
    config = FireworksAIConfig()
    model = "fireworks_ai/test-tools-without-tool-choice"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "supports_function_calling": True,
            "supports_tool_choice": False,
        },
    )

    supported_params = config.get_supported_openai_params(model)

    assert "tools" in supported_params
    assert "parallel_tool_calls" in supported_params
    assert "tool_choice" not in supported_params


def test_get_model_info_respects_explicit_fireworks_capabilities():
    """Test that get_model_info preserves explicit capability flags from the model map."""
    model_info = get_model_info("fireworks_ai/accounts/fireworks/models/glm-5p1")

    assert model_info["supports_function_calling"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_tool_choice"] is True


def test_get_provider_info_omits_false_supports_reasoning(monkeypatch):
    """Test that Fireworks only overrides supports_reasoning for supported models."""
    config = FireworksAIConfig()
    model = "fireworks_ai/test-reasoning-false"
    monkeypatch.setitem(litellm.model_cost, model, {"supports_reasoning": False})

    info = config.get_provider_info(model)

    assert "supports_reasoning" not in info


def test_add_transform_inline_image_block_skips_data_urls():
    """
    data: URLs must not have #transform=inline appended — doing so corrupts the
    base64 payload and raises binascii.Error: Incorrect padding on the Fireworks side.
    Regression test for https://github.com/BerriAI/litellm/issues/23583
    """
    config = FireworksAIConfig()
    data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRgAB"

    # str branch
    str_content = {"type": "image_url", "image_url": data_url}
    result = config._add_transform_inline_image_block(
        str_content, model="gpt-4", disable_add_transform_inline_image_block=False
    )
    assert result["image_url"] == data_url, "data URL must not be modified (str branch)"

    # dict branch
    dict_content = {"type": "image_url", "image_url": {"url": data_url}}
    result = config._add_transform_inline_image_block(
        dict_content, model="gpt-4", disable_add_transform_inline_image_block=False
    )
    assert (
        result["image_url"]["url"] == data_url
    ), "data URL must not be modified (dict branch)"

    # regular https URL should still get the suffix
    https_content = {"type": "image_url", "image_url": "https://example.com/image.jpg"}
    result = config._add_transform_inline_image_block(
        https_content, model="gpt-4", disable_add_transform_inline_image_block=False
    )
    assert result["image_url"].endswith(
        "#transform=inline"
    ), "https URL should get #transform=inline"


@pytest.mark.parametrize(
    "api_base, expected_url_prefix",
    [
        (
            "https://api.fireworks.ai/inference/v1",
            "https://api.fireworks.ai/inference/v1/accounts/",
        ),
        (
            "https://api.fireworks.ai/inference/v1/",
            "https://api.fireworks.ai/inference/v1/accounts/",
        ),
        (
            "https://custom-host.example.com/v1",
            "https://custom-host.example.com/v1/accounts/",
        ),
        (
            "https://custom-host.example.com/api",
            "https://custom-host.example.com/api/v1/accounts/",
        ),
    ],
    ids=["default", "trailing-slash", "custom-with-v1", "custom-without-v1"],
)
def test_get_models_url_no_double_v1(api_base, expected_url_prefix):
    """Ensure get_models never produces a /v1/v1/ URL segment (fixes #23106)."""
    config = FireworksAIConfig()
    account_id = "fireworks"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [{"name": "accounts/fireworks/models/llama-v3-70b"}]
    }

    with (
        patch(
            "litellm.module_level_client.get", return_value=mock_response
        ) as mock_get,
        patch(
            "litellm.llms.fireworks_ai.chat.transformation.get_secret_str",
            side_effect=lambda key: {
                "FIREWORKS_API_KEY": "test-key",
                "FIREWORKS_API_BASE": api_base,
                "FIREWORKS_ACCOUNT_ID": account_id,
            }.get(key),
        ),
    ):
        result = config.get_models(api_key="test-key", api_base=api_base)

        called_url = mock_get.call_args.kwargs.get("url") or mock_get.call_args[1].get(
            "url", ""
        )
        assert "/v1/v1/" not in called_url, f"Double /v1/ detected in URL: {called_url}"
        assert called_url.startswith(
            expected_url_prefix
        ), f"URL {called_url} does not start with {expected_url_prefix}"
        assert result == ["fireworks_ai/accounts/fireworks/models/llama-v3-70b"]


def test_transform_messages_helper_removes_provider_specific_fields():
    """
    Test that _transform_messages_helper removes provider_specific_fields from messages.
    """
    config = FireworksAIConfig()
    # Simulated messages, as dicts, including provider_specific_fields
    messages = [
        {
            "role": "user",
            "content": "Hello!",
            "provider_specific_fields": {"extra": "should be removed"},
        },
        {
            "role": "assistant",
            "content": "Hi there!",
            "provider_specific_fields": {"more": "remove this"},
        },
        {
            "role": "user",
            "content": "How are you?",
            # no provider_specific_fields
        },
    ]
    # Call helper
    out = config._transform_messages_helper(
        messages, model="fireworks/test", litellm_params={}
    )
    for msg in out:
        assert "provider_specific_fields" not in msg


def test_unmapped_model_fallback_function_calling():
    """Test that a model not in model_cost still defaults to supporting function calling for Fireworks."""
    config = FireworksAIConfig()
    model = "fireworks_ai/unmapped-future-model"
    info = config.get_provider_info(model)
    assert info["supports_function_calling"] is True


def test_transform_messages_helper_strips_thinking_blocks():
    """thinking_blocks must not be forwarded to Fireworks chat completions."""
    config = FireworksAIConfig()
    messages = [
        {"role": "user", "content": "Translate a poem."},
        {
            "role": "assistant",
            "content": "I can help.",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "internal", "signature": ""}
            ],
        },
    ]
    out = config._transform_messages_helper(
        messages, model="accounts/fireworks/models/glm-5p1", litellm_params={}
    )
    assert "thinking_blocks" not in out[1]
    assert out[1]["content"] == "I can help."


# -----------------------------------------------------------------------------
# Regression tests for legacy / OpenAPI $ref defs in tool parameters.
#
# Fireworks (like Anthropic) only resolves `$defs` (JSON Schema 2020-12). Tools
# coming from MCP servers (legacy `definitions`) or OpenAPI-derived gateways
# such as AWS AgentCore (`components.schemas`) used to leave dangling `$ref`
# pointers, causing upstream "Error resolving schema reference" failures. See
# https://github.com/BerriAI/litellm/issues/26692.
# -----------------------------------------------------------------------------


def _assert_no_unresolved_refs(parameters: dict) -> None:
    blob = json.dumps(parameters)
    assert "$ref" not in blob, f"unresolved $ref in transformed parameters: {blob}"


def test_transform_tools_inlines_components_schemas_refs():
    """OpenAPI `components.schemas` $refs (AgentCore-style) must be inlined."""
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "slides_presentations_create",
                "description": "Create a Google Slides presentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "body": {"$ref": "#/components/schemas/Presentation"},
                    },
                    "required": ["body"],
                    "components": {
                        "schemas": {
                            "Presentation": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "presentationId": {"type": "string"},
                                },
                            }
                        }
                    },
                },
            },
        }
    ]

    out = config._transform_tools(tools)

    params = out[0]["function"]["parameters"]
    _assert_no_unresolved_refs(params)
    assert params["properties"]["body"] == {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "presentationId": {"type": "string"},
        },
    }
    assert "components" not in params


def test_transform_tools_inlines_legacy_definitions_refs():
    """Legacy draft-04 `definitions` $refs must be inlined."""
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_thing",
                "description": "Create a thing",
                "parameters": {
                    "type": "object",
                    "properties": {"thing": {"$ref": "#/definitions/Thing"}},
                    "definitions": {
                        "Thing": {
                            "type": "object",
                            "properties": {"id": {"type": "string"}},
                        }
                    },
                },
            },
        }
    ]

    out = config._transform_tools(tools)

    params = out[0]["function"]["parameters"]
    _assert_no_unresolved_refs(params)
    assert params["properties"]["thing"] == {
        "type": "object",
        "properties": {"id": {"type": "string"}},
    }
    assert "definitions" not in params


def test_transform_tools_preserves_native_dollar_defs():
    """`$defs` is JSON Schema 2020-12 native; Fireworks resolves it itself."""
    config = FireworksAIConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "native_defs_tool",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/A"}},
                    "$defs": {"A": {"type": "string"}},
                },
            },
        }
    ]

    out = config._transform_tools(tools)

    params = out[0]["function"]["parameters"]
    assert params["$defs"] == {"A": {"type": "string"}}
    assert params["properties"]["a"] == {"$ref": "#/$defs/A"}


def test_transform_tools_skips_non_function_tools():
    """Non-``function`` tools (e.g. provider-native tool types) must pass
    through ``_transform_tools`` untouched -- no ``strict`` pop, no $ref
    inlining, no error.
    """
    config = FireworksAIConfig()
    non_function_tool = {
        "type": "code_interpreter",
        "code_interpreter": {"some": "config"},
    }
    function_tool = {
        "type": "function",
        "function": {
            "name": "create_thing",
            "description": "Create a thing",
            "parameters": {
                "type": "object",
                "properties": {"thing": {"$ref": "#/definitions/Thing"}},
                "definitions": {
                    "Thing": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    }
                },
            },
            "strict": True,
        },
    }

    out = config._transform_tools([non_function_tool, function_tool])

    # Non-function tool is preserved verbatim.
    assert out[0] == {
        "type": "code_interpreter",
        "code_interpreter": {"some": "config"},
    }
    # Function tool still goes through both transformations: `strict` popped
    # and the legacy $ref inlined.
    assert "strict" not in out[1]["function"]
    inlined = out[1]["function"]["parameters"]
    assert "definitions" not in inlined
    assert inlined["properties"]["thing"] == {
        "type": "object",
        "properties": {"id": {"type": "string"}},
    }


def test_map_response_format_passes_json_schema_through_unchanged():
    """
    json_schema response_format must reach Fireworks unchanged.

    Regression guard for the prior downgrade to {type: json_object, schema: ...}
    which silently dropped `strict` and `name` and disabled grammar-guided
    decoding on the Fireworks side.
    """
    config = FireworksAIConfig()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "priority_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    }
                },
                "required": ["priority"],
                "additionalProperties": False,
            },
        },
    }

    result = config.map_openai_params(
        {"response_format": response_format},
        {},
        "fireworks_ai/accounts/fireworks/models/qwen3-32b",
        drop_params=False,
    )

    rf = result["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "priority_classification"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"] == response_format["json_schema"]["schema"]


def test_map_response_format_json_object_unchanged():
    """
    The plain json_object form keeps working as before.
    """
    config = FireworksAIConfig()
    result = config.map_openai_params(
        {"response_format": {"type": "json_object"}},
        {},
        "fireworks_ai/accounts/fireworks/models/qwen3-32b",
        drop_params=False,
    )
    assert result == {"response_format": {"type": "json_object"}}


def test_transform_request_routes_short_form_router_to_routers_path():
    """A bare router model name ending in -fast must be rewritten to the
    ``accounts/fireworks/routers/`` path, not the default ``models/`` path."""
    config = FireworksAIConfig()
    result = config.transform_request(
        model="glm-5p1-fast",
        messages=[{"role": "user", "content": "Hi"}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert result["model"] == "accounts/fireworks/routers/glm-5p1-fast"


def test_transform_request_routes_short_form_model_to_models_path():
    """A bare direct-model name must still be rewritten to the
    ``accounts/fireworks/models/`` path."""
    config = FireworksAIConfig()
    result = config.transform_request(
        model="glm-5p2",
        messages=[{"role": "user", "content": "Hi"}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert result["model"] == "accounts/fireworks/models/glm-5p2"
