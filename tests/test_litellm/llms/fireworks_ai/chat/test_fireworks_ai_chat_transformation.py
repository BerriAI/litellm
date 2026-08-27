import json
from unittest.mock import MagicMock, patch

import pytest

import litellm


from litellm import get_model_info, supports_reasoning, supports_vision
from litellm.llms.fireworks_ai.chat.transformation import FireworksAIConfig
from litellm.llms.fireworks_ai.common_utils import get_fireworks_session_id
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
    Message,
    ModelResponse,
)


@pytest.fixture(autouse=True)
def force_local_model_cost(monkeypatch):
    """Force local model cost map usage for all tests in this file."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # Refresh model_cost from local map
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map

    litellm.model_cost = get_model_cost_map(url=litellm.model_cost_map_url)


def test_validate_environment_sets_session_affinity_from_litellm_session_id():
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={"litellm_session_id": "session-123"},
        api_key="test-key",
    )

    assert headers["x-session-affinity"] == "session-123"


def test_validate_environment_sets_session_affinity_from_metadata_session_id():
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={"metadata": {"session_id": "metadata-session-123"}},
        api_key="test-key",
    )

    assert headers["x-session-affinity"] == "metadata-session-123"


def test_validate_environment_sets_session_affinity_from_session_id():
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={"session_id": "session-id-123"},
        api_key="test-key",
    )

    assert headers["x-session-affinity"] == "session-id-123"


def test_validate_environment_ignores_trace_id_for_session_affinity():
    """A trace id must not become the session id.

    litellm_trace_id defaults to a fresh uuid4 per request, so pinning
    x-session-affinity to it sent every request to a different Fireworks node and
    prompt caching never hit (cached_tokens stayed 0 across identical prompts).
    """
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={"litellm_trace_id": "trace-id-123"},
        api_key="test-key",
    )

    assert "x-session-affinity" not in headers


def test_validate_environment_prefers_session_id_over_trace_id():
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={
            "litellm_session_id": "session-123",
            "litellm_trace_id": "trace-id-123",
        },
        api_key="test-key",
    )

    assert headers["x-session-affinity"] == "session-123"


def test_validate_environment_does_not_set_session_affinity_without_session_id():
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-key",
    )

    assert "x-session-affinity" not in headers


def test_validate_environment_preserves_explicit_session_affinity_header():
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={"x-session-affinity": "explicit-session"},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={"litellm_session_id": "session-123"},
        api_key="test-key",
    )

    assert headers["x-session-affinity"] == "explicit-session"


def test_validate_environment_sets_json_content_type():
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-key",
    )

    assert headers["Content-Type"] == "application/json"


def test_validate_environment_preserves_explicit_content_type():
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={"content-type": "multipart/form-data"},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-key",
    )

    assert headers["content-type"] == "multipart/form-data"
    assert "Content-Type" not in headers


def test_validate_environment_sets_json_content_type_with_session_affinity():
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={"litellm_session_id": "session-123"},
        api_key="test-key",
    )

    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["x-session-affinity"] == "session-123"


def test_validate_environment_resolves_api_key_from_env_and_sets_content_type(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw-env-key")
    config = FireworksAIConfig()

    headers = config.validate_environment(
        headers={},
        model="accounts/fireworks/models/test-model",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    assert headers["Authorization"] == "Bearer fw-env-key"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_raises_without_api_key(monkeypatch):
    for env_var in (
        "FIREWORKS_API_KEY",
        "FIREWORKS_AI_API_KEY",
        "FIREWORKSAI_API_KEY",
        "FIREWORKS_AI_TOKEN",
    ):
        monkeypatch.delenv(env_var, raising=False)
    config = FireworksAIConfig()

    with pytest.raises(ValueError, match="FIREWORKS_API_KEY is not set"):
        config.validate_environment(
            headers={},
            model="accounts/fireworks/models/test-model",
            messages=[],
            optional_params={},
            litellm_params={},
        )


def test_get_fireworks_session_id_prefers_litellm_session_id_over_trace_id():
    assert (
        get_fireworks_session_id(
            {"litellm_session_id": "session-123", "litellm_trace_id": "trace-123"}
        )
        == "session-123"
    )


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
            supports_reasoning(model=model, custom_llm_provider="fireworks_ai") is True
        ), f"{model} should support reasoning_effort"

    for model in unsupported_models:
        assert (
            supports_reasoning(model=model, custom_llm_provider="fireworks_ai") is False
        ), f"{model} should not support reasoning_effort"


def test_get_supported_openai_params_reasoning_effort():
    """Test that reasoning_effort is only included in supported params for models that support it."""
    config = FireworksAIConfig()

    supported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/glm-5p1"
    )
    assert "reasoning_effort" in supported_params
    assert "thinking" in supported_params

    unsupported_params = config.get_supported_openai_params(
        "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct"
    )
    assert "reasoning_effort" not in unsupported_params
    assert "thinking" not in unsupported_params


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
            "reasoning_content": "internal",
        },
    ]
    out = config._transform_messages_helper(
        messages, model="accounts/fireworks/models/glm-5p1", litellm_params={}
    )
    assert "thinking_blocks" not in out[1]
    assert "reasoning_content" not in out[1]
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


def _make_fireworks_raw_response(body: dict) -> MagicMock:
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = body
    mock.text = json.dumps(body)
    mock.headers = {}
    return mock


_BASE_CHAT_COMPLETION_RESPONSE: dict = {
    "id": "resp-test",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "glm-5p1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


def _run_transform_response(response_body: dict) -> ModelResponse:
    config = FireworksAIConfig()
    raw_response = _make_fireworks_raw_response(response_body)
    logging_obj = MagicMock()
    return config.transform_response(
        model="accounts/fireworks/models/glm-5p1",
        raw_response=raw_response,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "Hi"}],
        optional_params={},
        litellm_params={},
        encoding=None,
        api_key="test-key",
    )


_REASONING_MODEL = "fireworks_ai/accounts/fireworks/models/glm-5p1"
_NON_REASONING_MODEL = "fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct"


def test_get_supported_openai_params_includes_all_fireworks_params():
    config = FireworksAIConfig()
    params = config.get_supported_openai_params(_REASONING_MODEL)

    required = [
        "seed",
        "top_logprobs",
        "min_p",
        "typical_p",
        "repetition_penalty",
        "mirostat_target",
        "mirostat_lr",
        "logit_bias",
        "echo",
        "echo_last",
        "ignore_eos",
        "prompt_cache_key",
        "prompt_cache_isolation_key",
        "raw_output",
        "perf_metrics_in_response",
        "return_token_ids",
        "safe_tokenization",
        "service_tier",
        "speculation",
        "prediction",
        "stream_options",
        "sampling_mask",
        "thinking",
        "reasoning_history",
    ]
    missing = [p for p in required if p not in params]
    assert missing == [], f"Missing params: {missing}"


def test_native_openai_params_flow_end_to_end_with_drop_params_false():
    """
    The OpenAI-native params Fireworks supports (seed, top_logprobs, logit_bias,
    prompt_cache_key, service_tier, prediction) previously hit
    ``UnsupportedParamsError`` with ``drop_params=False`` because they were absent
    from ``get_supported_openai_params``. Listing them must let them survive the
    ``get_optional_params`` gate and reach the request, not just appear in the
    supported list. Asserting via ``get_optional_params`` (the real gate) rather
    than ``map_openai_params`` catches a revert of the supported-params additions,
    which a list-membership check would not.
    """
    native = {
        "seed": 42,
        "top_logprobs": 3,
        "logit_bias": {"1": 1},
        "prompt_cache_key": "cache-key",
        "service_tier": "auto",
        "prediction": {"type": "content", "content": "x"},
    }
    optional_params = litellm.get_optional_params(
        model="accounts/fireworks/models/llama-v3-70b-instruct",
        custom_llm_provider="fireworks_ai",
        drop_params=False,
        **native,
    )
    for key, value in native.items():
        assert optional_params.get(key) == value


def test_prompt_truncate_len_correct_name():
    config = FireworksAIConfig()
    params = config.get_supported_openai_params(_REASONING_MODEL)
    assert "prompt_truncate_len" in params
    assert "prompt_truncate_length" not in params

    result = config.map_openai_params(
        {"prompt_truncate_len": 4096},
        {},
        _REASONING_MODEL,
        drop_params=False,
    )
    assert result == {"prompt_truncate_len": 4096}


def test_stream_options_include_usage_auto_injected():
    config = FireworksAIConfig()
    result = config.transform_request(
        model="accounts/fireworks/models/glm-5p1",
        messages=[{"role": "user", "content": "Hi"}],
        optional_params={"stream": True},
        litellm_params={},
        headers={},
    )
    assert result["stream_options"] == {"include_usage": True}


def test_stream_options_not_injected_when_not_streaming():
    config = FireworksAIConfig()
    result = config.transform_request(
        model="accounts/fireworks/models/glm-5p1",
        messages=[{"role": "user", "content": "Hi"}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert "stream_options" not in result


def test_stream_options_preserves_user_override():
    config = FireworksAIConfig()
    result = config.transform_request(
        model="accounts/fireworks/models/glm-5p1",
        messages=[{"role": "user", "content": "Hi"}],
        optional_params={"stream": True, "stream_options": {"include_usage": False}},
        litellm_params={},
        headers={},
    )
    assert result["stream_options"]["include_usage"] is False


def test_reasoning_history_in_supported_params():
    config = FireworksAIConfig()
    reasoning_params = config.get_supported_openai_params(_REASONING_MODEL)
    assert "reasoning_history" in reasoning_params

    non_reasoning_params = config.get_supported_openai_params(_NON_REASONING_MODEL)
    assert "reasoning_history" not in non_reasoning_params


def test_thinking_param_passthrough():
    config = FireworksAIConfig()
    thinking = {"type": "disabled"}
    result = config.map_openai_params(
        {"thinking": thinking},
        {},
        _REASONING_MODEL,
        drop_params=False,
    )
    assert result == {"thinking": thinking}


def test_thinking_and_reasoning_effort_conflict_rejected():
    config = FireworksAIConfig()
    with pytest.raises(
        litellm.BadRequestError,
        match="does not support specifying both `thinking` and `reasoning_effort`",
    ):
        config.map_openai_params(
            {
                "thinking": {"type": "enabled", "budget_tokens": 4096},
                "reasoning_effort": "medium",
            },
            {},
            _REASONING_MODEL,
            drop_params=False,
        )


def test_minimax_m3_supports_vision_from_model_map():
    config = FireworksAIConfig()

    for model in [
        "fireworks_ai/accounts/fireworks/models/minimax-m3",
        "fireworks_ai/minimax-m3",
    ]:
        assert supports_vision(model=model, custom_llm_provider="fireworks_ai") is True
        assert config.get_provider_info(model)["supports_vision"] is True


def test_transform_messages_helper_rejects_file_blocks():
    config = FireworksAIConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "file",
                    "file": {
                        "file_data": "data:application/pdf;base64,JVBERi0xLjQKJSVFT0YK",
                        "filename": "tiny.pdf",
                    },
                },
                {"type": "text", "text": "Describe this"},
            ],
        }
    ]

    with pytest.raises(
        litellm.BadRequestError,
        match="Fireworks AI chat completions does not support file content blocks",
    ):
        config._transform_messages_helper(
            messages, model="accounts/fireworks/models/kimi-k2p6", litellm_params={}
        )


def test_transform_messages_helper_rejects_non_vision_image_inputs():
    config = FireworksAIConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="
                    },
                },
            ],
        }
    ]

    with pytest.raises(litellm.BadRequestError, match="does not support image inputs"):
        config._transform_messages_helper(
            messages, model="accounts/fireworks/models/glm-5p2", litellm_params={}
        )


def test_transform_messages_helper_allows_vision_image_inputs():
    config = FireworksAIConfig()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="
                    },
                },
            ],
        }
    ]

    out = config._transform_messages_helper(
        messages, model="accounts/fireworks/models/minimax-m3", litellm_params={}
    )
    assert out == messages


def test_image_inputs_not_rejected_for_fuzzy_non_vision_match():
    """
    A custom/fine-tuned model id that hyphen-matches a known non-vision model
    (glm-5p2 has supports_vision=False) must not inherit that False via the
    substring fallback and hard-reject valid image_url blocks. The capability
    gate for rejection uses an exact cost-map match; the fuzzy fallback stays a
    soft signal only, so an unmapped vision-capable deployment is not blocked.
    """
    config = FireworksAIConfig()
    custom_model = "accounts/myorg/models/custom-glm-5p2"

    assert config._get_model_cost_capability(custom_model, "supports_vision") is False
    assert (
        config._get_model_cost_capability_exact(custom_model, "supports_vision") is None
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAE="
                    },
                },
            ],
        }
    ]
    out = config._transform_messages_helper(
        messages, model=custom_model, litellm_params={}
    )
    assert out == messages


def test_transform_messages_helper_skips_non_dict_content():
    config = FireworksAIConfig()
    messages = [
        {
            "role": "user",
            "content": ["just a string", {"type": "text", "text": "hello"}],
        }
    ]

    out = config._transform_messages_helper(
        messages, model="accounts/fireworks/models/glm-5p2", litellm_params={}
    )
    assert out == messages


def test_transform_messages_helper_no_transform_inline():
    config = FireworksAIConfig()
    url = "https://example.com/image.jpg"
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": url}],
        }
    ]
    out = config._transform_messages_helper(
        messages, model="accounts/fireworks/models/minimax-m3", litellm_params={}
    )
    block = out[0]["content"][0]
    assert block["image_url"] == url
    assert "#transform=inline" not in block["image_url"]


def test_get_provider_info_vision_from_model_cost(monkeypatch):
    config = FireworksAIConfig()

    vision_model = "fireworks_ai/test-vision-from-cost"
    monkeypatch.setitem(
        litellm.model_cost,
        vision_model,
        {"supports_vision": True, "supports_pdf_input": True},
    )
    info = config.get_provider_info(vision_model)
    assert info["supports_vision"] is True
    assert info["supports_pdf_input"] is True

    no_vision_model = "fireworks_ai/test-no-vision-from-cost"
    monkeypatch.setitem(litellm.model_cost, no_vision_model, {})
    info_no_vision = config.get_provider_info(no_vision_model)
    assert info_no_vision.get("supports_vision") is not True
    assert "supports_pdf_input" not in info_no_vision


def test_reasoning_effort_boolean_true_to_medium():
    config = FireworksAIConfig()
    result = config.map_openai_params(
        {"reasoning_effort": True},
        {},
        _REASONING_MODEL,
        drop_params=False,
    )
    assert result["reasoning_effort"] == "medium"


def test_reasoning_effort_boolean_false_to_none():
    config = FireworksAIConfig()
    result = config.map_openai_params(
        {"reasoning_effort": False},
        {},
        _REASONING_MODEL,
        drop_params=False,
    )
    assert result["reasoning_effort"] == "none"


def test_reasoning_effort_string_passthrough():
    config = FireworksAIConfig()
    result = config.map_openai_params(
        {"reasoning_effort": "high"},
        {},
        _REASONING_MODEL,
        drop_params=False,
    )
    assert result["reasoning_effort"] == "high"


def test_reasoning_effort_integer_passthrough():
    config = FireworksAIConfig()
    result = config.map_openai_params(
        {"reasoning_effort": 1000},
        {},
        _REASONING_MODEL,
        drop_params=False,
    )
    assert result["reasoning_effort"] == 1000
    assert isinstance(result["reasoning_effort"], int)


def test_reasoning_effort_auto_dropped_to_model_default():
    config = FireworksAIConfig()
    result = config.map_openai_params(
        {"reasoning_effort": "auto"},
        {},
        _REASONING_MODEL,
        drop_params=False,
    )
    assert "reasoning_effort" not in result


def test_transform_response_captures_perf_metrics():
    body = {
        **_BASE_CHAT_COMPLETION_RESPONSE,
        "perf_metrics": {"prompt-tokens": 10},
    }
    result = _run_transform_response(body)
    assert result._hidden_params["fireworks_perf_metrics"] == {"prompt-tokens": 10}


def test_transform_response_captures_prompt_token_ids():
    body = {
        **_BASE_CHAT_COMPLETION_RESPONSE,
        "prompt_token_ids": [1, 2, 3],
    }
    result = _run_transform_response(body)
    assert result._hidden_params["fireworks_prompt_token_ids"] == [1, 2, 3]


def test_transform_response_captures_raw_output():
    raw_output = {
        "prompt_fragments": [],
        "prompt_token_ids": [],
        "completion": "test",
    }
    body = {
        **_BASE_CHAT_COMPLETION_RESPONSE,
        "choices": [
            {
                **_BASE_CHAT_COMPLETION_RESPONSE["choices"][0],
                "raw_output": raw_output,
            }
        ],
    }
    result = _run_transform_response(body)
    assert result._hidden_params["fireworks_raw_outputs"] == [raw_output]


def test_transform_response_captures_token_ids():
    body = {
        **_BASE_CHAT_COMPLETION_RESPONSE,
        "choices": [
            {
                **_BASE_CHAT_COMPLETION_RESPONSE["choices"][0],
                "token_ids": [4, 5, 6],
            }
        ],
    }
    result = _run_transform_response(body)
    assert result._hidden_params["fireworks_token_ids"] == [[4, 5, 6]]


def test_streaming_surfaces_fireworks_response_fields():
    """
    The Fireworks-specific response fields captured into _hidden_params for
    non-streaming calls must also reach streamed responses. They ride the
    streamed chunks' provider_specific_fields (litellm rebuilds each streamed
    chunk, so per-chunk _hidden_params does not survive): per-choice
    token_ids/raw_output on the content chunk, response-level
    perf_metrics/prompt_token_ids on the final usage chunk. Driving the real
    litellm.completion(stream=True) path also covers the get_model_response_iterator
    wiring; dropping the Fireworks iterator would leave these fields unset.
    """
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    model = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    raw_output = {"completion": "Hi"}
    sse_lines = [
        "data: "
        + json.dumps(
            {
                "id": "stream-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hi"},
                        "token_ids": [123],
                        "raw_output": raw_output,
                    }
                ],
            }
        ),
        "data: "
        + json.dumps(
            {
                "id": "stream-1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
                "perf_metrics": {"prompt-tokens": 5},
                "prompt_token_ids": [1, 2, 3],
            }
        ),
        "data: [DONE]",
    ]

    raw_response = MagicMock()
    raw_response.status_code = 200
    raw_response.headers = {}
    raw_response.iter_lines = lambda: iter(sse_lines)

    client = HTTPHandler()
    with patch.object(client, "post", return_value=raw_response):
        stream = litellm.completion(
            model=f"fireworks_ai/{model}",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            api_key="fw-test-key",
            client=client,
        )
        surfaced: dict = {}
        for chunk in stream:
            fields = getattr(chunk, "provider_specific_fields", None) or {}
            surfaced.update(
                {k: v for k, v in fields.items() if k.startswith("fireworks_")}
            )

    assert surfaced["fireworks_token_ids"] == [[123]]
    assert surfaced["fireworks_raw_outputs"] == [raw_output]
    assert surfaced["fireworks_perf_metrics"] == {"prompt-tokens": 5}
    assert surfaced["fireworks_prompt_token_ids"] == [1, 2, 3]


def test_transform_request_routes_router_slug():
    config = FireworksAIConfig()

    data = config.transform_request(
        model="routers/glm-latest",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert data["model"] == "accounts/fireworks/routers/glm-latest"


def test_transform_request_bare_slug_stays_model():
    config = FireworksAIConfig()

    data = config.transform_request(
        model="glm-4p6",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert data["model"] == "accounts/fireworks/models/glm-4p6"


def test_transform_request_direct_route_passthrough():
    config = FireworksAIConfig()
    model = "accounts/fireworks/models/qwen2p5-coder-7b#accounts/gitlab/deployments/2fb7764c"

    data = config.transform_request(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert data["model"] == model


def test_map_extra_body_params_translates_truncate_prompt_tokens():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"truncate_prompt_tokens": 4096}}, _REASONING_MODEL
    )
    assert result == {"prompt_truncate_len": 4096}


def test_map_extra_body_params_truncate_prompt_tokens_native_wins():
    config = FireworksAIConfig()
    top_level = config.map_extra_body_params(
        {"prompt_truncate_len": 2048, "extra_body": {"truncate_prompt_tokens": 4096}},
        _REASONING_MODEL,
    )
    assert top_level == {"prompt_truncate_len": 2048}

    nested = config.map_extra_body_params(
        {"extra_body": {"truncate_prompt_tokens": 4096, "prompt_truncate_len": 2048}},
        _REASONING_MODEL,
    )
    assert nested == {"extra_body": {"prompt_truncate_len": 2048}}


def test_map_extra_body_params_chat_template_kwargs_enable_thinking():
    config = FireworksAIConfig()
    disabled = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        _REASONING_MODEL,
    )
    assert disabled == {"reasoning_effort": "none"}

    enabled = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        _REASONING_MODEL,
    )
    assert enabled == {}


def test_map_extra_body_params_chat_template_kwargs_thinking_alias():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"thinking": False}}},
        _REASONING_MODEL,
    )
    assert result == {"reasoning_effort": "none"}


def test_map_extra_body_params_chat_template_kwargs_enable_thinking_wins_over_thinking():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True, "thinking": False}}},
        _REASONING_MODEL,
    )
    assert result == {}


def test_map_extra_body_params_chat_template_kwargs_reasoning_budget():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"reasoning_budget": 512}}},
        _REASONING_MODEL,
    )
    assert result == {"reasoning_effort": 512}


def test_map_extra_body_params_chat_template_kwargs_budget_ignored_when_thinking_off():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False, "reasoning_budget": 512}}},
        _REASONING_MODEL,
    )
    assert result == {"reasoning_effort": "none"}


def test_map_extra_body_params_chat_template_kwargs_low_effort():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"low_effort": True}}},
        _REASONING_MODEL,
    )
    assert result == {"reasoning_effort": "low"}

    budget_wins = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"low_effort": True, "reasoning_budget": 256}}},
        _REASONING_MODEL,
    )
    assert budget_wins == {"reasoning_effort": 256}


def test_map_extra_body_params_chat_template_kwargs_effort_keys_dropped_for_non_reasoning_model():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"reasoning_budget": 512, "low_effort": True}}},
        _NON_REASONING_MODEL,
    )
    assert result == {}


def test_map_extra_body_params_chat_template_kwargs_native_reasoning_effort_wins():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {
            "reasoning_effort": "high",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
        _REASONING_MODEL,
    )
    assert result == {"reasoning_effort": "high"}


def test_map_extra_body_params_chat_template_kwargs_native_thinking_wins():
    config = FireworksAIConfig()
    thinking = {"type": "enabled", "budget_tokens": 4096}
    result = config.map_extra_body_params(
        {
            "thinking": thinking,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        },
        _REASONING_MODEL,
    )
    assert result == {"thinking": thinking}


def test_map_extra_body_params_chat_template_kwargs_extra_body_thinking_wins():
    config = FireworksAIConfig()
    thinking = {"type": "enabled", "budget_tokens": 4096}
    result = config.map_extra_body_params(
        {"extra_body": {"thinking": thinking, "chat_template_kwargs": {"enable_thinking": False}}},
        _REASONING_MODEL,
    )
    assert result == {"extra_body": {"thinking": thinking}}


def test_map_extra_body_params_chat_template_kwargs_extra_body_reasoning_effort_wins():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"reasoning_effort": "high", "chat_template_kwargs": {"enable_thinking": False}}},
        _REASONING_MODEL,
    )
    assert result == {"extra_body": {"reasoning_effort": "high"}}


def test_map_extra_body_params_chat_template_kwargs_dropped_for_non_reasoning_model():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False, "custom_flag": 1}}},
        _NON_REASONING_MODEL,
    )
    assert result == {}


def test_map_extra_body_params_non_dict_chat_template_kwargs_dropped():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"chat_template_kwargs": "enable_thinking"}},
        _REASONING_MODEL,
    )
    assert result == {}


def test_map_extra_body_params_guided_json():
    config = FireworksAIConfig()
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    result = config.map_extra_body_params(
        {"extra_body": {"guided_json": schema}}, _REASONING_MODEL
    )
    assert result == {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": schema},
        }
    }


def test_map_extra_body_params_guided_grammar_and_choice():
    config = FireworksAIConfig()
    grammar = config.map_extra_body_params(
        {"extra_body": {"guided_grammar": "root ::= 'hello'"}}, _REASONING_MODEL
    )
    assert grammar == {
        "response_format": {"type": "grammar", "grammar": "root ::= 'hello'"}
    }

    choice = config.map_extra_body_params(
        {"extra_body": {"guided_choice": ["yes", "no"]}}, _REASONING_MODEL
    )
    assert choice == {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "choice",
                "schema": {"type": "string", "enum": ["yes", "no"]},
            },
        }
    }


def test_map_extra_body_params_guided_native_response_format_wins():
    config = FireworksAIConfig()
    top_level = config.map_extra_body_params(
        {
            "response_format": {"type": "json_object"},
            "extra_body": {"guided_json": {"type": "object"}},
        },
        _REASONING_MODEL,
    )
    assert top_level == {"response_format": {"type": "json_object"}}

    nested_format = {"type": "json_object"}
    nested = config.map_extra_body_params(
        {"extra_body": {"guided_json": {"type": "object"}, "response_format": nested_format}},
        _REASONING_MODEL,
    )
    assert nested == {"extra_body": {"response_format": nested_format}}


def test_map_extra_body_params_top_level_response_format_beats_nested():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {
            "response_format": {"type": "json_object"},
            "extra_body": {
                "guided_json": {"type": "object"},
                "response_format": {"type": "json_schema", "json_schema": {"schema": {}}},
            },
        },
        _REASONING_MODEL,
    )
    assert result == {"response_format": {"type": "json_object"}}


def test_map_extra_body_params_multiple_guided_params_priority_order():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"guided_grammar": "root ::= 'x'", "guided_json": {"type": "object"}}},
        _REASONING_MODEL,
    )
    assert result == {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": {"type": "object"}},
        }
    }


@pytest.mark.parametrize(
    "param,value",
    [
        ("stop_token_ids", [1, 2]),
        ("include_stop_str_in_output", True),
        ("skip_special_tokens", False),
        ("spaces_between_special_tokens", True),
        ("best_of", 2),
        ("use_beam_search", True),
        ("guided_decoding_backend", "outlines"),
        ("guided_regex", "[0-9]+"),
        ("add_generation_prompt", True),
        ("continue_final_message", True),
        ("add_special_tokens", False),
        ("detokenize", True),
        ("allowed_token_ids", [1]),
        ("bad_words", ["foo"]),
        ("include_reasoning", False),
        ("nvext", {"verbosity": 1}),
    ],
)
def test_map_extra_body_params_strips_unsupported_nim_vllm_params(param, value, caplog):
    import logging

    config = FireworksAIConfig()
    with caplog.at_level(logging.DEBUG):
        result = config.map_extra_body_params(
            {"extra_body": {param: value}}, _REASONING_MODEL
        )
    assert result == {}
    assert param in caplog.text


def test_map_extra_body_params_preserves_unknown_passthrough():
    config = FireworksAIConfig()
    result = config.map_extra_body_params(
        {"extra_body": {"top_k": 40, "some_future_param": "x", "truncate_prompt_tokens": 100}},
        _REASONING_MODEL,
    )
    assert result == {
        "prompt_truncate_len": 100,
        "extra_body": {"top_k": 40, "some_future_param": "x"},
    }


def test_map_extra_body_params_no_extra_body():
    config = FireworksAIConfig()
    assert config.map_extra_body_params({}, _REASONING_MODEL) == {}
    unchanged = {"temperature": 0.5, "extra_body": None}
    assert config.map_extra_body_params(unchanged, _REASONING_MODEL) == unchanged


def test_nim_vllm_extras_translated_end_to_end_in_request_body():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    model = "accounts/fireworks/models/glm-5p1"
    body = {
        "id": "chat-1",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    raw_response = MagicMock()
    raw_response.status_code = 200
    raw_response.headers = {}
    raw_response.text = json.dumps(body)
    raw_response.json = lambda: body

    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = raw_response
    litellm.completion(
        model=f"fireworks_ai/{model}",
        messages=[{"role": "user", "content": "hi"}],
        api_key="fw-test-key",
        client=client,
        truncate_prompt_tokens=4096,
        chat_template_kwargs={"enable_thinking": False},
        min_tokens=10,
        include_reasoning=False,
        top_k=40,
    )

    request_body = json.loads(client.post.call_args.kwargs["data"])
    assert request_body["prompt_truncate_len"] == 4096
    assert "truncate_prompt_tokens" not in request_body
    assert request_body["reasoning_effort"] == "none"
    assert "chat_template_kwargs" not in request_body
    assert "include_reasoning" not in request_body
    assert request_body["min_tokens"] == 10
    assert request_body["top_k"] == 40


def test_in_schema_unsupported_params_still_raise():
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.get_optional_params(
            model="accounts/fireworks/models/llama-v3-70b-instruct",
            custom_llm_provider="fireworks_ai",
            drop_params=False,
            store=True,
        )
    optional_params = litellm.get_optional_params(
        model="accounts/fireworks/models/llama-v3-70b-instruct",
        custom_llm_provider="fireworks_ai",
        drop_params=True,
        store=True,
    )
    assert "store" not in optional_params
