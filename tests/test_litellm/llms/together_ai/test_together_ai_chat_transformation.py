"""Tests for litellm/llms/together_ai/chat.py"""

from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.together_ai.chat import TogetherAIConfig

# A model that is deliberately absent from model_prices_and_context_window.json: Together
# ships models faster than the map can follow, and those calls must keep working.
UNMAPPED_MODEL = "together_ai/some-org/Brand-New-Model-42B"
# The one mapped entry that opts out of tool calling.
NO_TOOLS_MODEL = "together_ai/Qwen/Qwen3-235B-A22B-fp8-tput"
REASONING_MODEL = "together_ai/openai/gpt-oss-120b"
NON_REASONING_MODEL = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
CACHED_INPUT_MODEL = "together_ai/moonshotai/Kimi-K3"

TOGETHER_ENV_VARS = (
    "TOGETHER_API_KEY",
    "TOGETHER_AI_API_KEY",
    "TOGETHERAI_API_KEY",
    "TOGETHER_AI_TOKEN",
    "TOGETHER_AI_API_BASE",
)


@pytest.fixture
def no_together_env(monkeypatch):
    for name in TOGETHER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    import litellm

    monkeypatch.setattr(litellm, "togetherai_api_key", None, raising=False)
    return monkeypatch


def map_params(model, optional_params=None, **non_default_params):
    return TogetherAIConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params if optional_params is not None else {},
        model=model,
        drop_params=False,
    )


def test_unmapped_model_keeps_tools_and_structured_outputs(local_model_cost_map):
    """An id the cost map has never seen must still advertise Together's real feature set.

    Before the provider-level baseline, `supports_function_calling` resolved to "unknown",
    which the param gate read as "unsupported", so `tools`, `tool_choice`, `function_call`
    and `response_format` were stripped from every request to a newly launched model.
    """
    supported = TogetherAIConfig().get_supported_openai_params(UNMAPPED_MODEL)

    assert "tools" in supported
    assert "tool_choice" in supported
    assert "function_call" in supported
    assert "response_format" in supported
    assert local_model_cost_map.supports_function_calling(UNMAPPED_MODEL, "together_ai") is True


def test_model_map_can_still_opt_a_model_out_of_tool_calling(local_model_cost_map):
    supported = TogetherAIConfig().get_supported_openai_params(NO_TOOLS_MODEL)

    assert "tools" not in supported
    assert "tool_choice" not in supported
    assert "function_call" not in supported
    assert "response_format" not in supported
    assert local_model_cost_map.supports_function_calling(NO_TOOLS_MODEL, "together_ai") is False


def test_provider_info_never_invents_reasoning_or_vision(local_model_cost_map):
    """The baseline covers what every model shares; per-model features stay unknown."""
    unmapped = TogetherAIConfig().get_provider_info(UNMAPPED_MODEL)

    assert unmapped["supports_function_calling"] is True
    assert unmapped["supports_tool_choice"] is True
    assert unmapped["supports_response_schema"] is True
    assert unmapped["supports_reasoning"] is None
    assert unmapped["supports_vision"] is None

    mapped = TogetherAIConfig().get_provider_info(CACHED_INPUT_MODEL)
    assert mapped["supports_reasoning"] is True
    assert mapped["supports_vision"] is True


def test_prompt_caching_follows_the_cached_input_rate(local_model_cost_map):
    """Together caches prefixes with no opt-in, and only some models publish a cached rate."""
    assert TogetherAIConfig().get_provider_info(CACHED_INPUT_MODEL)["supports_prompt_caching"] is True
    assert TogetherAIConfig().get_provider_info(NON_REASONING_MODEL)["supports_prompt_caching"] is None


def test_reasoning_params_are_offered_only_to_reasoning_models(local_model_cost_map):
    reasoning_supported = TogetherAIConfig().get_supported_openai_params(REASONING_MODEL)
    text_only = TogetherAIConfig().get_supported_openai_params(NON_REASONING_MODEL)

    assert "reasoning_effort" in reasoning_supported
    assert "thinking" in reasoning_supported
    assert "reasoning_effort" not in text_only
    assert "thinking" not in text_only


def test_native_params_are_advertised(local_model_cost_map):
    supported = TogetherAIConfig().get_supported_openai_params(UNMAPPED_MODEL)

    for param in (
        "top_k",
        "min_p",
        "repetition_penalty",
        "echo",
        "context_length_exceeded_behavior",
        "safety_model",
        "chat_template_kwargs",
        "reasoning",
    ):
        assert param in supported, param


def test_reasoning_effort_none_disables_thinking(local_model_cost_map):
    """Together turns thinking off through `reasoning`, not through `reasoning_effort`.

    `reasoning` is not an OpenAI param, so it has to ride in `extra_body`: handed to the
    SDK as a top-level keyword it raises TypeError before the request is ever sent.
    """
    mapped = map_params(REASONING_MODEL, reasoning_effort="none")

    assert mapped["extra_body"]["reasoning"] == {"enabled": False}
    assert "reasoning" not in mapped
    assert "reasoning_effort" not in mapped


def test_reasoning_effort_minimal_maps_to_low(local_model_cost_map):
    """Together's effort scale is low/medium/high, so `minimal` has to land on `low`."""
    assert map_params(REASONING_MODEL, reasoning_effort="minimal")["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    "effort",
    ["low", "medium", "high"],
)
def test_supported_reasoning_effort_values_pass_through(local_model_cost_map, effort):
    mapped = map_params(REASONING_MODEL, reasoning_effort=effort)

    assert mapped["reasoning_effort"] == effort
    assert "reasoning" not in mapped


@pytest.mark.parametrize(
    "thinking, expected_enabled",
    [
        ({"type": "enabled", "budget_tokens": 1024}, True),
        ({"type": "disabled"}, False),
    ],
)
def test_anthropic_thinking_param_becomes_the_reasoning_toggle(local_model_cost_map, thinking, expected_enabled):
    mapped = map_params(REASONING_MODEL, thinking=thinking)

    assert mapped["extra_body"]["reasoning"] == {"enabled": expected_enabled}
    assert "reasoning" not in mapped
    assert "thinking" not in mapped


def test_native_reasoning_param_wins_over_thinking(local_model_cost_map):
    mapped = map_params(REASONING_MODEL, reasoning={"enabled": True}, thinking={"type": "disabled"})

    assert mapped["extra_body"]["reasoning"] == {"enabled": True}
    assert "reasoning" not in mapped
    assert "thinking" not in mapped


def test_reasoning_toggle_joins_an_existing_extra_body(local_model_cost_map):
    mapped = map_params(
        REASONING_MODEL,
        optional_params={"extra_body": {"safety_model": "meta-llama/Llama-Guard-4-12B"}},
        reasoning_effort="none",
    )

    assert mapped["extra_body"] == {
        "safety_model": "meta-llama/Llama-Guard-4-12B",
        "reasoning": {"enabled": False},
    }


def test_max_completion_tokens_becomes_max_tokens(local_model_cost_map):
    """Together reads `max_tokens`; it accepts `max_completion_tokens` and ignores it."""
    mapped = map_params(UNMAPPED_MODEL, max_completion_tokens=512)

    assert mapped["max_tokens"] == 512
    assert "max_completion_tokens" not in mapped


def test_explicit_max_tokens_is_not_overwritten(local_model_cost_map):
    mapped = map_params(UNMAPPED_MODEL, max_tokens=128, max_completion_tokens=512)

    assert mapped["max_tokens"] == 128
    assert "max_completion_tokens" not in mapped


@pytest.mark.parametrize(
    "passed, expected",
    [
        ({"logprobs": True, "top_logprobs": 3}, 3),
        ({"logprobs": True}, 1),
        ({"top_logprobs": 5}, 5),
        ({"logprobs": 2}, 2),
    ],
)
def test_openai_logprobs_become_a_token_count(local_model_cost_map, passed, expected):
    """Together's `logprobs` is the number of top tokens to return, not a boolean."""
    mapped = map_params(UNMAPPED_MODEL, **passed)

    assert mapped["logprobs"] == expected
    assert "top_logprobs" not in mapped


def test_logprobs_false_is_dropped(local_model_cost_map):
    mapped = map_params(UNMAPPED_MODEL, logprobs=False)

    assert "logprobs" not in mapped
    assert "top_logprobs" not in mapped


def test_text_response_format_is_stripped(local_model_cost_map):
    """Together rejects the no-op `{"type": "text"}` response format."""
    assert "response_format" not in map_params(UNMAPPED_MODEL, response_format={"type": "text"})
    assert map_params(UNMAPPED_MODEL, response_format={"type": "json_object"})["response_format"] == {
        "type": "json_object"
    }


def test_request_body_reaching_the_openai_sdk(local_model_cost_map, no_together_env):
    """End to end: what a caller passes has to arrive in the shape Together documents."""
    import openai

    client = openai.OpenAI(api_key="test-key", base_url="https://api.together.ai/v1")
    with patch.object(client.chat.completions.with_raw_response, "create", new=MagicMock()) as mocked_create:
        local_model_cost_map.completion(
            model="together_ai/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "hi"}],
            client=client,
            reasoning_effort="low",
            max_completion_tokens=100,
            logprobs=True,
            top_logprobs=3,
            top_k=40,
            repetition_penalty=1.1,
            safety_model="meta-llama/Llama-Guard-4-12B",
        )

    sent = mocked_create.call_args.kwargs
    assert sent["model"] == "openai/gpt-oss-120b"
    # every key here has to be one the SDK's create() accepts, or the call raises TypeError
    # before it is ever sent; Together-native fields belong in extra_body
    assert "reasoning" not in sent
    assert sent["max_tokens"] == 100
    assert sent["logprobs"] == 3
    assert sent["reasoning_effort"] == "low"
    assert "max_completion_tokens" not in sent
    assert "top_logprobs" not in sent
    # Together-native params ride in extra_body, which the OpenAI SDK spreads into the body.
    assert sent["extra_body"] == {
        "top_k": 40,
        "repetition_penalty": 1.1,
        "safety_model": "meta-llama/Llama-Guard-4-12B",
    }


def test_reasoning_toggle_reaches_the_sdk_in_extra_body(local_model_cost_map, no_together_env):
    """Regression: `reasoning` handed to the SDK as a keyword raised
    `AsyncCompletions.create() got an unexpected keyword argument 'reasoning'`."""
    import openai

    client = openai.OpenAI(api_key="test-key", base_url="https://api.together.ai/v1")
    with patch.object(client.chat.completions.with_raw_response, "create", new=MagicMock()) as mocked_create:
        local_model_cost_map.completion(
            model="together_ai/moonshotai/Kimi-K3",
            messages=[{"role": "user", "content": "hi"}],
            client=client,
            reasoning_effort="none",
        )

    sent = mocked_create.call_args.kwargs
    assert sent["extra_body"]["reasoning"] == {"enabled": False}
    assert "reasoning" not in sent
    assert "reasoning_effort" not in sent


def test_api_key_resolution_order(no_together_env):
    import litellm

    assert TogetherAIConfig.get_api_key("explicit-key") == "explicit-key"

    no_together_env.setenv("TOGETHER_AI_TOKEN", "legacy-token")
    assert TogetherAIConfig.get_api_key() == "legacy-token"

    no_together_env.setenv("TOGETHER_API_KEY", "primary-key")
    assert TogetherAIConfig.get_api_key() == "primary-key"

    no_together_env.delenv("TOGETHER_API_KEY")
    no_together_env.delenv("TOGETHER_AI_TOKEN")
    no_together_env.setattr(litellm, "togetherai_api_key", "sdk-key", raising=False)
    assert TogetherAIConfig.get_api_key() == "sdk-key"


def test_api_base_resolution(no_together_env):
    """Together documents https://api.together.ai/v1; api.together.xyz is the legacy host."""
    assert TogetherAIConfig.get_api_base() == "https://api.together.ai/v1"

    no_together_env.setenv("TOGETHER_AI_API_BASE", "https://gateway.internal/v1")
    assert TogetherAIConfig.get_api_base() == "https://gateway.internal/v1"
    assert TogetherAIConfig.get_api_base("https://explicit/v1") == "https://explicit/v1"


def test_get_llm_provider_uses_the_config(no_together_env):
    from litellm import get_llm_provider

    no_together_env.setenv("TOGETHER_API_KEY", "primary-key")
    model, provider, api_key, api_base = get_llm_provider(model="together_ai/moonshotai/Kimi-K3")

    assert (model, provider) == ("moonshotai/Kimi-K3", "together_ai")
    assert api_key == "primary-key"
    assert api_base == "https://api.together.ai/v1"


@pytest.mark.parametrize(
    "payload",
    [
        {"object": "list", "data": [{"id": "zai-org/GLM-5.2"}, {"id": "openai/whisper-large-v3"}]},
        [{"id": "zai-org/GLM-5.2"}, {"id": "openai/whisper-large-v3"}],
    ],
    ids=["openai-shaped", "bare-array"],
)
def test_get_models_prefixes_ids_for_both_payload_shapes(no_together_env, payload):
    """Together's /models returns a bare array; the OpenAI SDK shape wraps it in `data`."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload

    with patch("litellm.module_level_client.get", return_value=response) as mocked_get:
        models = TogetherAIConfig().get_models(api_key="test-key")

    assert models == ["together_ai/zai-org/GLM-5.2", "together_ai/openai/whisper-large-v3"]
    assert mocked_get.call_args.kwargs["url"] == "https://api.together.ai/v1/models"
    assert mocked_get.call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_get_models_honors_a_custom_api_base(no_together_env):
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"data": []}

    with patch("litellm.module_level_client.get", return_value=response) as mocked_get:
        TogetherAIConfig().get_models(api_key="test-key", api_base="https://gateway.internal/v1/")

    assert mocked_get.call_args.kwargs["url"] == "https://gateway.internal/v1/models"


def test_get_models_without_a_key_raises(no_together_env):
    with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
        TogetherAIConfig().get_models()


def test_get_models_surfaces_an_http_error(no_together_env):
    response = MagicMock()
    response.status_code = 401
    response.text = "invalid api key"

    with patch("litellm.module_level_client.get", return_value=response):
        with pytest.raises(ValueError, match="401"):
            TogetherAIConfig().get_models(api_key="bad-key")
