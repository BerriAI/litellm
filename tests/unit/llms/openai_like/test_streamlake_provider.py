import json
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.proxy.auth.model_checks import get_known_models_from_wildcard
from litellm.types.utils import ModelResponse, Usage

STREAMLAKE_BASE_URL: Final = "https://vanchin.streamlake.ai/api/gateway/v1/endpoints"
STREAMLAKE_CHAT_URL: Final = f"{STREAMLAKE_BASE_URL}/chat/completions"

REPO_ROOT: Final = Path(__file__).parents[4]


class TestStreamlakeProviderConfig:
    def test_streamlake_provider_resolution(self) -> None:
        model, provider, api_key, api_base = get_llm_provider(
            model="streamlake/ep-test-endpoint",
            custom_llm_provider=None,
            api_base=None,
            api_key="sk-test",
        )

        assert model == "ep-test-endpoint"
        assert provider == "streamlake"
        assert api_base == STREAMLAKE_BASE_URL
        assert api_key == "sk-test"

    def test_streamlake_api_base_override(self) -> None:
        model, provider, api_key, api_base = get_llm_provider(
            model="streamlake/ep-test-endpoint",
            custom_llm_provider=None,
            api_base="https://custom.streamlake.example/v1",
            api_key="sk-test",
        )

        assert model == "ep-test-endpoint"
        assert provider == "streamlake"
        assert api_base == "https://custom.streamlake.example/v1"
        assert api_key == "sk-test"

    def test_streamlake_url_autodetection(self) -> None:
        _, provider, _, api_base = get_llm_provider(
            model="some-public-model",
            custom_llm_provider=None,
            api_base=STREAMLAKE_BASE_URL,
            api_key=None,
        )
        assert provider == "streamlake"
        assert api_base == STREAMLAKE_BASE_URL

    def test_streamlake_api_key_resolved_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STREAMLAKE_API_KEY", "sk-env-key")

        _, provider, api_key, _ = get_llm_provider(
            model="streamlake/ep-test-endpoint",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )
        assert provider == "streamlake"
        assert api_key == "sk-env-key"

    def test_streamlake_validate_environment_sets_bearer(self) -> None:
        provider: Final = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config: Final = create_config_class(provider)()

        headers: Final = config.validate_environment(
            headers={},
            model="ep-test-endpoint",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="sk-resolved-key",
            api_base=None,
        )
        assert headers["Authorization"] == "Bearer sk-resolved-key"

    def test_streamlake_complete_url_appends_chat_completions(self) -> None:
        provider: Final = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config: Final = create_config_class(provider)()

        url: Final = config.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="ep-test-endpoint",
            optional_params={},
            litellm_params={},
        )
        assert url == STREAMLAKE_CHAT_URL

    @pytest.mark.parametrize("cap_param", ("max_tokens", "max_completion_tokens"))
    def test_streamlake_token_cap_sent_as_max_completion_tokens(self, cap_param: str) -> None:
        provider: Final = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config: Final = create_config_class(provider)()

        optional_params: Final = config.map_openai_params(
            non_default_params={cap_param: 1024},
            optional_params={},
            model="ep-test-endpoint",
            drop_params=False,
        )
        assert optional_params == {"max_completion_tokens": 1024}

    def test_streamlake_chat_params_pass_through(self) -> None:
        provider: Final = JSONProviderRegistry.get("streamlake")
        assert provider is not None
        config: Final = create_config_class(provider)()

        optional_params: Final = config.map_openai_params(
            non_default_params={"temperature": 0.2, "top_p": 0.9, "stream": True},
            optional_params={},
            model="ep-test-endpoint",
            drop_params=False,
        )
        assert optional_params == {"temperature": 0.2, "top_p": 0.9, "stream": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_async, api_base",
    (
        (False, None),
        (False, "https://custom.streamlake.example/v1"),
        (False, "https://custom.streamlake.example/v1/"),
        (True, None),
        (True, "https://custom.streamlake.example/v1"),
        (True, "https://custom.streamlake.example/v1/"),
    ),
    ids=(
        "sync-default",
        "sync-override",
        "sync-override-trailing-slash",
        "async-default",
        "async-override",
        "async-override-trailing-slash",
    ),
)
async def test_streamlake_completion_uses_openai_compatible_http(
    is_async: bool, api_base: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    response_body: Final[dict[str, object]] = {
        "id": "chatcmpl-streamlake-test",
        "object": "chat.completion",
        "created": 1710000000,
        "model": "ep-test-endpoint",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "StreamLake response"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }
    expected_url: Final[str] = f"{(api_base or STREAMLAKE_BASE_URL).rstrip('/')}/chat/completions"
    expected_request_body: Final[dict[str, object]] = {
        "messages": [{"role": "user", "content": "hello"}],
        "model": "ep-test-endpoint",
        "max_completion_tokens": 512,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == expected_url
        assert request.headers["authorization"] == "Bearer sk-streamlake-test"
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == expected_request_body
        return httpx.Response(200, json=response_body, request=request)

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(expected_url).mock(side_effect=handler)
        if is_async:
            response: Final = await litellm.acompletion(
                model="streamlake/ep-test-endpoint",
                messages=[{"role": "user", "content": "hello"}],
                api_key="sk-streamlake-test",
                api_base=api_base,
                max_tokens=512,
            )
        else:
            response: Final = litellm.completion(
                model="streamlake/ep-test-endpoint",
                messages=[{"role": "user", "content": "hello"}],
                api_key="sk-streamlake-test",
                api_base=api_base,
                max_tokens=512,
            )

    assert route.call_count == 1
    assert len(response.choices) == 1
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "StreamLake response"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 2
    assert response.usage.total_tokens == 5


@pytest.mark.usefixtures("local_model_cost_map")
def test_endpoint_id_call_forwards_tools_and_reasoning_effort_to_streamlake(monkeypatch: pytest.MonkeyPatch) -> None:
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    response_body: Final[dict[str, object]] = {
        "id": "chatcmpl-streamlake-tools",
        "object": "chat.completion",
        "created": 1710000000,
        "model": "glm-5.3",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(STREAMLAKE_CHAT_URL).mock(return_value=httpx.Response(200, json=response_body))
        litellm.completion(
            model="streamlake/ep-test-endpoint",
            messages=[{"role": "user", "content": "weather in Paris?"}],
            api_key="sk-streamlake-test",
            tools=tools,
            tool_choice="auto",
            reasoning_effort="low",
        )

    sent: Final = json.loads(route.calls.last.request.content)
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"
    assert sent["reasoning_effort"] == "low"


def _streamlake_keys(cost_map: dict[str, object]) -> tuple[str, ...]:
    return tuple(key for key in cost_map if key.startswith("streamlake/"))


def test_root_and_backup_cost_maps_carry_the_same_streamlake_rows() -> None:
    root: Final = json.loads((REPO_ROOT / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json").read_text())

    assert _streamlake_keys(root)
    assert {key: root[key] for key in _streamlake_keys(root)} == {key: backup[key] for key in _streamlake_keys(backup)}


@pytest.mark.usefixtures("local_model_cost_map")
def test_endpoint_id_call_is_priced_from_the_lowercase_model_streamlake_returns() -> None:
    keys: Final = _streamlake_keys(litellm.model_cost)
    assert keys

    for key in keys:
        info: Final = litellm.get_model_info(model=key)
        response: Final = ModelResponse(
            model=key.split("/", 1)[1].lower(),
            usage=Usage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
        )
        cost: Final = litellm.completion_cost(
            completion_response=response,
            model="streamlake/ep-test-endpoint",
            custom_llm_provider="streamlake",
        )
        expected: Final = 1000 * info["input_cost_per_token"] + 500 * info["output_cost_per_token"]
        assert cost == pytest.approx(expected), key


@pytest.mark.usefixtures("local_model_cost_map")
def test_named_model_params_follow_the_row_and_fall_back_to_the_provider_default() -> None:
    provider: Final = JSONProviderRegistry.get("streamlake")
    assert provider is not None
    config: Final = create_config_class(provider)()
    keys: Final = _streamlake_keys(litellm.model_cost)
    assert keys

    for key in keys:
        row: Final = litellm.model_cost[key]
        supported: Final = config.get_supported_openai_params(model=key.split("/", 1)[1].lower())
        assert ("tools" in supported) is (row.get("supports_function_calling") is not False), key
        assert ("reasoning_effort" in supported) is (row.get("supports_reasoning") is not False), key


@pytest.mark.usefixtures("local_model_cost_map")
def test_streamlake_wildcard_expands_to_every_streamlake_row() -> None:
    expanded: Final = get_known_models_from_wildcard("streamlake/*")
    assert set(expanded) == set(_streamlake_keys(litellm.model_cost))


@pytest.mark.usefixtures("local_model_cost_map")
def test_cache_control_marks_reach_streamlake(monkeypatch: pytest.MonkeyPatch) -> None:
    system_block: Final = {"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}
    tool: Final = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}},
        "cache_control": {"type": "ephemeral"},
    }
    response_body: Final[dict[str, object]] = {
        "id": "chatcmpl-streamlake-cache",
        "object": "chat.completion",
        "created": 1710000000,
        "model": "qwen3.8-27b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(STREAMLAKE_CHAT_URL).mock(return_value=httpx.Response(200, json=response_body))
        litellm.completion(
            model="streamlake/qwen3.8-27b",
            messages=[{"role": "system", "content": [system_block]}, {"role": "user", "content": "hi"}],
            api_key="sk-streamlake-test",
            tools=[tool],
        )

    sent: Final = json.loads(route.calls.last.request.content)
    assert sent["messages"][0]["content"] == [system_block]
    assert sent["tools"][0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", ("kimi-k2.6", "qwen3.6-27b", "qwen3.6-35b-a3b"))
def test_reasoning_effort_is_dropped_for_models_streamlake_lists_without_reasoning(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # StreamLake's Models Overview (https://www.streamlake.ai/document/DOC/mh1gbfvrdn6hpbzxixv, read 2026-10-28)
    # gives these models no Reasoning tag, and kimi-k2.6 answers 400 to reasoning_effort under an 8193 token cap
    response_body: Final[dict[str, object]] = {
        "id": "chatcmpl-streamlake-reasoning",
        "object": "chat.completion",
        "created": 1710000000,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    with respx.mock(assert_all_called=True) as respx_mock:
        route: Final = respx_mock.post(STREAMLAKE_CHAT_URL).mock(return_value=httpx.Response(200, json=response_body))
        litellm.completion(
            model=f"streamlake/{model}",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-streamlake-test",
            reasoning_effort="low",
            drop_params=True,
        )

    assert "reasoning_effort" not in json.loads(route.calls.last.request.content)
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.completion(
            model=f"streamlake/{model}",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-streamlake-test",
            reasoning_effort="low",
        )


@pytest.mark.usefixtures("local_model_cost_map")
def test_streamlake_rows_with_a_cache_read_price_declare_prompt_caching() -> None:
    cached_rows: Final = tuple(
        key for key in _streamlake_keys(litellm.model_cost) if "cache_read_input_token_cost" in litellm.model_cost[key]
    )
    assert cached_rows
    assert [key for key in cached_rows if litellm.model_cost[key].get("supports_prompt_caching") is not True] == []


@pytest.mark.usefixtures("local_model_cost_map")
@pytest.mark.parametrize("model", ("qwen3.8-27b", "qwen3.8-2.4t-a95b"))
@pytest.mark.parametrize("cache_type", ("ephemeral", None))
def test_qwen3_8_explicit_cache_hits_and_prefix_hits_use_their_own_read_price(
    model: str, cache_type: str | None
) -> None:
    # StreamLake's price page (https://www.streamlake.ai/document/DOC/mgrnm4xm362hvp5wyce, read 2026-10-28) lists
    # separate PrefixCache and Cache Read prices for Qwen3.8, and its usage tags only explicit hits with cache_type
    rate_key: Final = "explicit_cache_read_input_token_cost" if cache_type else "cache_read_input_token_cost"
    row: Final = litellm.model_cost[f"streamlake/{model}"]
    details: Final = {"cached_tokens": 4000} | ({"cache_type": cache_type} if cache_type else {})
    response: Final = ModelResponse(
        model=model,
        usage=Usage(prompt_tokens=5000, completion_tokens=100, total_tokens=5100, prompt_tokens_details=details),
    )

    cost: Final = litellm.completion_cost(
        completion_response=response, model=f"streamlake/{model}", custom_llm_provider="streamlake"
    )

    expected: Final = 4000 * row[rate_key] + 1000 * row["input_cost_per_token"] + 100 * row["output_cost_per_token"]
    assert cost == pytest.approx(expected)
