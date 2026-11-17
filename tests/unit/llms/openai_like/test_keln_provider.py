import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.llms.openai import AllMessageValues

_DEFAULT_API_BASE: Final = "https://api.keln.ai/v1"
_CHAT_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_RESPONSES_URL: Final = f"{_DEFAULT_API_BASE}/responses"
_MODEL: Final = "zai-org/glm-5.3"
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_keln",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": _MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Keln", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_RESPONSE: Final = {
    "id": "resp_keln",
    "object": "response",
    "created_at": 1_790_000_000,
    "model": _MODEL,
    "status": "completed",
    "output": [
        {
            "id": "msg_keln",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello from Keln", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
}
KELN_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("keln/")))


def _sent_chat_body(messages: list[AllMessageValues], **params: object) -> dict[str, object]:
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(model=f"keln/{_MODEL}", messages=messages, api_key="keln-test-key", **params)
    assert route.call_count == 1
    return json.loads(route.calls.last.request.content)


def test_keln_is_a_registered_provider():
    assert litellm.LlmProviders.KELN.value == "keln"
    assert "keln" in litellm.provider_list
    assert "keln" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_keln_provider_resolution_keeps_the_org_in_the_model_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KELN_API_KEY", "keln-test-key")
    monkeypatch.delenv("KELN_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=f"keln/{_MODEL}",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == _MODEL
    assert provider == "keln"
    assert api_key == "keln-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_keln_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KELN_API_KEY", "keln-env-key")
    monkeypatch.setenv("KELN_API_BASE", "https://keln.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=f"keln/{_MODEL}",
        custom_llm_provider=None,
        api_base="https://keln.internal.example/v1",
        api_key="keln-explicit-key",
    )

    assert provider == "keln"
    assert api_key == "keln-explicit-key"
    assert api_base == "https://keln.internal.example/v1"


def test_keln_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KELN_API_KEY", "keln-env-key")
    monkeypatch.setenv("KELN_API_BASE", "https://keln.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model=f"keln/{_MODEL}",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "keln"
    assert api_base == "https://keln.env.example/v1"


def test_keln_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KELN_API_KEY", "keln-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == _MODEL
    assert provider == "keln"
    assert api_key == "keln-env-key"
    assert api_base == _DEFAULT_API_BASE


def test_keln_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    keln: Final = next(provider for provider in providers if provider["litellm_provider"] == "keln")

    assert keln["provider"] == "KELN"
    assert keln["provider_display_name"] == "Keln"
    assert keln["default_model_placeholder"] in KELN_MODELS
    assert {field["key"]: field["required"] for field in keln["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


@pytest.mark.parametrize("model", KELN_MODELS)
def test_keln_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import supports_function_calling, supports_reasoning, supports_vision

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="keln",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="keln",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000_000)
    assert model_info["max_input_tokens"] == entry["max_input_tokens"]
    assert model_info["litellm_provider"] == "keln"
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_vision(model) is ("image" in entry["supported_modalities"])


@pytest.mark.usefixtures("local_model_cost_map")
def test_keln_wildcard_expands_to_every_keln_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("keln/*")) == set(KELN_MODELS)


def test_keln_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling, supports_reasoning

    assert supports_function_calling("keln/a-model-not-in-the-cost-map") is True
    assert supports_reasoning("keln/a-model-not-in-the-cost-map") is True


def test_keln_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    keln_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("keln/")}

    assert keln_entries
    assert tuple(sorted(keln_entries)) == KELN_MODELS
    assert keln_entries == {name: backup[name] for name in keln_entries}


def test_keln_supported_endpoints_follow_its_chat_rows():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["keln"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["keln"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in ("chat_completions", "messages", "responses")} == {
        "chat_completions": True,
        "messages": True,
        "responses": True,
    }


def test_keln_chat_completion_is_priced_from_its_cost_map_row():
    with respx.mock() as upstream:
        upstream.post(_CHAT_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=f"keln/{_MODEL}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="keln-test-key",
        )

    entry: Final = litellm.model_cost[f"keln/{_MODEL}"]
    expected: Final = 4 * entry["input_cost_per_token"] + 3 * entry["output_cost_per_token"]
    assert response._hidden_params["response_cost"] == pytest.approx(expected)


def test_keln_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=f"keln/{_MODEL}",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="keln-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_URL
    assert request.headers["authorization"] == "Bearer keln-test-key"
    assert body["model"] == _MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Keln"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_keln_receives_max_completion_tokens_as_max_tokens():
    body: Final = _sent_chat_body([{"role": "user", "content": "Say hello"}], max_completion_tokens=64)

    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body


def test_keln_receives_developer_instructions_as_a_system_message():
    body: Final = _sent_chat_body(
        [{"role": "developer", "content": "Answer briefly"}, {"role": "user", "content": "Say hello"}]
    )

    assert body["messages"] == [
        {"role": "system", "content": "Answer briefly"},
        {"role": "user", "content": "Say hello"},
    ]


@pytest.mark.parametrize("reasoning_effort", ["low", "max"])
def test_keln_receives_reasoning_effort_for_a_model_without_a_cost_map_row(reasoning_effort: str):
    body: Final = _sent_chat_body([{"role": "user", "content": "Say hello"}], reasoning_effort=reasoning_effort)

    assert body["reasoning_effort"] == reasoning_effort


def test_keln_receives_tools_for_a_model_without_a_cost_map_row():
    body: Final = _sent_chat_body(
        [{"role": "user", "content": "Weather in Paris?"}], tools=[_WEATHER_TOOL], tool_choice="auto"
    )

    assert body["tools"] == [_WEATHER_TOOL]
    assert body["tool_choice"] == "auto"


def test_keln_streamed_tool_call_is_reassembled():
    chunks: Final = (
        'data: {"id":"chatcmpl_keln","object":"chat.completion.chunk","created":1790000000,"model":"zai-org/glm-5.3",'
        '"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function",'
        '"function":{"name":"get_weather","arguments":"{\\"city\\":"}}]},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl_keln","object":"chat.completion.chunk","created":1790000000,"model":"zai-org/glm-5.3",'
        '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Paris\\"}"}}]},'
        '"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl_keln","object":"chat.completion.chunk","created":1790000000,"model":"zai-org/glm-5.3",'
        '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        "data: [DONE]\n\n",
    )
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(
            200, headers={"content-type": "text/event-stream"}, content="".join(chunks).encode()
        )
        stream: Final = litellm.completion(
            model=f"keln/{_MODEL}",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            stream=True,
            api_key="keln-test-key",
        )
        collected: Final = tuple(stream)

    rebuilt: Final = litellm.stream_chunk_builder(chunks=list(collected))
    body: Final = json.loads(route.calls.last.request.content)
    assert body["stream"] is True
    assert body["tools"] == [_WEATHER_TOOL]
    assert rebuilt is not None
    tool_call: Final = rebuilt.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {"city": "Paris"}
    assert rebuilt.choices[0].finish_reason == "tool_calls"


def test_keln_responses_request_goes_to_keln_natively():
    with respx.mock() as upstream:
        route: Final = upstream.post(_RESPONSES_URL).respond(200, json=_RESPONSE)
        response: Final = litellm.responses(model=f"keln/{_MODEL}", input="Say hello", api_key="keln-test-key")

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer keln-test-key"
    assert body["model"] == _MODEL
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Keln"


@pytest.mark.asyncio
async def test_keln_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"keln/{_MODEL}",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="keln-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer keln-test-key"
    assert body["model"] == _MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from Keln" for block in response["content"])
