import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.blockrun.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_UNLISTED_MODEL: Final = "blockrun/openai/gpt-6-luna"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_blockrun",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "nvidia/nemotron-3.5-lightning",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello from BlockRun"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_TOOL_CALL_COMPLETION: Final = {
    **_CHAT_COMPLETION,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
BLOCKRUN_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("blockrun/")))


def test_blockrun_provider_resolution_keeps_the_nested_model_slug(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLOCKRUN_API_KEY", "brk-env")
    monkeypatch.delenv("BLOCKRUN_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="blockrun/nvidia/nemotron-3.5-lightning",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "nvidia/nemotron-3.5-lightning"
    assert provider == "blockrun"
    assert api_key == "brk-env"
    assert api_base == _DEFAULT_API_BASE


def test_blockrun_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BLOCKRUN_API_KEY", "brk-env")
    monkeypatch.setenv("BLOCKRUN_API_BASE", "https://blockrun.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model=_UNLISTED_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "blockrun"
    assert api_base == "https://blockrun.env.example/v1"


def test_blockrun_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="blockrun/nvidia/nemotron-3.5-lightning",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=64,
            api_key="brk-test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer brk-test"
    assert body["model"] == "nvidia/nemotron-3.5-lightning"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 64
    assert response.choices[0].message.content == "Hello from BlockRun"


def test_blockrun_sends_tools_for_a_model_without_a_cost_map_row():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_TOOL_CALL_COMPLETION)
        response: Final = litellm.completion(
            model=_UNLISTED_MODEL,
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="required",
            api_key="brk-test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["model"] == "openai/gpt-6-luna"
    assert body["tools"] == [_WEATHER_TOOL]
    assert body["tool_choice"] == "required"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_blockrun_chat_never_borrows_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BLOCKRUN_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="BLOCKRUN_API_KEY"):
            litellm.completion(model=_UNLISTED_MODEL, messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 0


def test_blockrun_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="blockrun/nvidia/nemotron-3.5-lightning",
            input="Say hello",
            api_key="brk-test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "nvidia/nemotron-3.5-lightning"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from BlockRun"


@pytest.mark.usefixtures("local_model_cost_map")
def test_blockrun_wildcard_expands_to_every_blockrun_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert BLOCKRUN_MODELS
    assert set(get_known_models_from_wildcard("blockrun/*")) == set(BLOCKRUN_MODELS)


def test_blockrun_backup_maps_mirror_the_root_maps():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    cost_backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    endpoints: Final = json.loads((package_root.parent / "provider_endpoints_support.json").read_text())
    endpoints_backup: Final = json.loads((package_root / "provider_endpoints_support_backup.json").read_text())
    blockrun_rows: Final = {name: entry for name, entry in cost_map.items() if name.startswith("blockrun/")}

    assert tuple(sorted(blockrun_rows)) == BLOCKRUN_MODELS
    assert blockrun_rows == {name: cost_backup[name] for name in blockrun_rows}
    assert endpoints["providers"]["blockrun"] == endpoints_backup["providers"]["blockrun"]


def test_blockrun_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    blockrun: Final = next(provider for provider in providers if provider["litellm_provider"] == "blockrun")

    assert blockrun["provider"] == "BLOCKRUN"
    assert blockrun["default_model_placeholder"] in BLOCKRUN_MODELS
    assert {field["key"]: field["required"] for field in blockrun["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
