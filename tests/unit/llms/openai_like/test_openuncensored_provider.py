import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import trim_messages

_API_BASE: Final = "https://openuncensored.com/api/v1"
_MODEL: Final = "openuncensored/glm-5.3-flash-uncensored"


def test_openuncensored_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENUNCENSORED_API_KEY", "fake-environment-key")
    monkeypatch.delenv("OPENUNCENSORED_API_BASE", raising=False)
    assert get_llm_provider(model=_MODEL) == (
        "glm-5.3-flash-uncensored",
        "openuncensored",
        "fake-environment-key",
        _API_BASE,
    )
    assert get_llm_provider(model=_MODEL, api_key="fake-explicit-key", api_base="https://gateway.example/v1") == (
        "glm-5.3-flash-uncensored",
        "openuncensored",
        "fake-explicit-key",
        "https://gateway.example/v1",
    )


def test_openuncensored_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENUNCENSORED_API_KEY", "fake-environment-key")
    monkeypatch.setenv("OPENUNCENSORED_API_BASE", "https://ou.env.example/v1")
    assert get_llm_provider(model=_MODEL)[3] == "https://ou.env.example/v1"


def test_openuncensored_chat_forwards_tools_and_reasoning(local_model_cost_map: None) -> None:
    tool: Final = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_API_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "glm-5.3-flash-uncensored",
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
                                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
            },
        )
        response: Final = litellm.completion(
            model=_MODEL,
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            reasoning_effort="low",
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "glm-5.3-flash-uncensored"
    assert body["reasoning_effort"] == "low"
    assert body["tools"] == [tool]
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
    rates: Final = litellm.model_cost[_MODEL]
    expected: Final = 100 * rates["input_cost_per_token"] + 5 * rates["output_cost_per_token"]
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_openuncensored_uses_native_responses_endpoint() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp-test",
                "object": "response",
                "created_at": 1,
                "model": "glm-5.3-flash-uncensored",
                "status": "completed",
                "output": [
                    {
                        "id": "msg-test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hello", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(model=_MODEL, api_key="fake-test-key", input="Say hello", store=False)
    assert route.calls.last.request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(route.calls.last.request.content)["model"] == "glm-5.3-flash-uncensored"
    assert response.output[0].content[0].text == "hello"


def test_openuncensored_backup_cost_map_and_endpoints_mirror_the_root_files() -> None:
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    rows: Final = {name: row for name, row in cost_map.items() if name.startswith("openuncensored/")}
    assert rows
    assert rows == {name: backup[name] for name in rows}
    endpoints: Final = json.loads((package_root.parent / "provider_endpoints_support.json").read_text())
    endpoints_backup: Final = json.loads((package_root / "provider_endpoints_support_backup.json").read_text())
    assert endpoints["providers"]["openuncensored"] == endpoints_backup["providers"]["openuncensored"]


def test_openuncensored_models_trim_to_their_input_budget(local_model_cost_map: None) -> None:
    model_names: Final = tuple(model for model in litellm.model_cost if model.startswith("openuncensored/"))
    assert model_names
    messages: Final = [{"role": "user", "content": "hello " * 500}]
    budget: Final = 100
    for model in model_names:
        trimmed = trim_messages(
            messages,
            model=model,
            trim_ratio=budget / litellm.model_cost[model]["max_input_tokens"],
        )
        assert litellm.token_counter(model=model, messages=messages) > budget
        assert litellm.token_counter(model=model, messages=trimmed) <= budget


def test_openuncensored_model_missing_from_cost_map_uses_provider_defaults() -> None:
    from litellm.utils import supports_function_calling, supports_reasoning

    assert supports_function_calling("openuncensored/a-model-not-in-the-cost-map") is True
    assert supports_reasoning("openuncensored/a-model-not-in-the-cost-map") is True


def test_openuncensored_never_sends_the_openai_key(monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None) -> None:
    """Without an Open Uncensored key, the call fails instead of falling back to OPENAI_API_KEY."""
    monkeypatch.delenv("OPENUNCENSORED_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key-must-not-leak")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-must-not-leak")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(f"{_API_BASE}/chat/completions").respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match="OPENUNCENSORED_API_KEY"):
            litellm.completion(model=_MODEL, messages=[{"role": "user", "content": "hi"}])
    assert not route.called
