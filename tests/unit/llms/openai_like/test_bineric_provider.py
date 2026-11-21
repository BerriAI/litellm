import json
from collections.abc import Mapping
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.bineric.com/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_bineric",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "claude-sonnet-5.5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Bineric"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_MODELS_THAT_TAKE_REASONING_EFFORT: Final = (
    "bineric/claude-haiku-5.5",
    "bineric/claude-opus-4.8",
    "bineric/claude-opus-5.5",
    "bineric/claude-sonnet-5.5",
    "bineric/gpt-oss-120b",
    "bineric/gpt-oss-20b",
)
BINERIC_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("bineric/")))
BINERIC_CHAT_MODELS: Final = tuple(name for name in BINERIC_MODELS if litellm.model_cost[name]["mode"] == "chat")


def test_bineric_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BINERIC_API_KEY", "bineric-test-key")
    monkeypatch.delenv("BINERIC_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="bineric/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-5.5"
    assert provider == "bineric"
    assert api_key == "bineric-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_bineric_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BINERIC_API_KEY", "bineric-env-key")
    monkeypatch.setenv("BINERIC_API_BASE", "https://bineric.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="bineric/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base="https://bineric.internal.example/v1",
        api_key="bineric-explicit-key",
    )

    assert provider == "bineric"
    assert api_key == "bineric-explicit-key"
    assert api_base == "https://bineric.internal.example/v1"


def test_bineric_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BINERIC_API_KEY", "bineric-env-key")
    monkeypatch.setenv("BINERIC_API_BASE", "https://bineric.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="bineric/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "bineric"
    assert api_base == "https://bineric.env.example/v1"


def test_bare_openai_model_on_bineric_api_base_keeps_openai_params():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(
            200, json={**_CHAT_COMPLETION, "model": "gpt-5-mini"}
        )
        litellm.completion(
            model="gpt-5-mini",
            api_base=_DEFAULT_API_BASE,
            messages=[{"role": "user", "content": "Say hello"}],
            verbosity="low",
            api_key="bineric-test-key",
        )

    assert json.loads(route.calls.last.request.content)["verbosity"] == "low"


@pytest.mark.parametrize("model", BINERIC_CHAT_MODELS)
def test_bineric_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import supports_vision

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="bineric",
    )
    model_info: Final = litellm.get_model_info(model)
    supported_params: Final = litellm.get_supported_openai_params(
        model=model.removeprefix("bineric/"), custom_llm_provider="bineric"
    )

    assert entry["input_cost_per_token"] > 0
    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert model_info["max_tokens"] == entry.get("max_output_tokens")
    assert model_info["max_input_tokens"] == entry.get("max_input_tokens")
    assert model_info["litellm_provider"] == "bineric"
    assert ("tools" in supported_params) is entry.get("supports_function_calling", False)
    assert supports_vision(model) is entry.get("supports_vision", False)


@pytest.mark.parametrize("model", _MODELS_THAT_TAKE_REASONING_EFFORT)
def test_bineric_forwards_reasoning_effort_to_the_models_that_take_it(model: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort="low",
            api_key="bineric-test-key",
        )

    assert json.loads(route.calls.last.request.content)["reasoning_effort"] == "low"


@pytest.mark.usefixtures("local_model_cost_map")
def test_bineric_wildcard_expands_to_every_bineric_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    expanded: Final = set(get_known_models_from_wildcard("bineric/*"))

    assert {"bineric/claude-sonnet-5.5", "bineric/zuse-27b"} <= expanded
    assert expanded == set(BINERIC_MODELS)


def test_bineric_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="bineric/claude-sonnet-5.5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="bineric-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer bineric-test-key"
    assert body["model"] == "claude-sonnet-5.5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "Hello from Bineric"


def _complete_with_an_unpriced_model(usage_extra: Mapping[str, float]) -> tuple[respx.Route, litellm.ModelResponse]:
    completion: Final = {
        **_CHAT_COMPLETION,
        "model": "unpriced-transcriber",
        "usage": {**_CHAT_COMPLETION["usage"], **usage_extra},
    }
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=completion)
        response: Final = litellm.completion(
            model="bineric/unpriced-transcriber",
            messages=[{"role": "user", "content": "Transcribe this"}],
            api_key="bineric-test-key",
        )
    return route, response


def test_bineric_per_request_model_bills_the_cost_bineric_reports():
    route, response = _complete_with_an_unpriced_model({"cost": 0.18})

    assert json.loads(route.calls.last.request.content)["model"] == "unpriced-transcriber"
    assert response._hidden_params["response_cost"] == pytest.approx(0.18)


def test_bineric_per_request_model_without_a_cost_row_or_reported_cost_still_routes():
    route, response = _complete_with_an_unpriced_model({})

    assert route.call_count == 1
    assert json.loads(route.calls.last.request.content)["model"] == "unpriced-transcriber"
    assert response._hidden_params["response_cost"] is None
