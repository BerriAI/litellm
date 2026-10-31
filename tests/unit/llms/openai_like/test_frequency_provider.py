import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

FREQUENCY_API_BASE: Final = "https://llm.scalefrequency.com/v2"
FREQUENCY_MODEL_ID: Final = "google/gemma-4-26b-a4b-it"
FREQUENCY_MODEL: Final = f"frequency/{FREQUENCY_MODEL_ID}"
_REPO_ROOT: Final = Path(__file__).resolve().parents[4]
_COST_MAP_PATHS: Final = (
    _REPO_ROOT / "model_prices_and_context_window.json",
    _REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json",
)
_SMALL_USAGE: Final = {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_CITY_SCHEMA: Final = {
    "type": "json_schema",
    "json_schema": {
        "name": "city",
        "schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        "strict": True,
    },
}


def _chat_completion_body(usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl-frequency",
        "object": "chat.completion",
        "created": 1_792_000_000,
        "model": FREQUENCY_MODEL_ID,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": usage,
    }


def test_frequency_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREQUENCY_API_KEY", "fake-environment-key")

    assert get_llm_provider(model=FREQUENCY_MODEL) == (
        FREQUENCY_MODEL_ID,
        "frequency",
        "fake-environment-key",
        FREQUENCY_API_BASE,
    )
    assert get_llm_provider(
        model=FREQUENCY_MODEL, api_key="fake-explicit-key", api_base="https://gateway.example/v2"
    ) == (FREQUENCY_MODEL_ID, "frequency", "fake-explicit-key", "https://gateway.example/v2")


def test_frequency_is_detected_from_its_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREQUENCY_API_KEY", "fake-environment-key")

    assert get_llm_provider(model=FREQUENCY_MODEL_ID, api_base=FREQUENCY_API_BASE) == (
        FREQUENCY_MODEL_ID,
        "frequency",
        "fake-environment-key",
        FREQUENCY_API_BASE,
    )


def test_frequency_chat_reads_key_and_base_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FREQUENCY_API_KEY", "fake-environment-key")
    monkeypatch.setenv("FREQUENCY_API_BASE", "https://frequency.internal.example/v2")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://frequency.internal.example/v2/chat/completions").respond(
            200, json=_chat_completion_body(_SMALL_USAGE)
        )
        litellm.completion(model=FREQUENCY_MODEL, messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer fake-environment-key"


@pytest.mark.parametrize(
    "params",
    (
        {"tools": [_WEATHER_TOOL], "tool_choice": "required"},
        {"response_format": _CITY_SCHEMA},
        {"top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1, "seed": 7, "stop": ["\n\n"]},
    ),
    ids=("tools", "json_schema", "sampling"),
)
def test_frequency_chat_sends_the_bare_model_id_and_params_unchanged(params: dict[str, object]) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{FREQUENCY_API_BASE}/chat/completions").respond(
            200, json=_chat_completion_body(_SMALL_USAGE)
        )
        response: Final = litellm.completion(
            model=FREQUENCY_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="fake-test-key",
            **params,
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == FREQUENCY_MODEL_ID
    assert {key: body.get(key) for key in params} == params
    assert response.choices[0].message.content == "hello"


def test_frequency_rejected_key_raises_authentication_error() -> None:
    with respx.mock() as upstream:
        upstream.post(f"{FREQUENCY_API_BASE}/chat/completions").respond(
            401, json={"error": {"code": "unauthorized", "message": "Unknown or invalid key."}}
        )
        with pytest.raises(litellm.AuthenticationError):
            litellm.completion(
                model=FREQUENCY_MODEL,
                messages=[{"role": "user", "content": "Say hello"}],
                api_key="fake-rejected-key",
            )


def test_frequency_chat_cost_bills_cached_input_at_the_cache_read_rate(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        upstream.post(f"{FREQUENCY_API_BASE}/chat/completions").respond(
            200,
            json=_chat_completion_body(
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 200,
                    "total_tokens": 1200,
                    "prompt_tokens_details": {"cached_tokens": 800},
                }
            ),
        )
        response: Final = litellm.completion(
            model=FREQUENCY_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="fake-test-key",
        )

    rates: Final = litellm.model_cost[FREQUENCY_MODEL]
    expected: Final = (
        200 * rates["input_cost_per_token"]
        + 800 * rates["cache_read_input_token_cost"]
        + 200 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_frequency_catalog_row_matches_its_backup_and_fits_its_context() -> None:
    primary, backup = (json.loads(path.read_text())[FREQUENCY_MODEL] for path in _COST_MAP_PATHS)

    assert primary == backup
    assert primary["litellm_provider"] == "frequency"
    assert 0 < primary["max_output_tokens"] <= primary["max_input_tokens"]
    assert 0 < primary["cache_read_input_token_cost"] < primary["input_cost_per_token"]
