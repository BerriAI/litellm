import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_vispark",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "vision-medium",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Vispark"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}

_TEXT_COMPLETION: Final = {
    "id": "cmpl_vispark",
    "object": "text_completion",
    "created": 1_790_000_000,
    "model": "vision-small",
    "choices": [{"index": 0, "text": "Hello from Vispark", "finish_reason": "stop", "logprobs": None}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_vispark_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VISPARK_LAB_API_KEY", "vispark-env-key")
    monkeypatch.delenv("VISPARK_LAB_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="vispark/vision-medium",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "vision-medium"
    assert provider == "vispark"
    assert api_key == "vispark-env-key"
    assert api_base == "https://api.lab.vispark.in/v1"


def test_vispark_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VISPARK_LAB_API_KEY", "vispark-env-key")
    monkeypatch.setenv("VISPARK_LAB_API_BASE", "https://vispark.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="vispark/vision-medium",
        custom_llm_provider=None,
        api_base="https://vispark.internal.example/v1",
        api_key="vispark-explicit-key",
    )

    assert provider == "vispark"
    assert api_key == "vispark-explicit-key"
    assert api_base == "https://vispark.internal.example/v1"


def test_vispark_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VISPARK_LAB_API_KEY", "vispark-env-key")
    monkeypatch.setenv("VISPARK_LAB_API_BASE", "https://vispark.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="vispark/vision-medium",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "vispark"
    assert api_base == "https://vispark.env.example/v1"


def test_vispark_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("VISPARK_LAB_API_KEY", "vispark-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="vision-medium",
        custom_llm_provider=None,
        api_base="https://api.lab.vispark.in/v1",
        api_key=None,
    )

    assert model == "vision-medium"
    assert provider == "vispark"
    assert api_key == "vispark-env-key"
    assert api_base == "https://api.lab.vispark.in/v1"


def test_vispark_chat_completion_request_maps_max_completion_tokens():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.completion(
            model="vispark/vision-medium",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="vispark-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer vispark-test-key"
    assert body["model"] == "vision-medium"
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body
    assert response.choices[0].message.content == "Hello from Vispark"


def test_vispark_text_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.lab.vispark.in/v1/completions").respond(200, json=_TEXT_COMPLETION)
        response: Final = litellm.text_completion(
            model="vispark/vision-small",
            prompt="Say hello",
            api_key="vispark-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "vision-small"
    assert body["prompt"] == "Say hello"
    assert response.choices[0].text == "Hello from Vispark"


def test_vispark_cost_follows_the_cost_map():
    cost_entry: Final = litellm.model_cost["vispark/vision-large"]

    prompt_cost, completion_cost = litellm.cost_per_token(
        model="vision-large",
        custom_llm_provider="vispark",
        prompt_tokens=1_000,
        completion_tokens=500,
    )

    assert prompt_cost == pytest.approx(1_000 * cost_entry["input_cost_per_token"])
    assert completion_cost == pytest.approx(500 * cost_entry["output_cost_per_token"])


def test_vispark_models_are_listed_for_the_provider():
    from litellm.utils import get_valid_models

    expected: Final = {name for name, entry in litellm.model_cost.items() if entry["litellm_provider"] == "vispark"}

    assert expected
    assert set(litellm.models_by_provider["vispark"]) == expected
    assert expected <= set(get_valid_models(custom_llm_provider="vispark"))


def test_vispark_endpoint_matrix_matches_between_root_and_backup():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup_entry: Final = json.loads(backup_path.read_text())["providers"]["vispark"]
    root_entry: Final = json.loads(root_path.read_text())["providers"]["vispark"]

    assert backup_entry == root_entry
    assert root_entry["endpoints"]["chat_completions"] is True
