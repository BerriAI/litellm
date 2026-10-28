import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

MODEL: Final = "nexus/anthropic/claude-sonnet-5.5"


def test_nexus_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXUS_API_KEY", "fake-environment-key")
    monkeypatch.delenv("NEXUS_API_BASE", raising=False)
    assert get_llm_provider(model=MODEL) == (
        "anthropic/claude-sonnet-5.5",
        "nexus",
        "fake-environment-key",
        "https://api.nexus.inc/v1",
    )
    monkeypatch.setenv("NEXUS_API_BASE", "https://gateway.example/v1")
    assert get_llm_provider(model=MODEL)[3] == "https://gateway.example/v1"


def test_nexus_chat_keeps_the_lab_prefix_and_sends_max_tokens(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.nexus.inc/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "anthropic/claude-sonnet-5.5",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
            },
        )
        response: Final = litellm.completion(
            model=MODEL,
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Reply with pong"}],
            max_completion_tokens=50,
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "anthropic/claude-sonnet-5.5"
    assert body["max_tokens"] == 50
    assert "max_completion_tokens" not in body
    assert response.choices[0].message.content == "pong"
    rates: Final = litellm.model_cost[MODEL]
    expected: Final = 100 * rates["input_cost_per_token"] + 5 * rates["output_cost_per_token"]
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_nexus_rows_describe_chat_models_with_tools(local_model_cost_map: None) -> None:
    rows: Final = {key: row for key, row in litellm.model_cost.items() if row.get("litellm_provider") == "nexus"}
    assert MODEL in rows
    for key, row in rows.items():
        assert key.startswith("nexus/") and key.count("/") == 2, key
        assert row["mode"] == "chat"
        assert row["supports_function_calling"] and row["supports_tool_choice"]
        assert row["input_cost_per_token"] > 0 and row["output_cost_per_token"] > 0
    assert litellm.supports_function_calling(model=MODEL)
    assert litellm.supports_vision(model=MODEL)
