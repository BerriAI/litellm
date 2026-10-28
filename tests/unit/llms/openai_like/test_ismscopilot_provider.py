import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

ISMSCOPILOT_API_BASE: Final = "https://api.ismscopilot.com/v1"
ISMSCOPILOT_ALIASES: Final = (
    "isms-fast",
    "isms-thinking",
    "isms-fast-eu",
    "isms-thinking-eu",
    "isms-mini",
)


def _chat_completion_body(model: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-ismscopilot",
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Article 21 lists the measures."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
        "ismscopilot": {
            "frameworks": ["NIS_2_DE"],
            "knowledge_characters": 9000,
            "knowledge_tokens": 3000,
            "knowledge_tokens_method": "chars_div_3_v1",
        },
    }


def test_ismscopilot_is_a_registered_chat_only_json_provider() -> None:
    assert litellm.LlmProviders.ISMSCOPILOT.value == "ismscopilot"
    provider: Final = JSONProviderRegistry.get("ismscopilot")
    assert provider is not None
    assert provider.base_url == ISMSCOPILOT_API_BASE
    assert provider.api_key_env == "ISMS_COPILOT_API_KEY"
    assert provider.api_base_env == "ISMS_COPILOT_API_BASE"
    assert provider.supported_endpoints == ["/v1/chat/completions"]
    assert JSONProviderRegistry.supports_responses_api("ismscopilot") is False


@pytest.mark.parametrize("alias", ISMSCOPILOT_ALIASES)
def test_ismscopilot_resolves_every_alias_from_the_environment(monkeypatch: pytest.MonkeyPatch, alias: str) -> None:
    monkeypatch.setenv("ISMS_COPILOT_API_KEY", "fake-environment-key")
    monkeypatch.delenv("ISMS_COPILOT_API_BASE", raising=False)
    assert get_llm_provider(model=f"ismscopilot/{alias}") == (
        alias,
        "ismscopilot",
        "fake-environment-key",
        ISMSCOPILOT_API_BASE,
    )


def test_ismscopilot_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISMS_COPILOT_API_KEY", "fake-environment-key")
    assert get_llm_provider(
        model="ismscopilot/isms-fast",
        api_key="fake-explicit-key",
        api_base="https://gateway.example/v1",
    ) == ("isms-fast", "ismscopilot", "fake-explicit-key", "https://gateway.example/v1")


def test_ismscopilot_api_base_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISMS_COPILOT_API_KEY", "fake-environment-key")
    monkeypatch.setenv("ISMS_COPILOT_API_BASE", "https://staging.example/v1")
    _, _, _, api_base = get_llm_provider(model="ismscopilot/isms-fast")
    assert api_base == "https://staging.example/v1"


def test_ismscopilot_chat_forwards_the_framework_pin_and_prices_usage(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{ISMSCOPILOT_API_BASE}/chat/completions").respond(
            200, json=_chat_completion_body("isms-fast-eu")
        )
        response: Final = litellm.completion(
            model="ismscopilot/isms-fast-eu",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Which NIS 2 measures apply to an essential entity in Germany?"}],
            extra_body={"ismscopilot": {"frameworks": ["NIS_2_DE"]}},
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert body["model"] == "isms-fast-eu"
    assert body["ismscopilot"] == {"frameworks": ["NIS_2_DE"]}
    assert response.choices[0].message.content == "Article 21 lists the measures."
    rates: Final = litellm.model_cost["ismscopilot/isms-fast-eu"]
    expected: Final = 1000 * rates["input_cost_per_token"] + 100 * rates["output_cost_per_token"]
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_ismscopilot_cost_map_rows_match_the_provider(local_model_cost_map: None) -> None:
    rows: Final = {
        name: info for name, info in litellm.model_cost.items() if info.get("litellm_provider") == "ismscopilot"
    }
    assert set(rows) == {f"ismscopilot/{alias}" for alias in ISMSCOPILOT_ALIASES}
    for info in rows.values():
        assert info["mode"] == "chat"
        assert info["supported_endpoints"] == ["/v1/chat/completions"]
        assert info["max_input_tokens"] == 128000
        assert info["max_output_tokens"] == 64000
        assert info["supports_function_calling"] is False
        assert info["supports_tool_choice"] is False
        assert info["supports_response_schema"] is False
        assert info["supports_vision"] is False
    assert rows["ismscopilot/isms-thinking"]["supports_reasoning"] is True
    assert rows["ismscopilot/isms-mini"]["supports_reasoning"] is False
    assert rows["ismscopilot/isms-mini"]["input_cost_per_token"] < rows["ismscopilot/isms-fast"]["input_cost_per_token"]
