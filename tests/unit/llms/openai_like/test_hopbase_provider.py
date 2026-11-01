import json
from datetime import datetime, timezone
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.utils import trim_messages


def _chat_response(model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    }


def test_hopbase_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOPBASE_API_KEY", "fake-environment-key")
    assert get_llm_provider(model="hopbase/glm-5.3") == (
        "glm-5.3",
        "hopbase",
        "fake-environment-key",
        "https://api.hop-base.com/v1",
    )
    assert get_llm_provider(
        model="hopbase/glm-5.3", api_key="fake-explicit-key", api_base="https://gateway.example/v1"
    ) == ("glm-5.3", "hopbase", "fake-explicit-key", "https://gateway.example/v1")


def test_hopbase_chat_sends_openai_request_and_accounts_for_cache(local_model_cost_map: None) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.hop-base.com/v1/chat/completions").respond(
            200, json=_chat_response("glm-5.3", prompt_tokens=100, cached_tokens=80, completion_tokens=5)
        )
        response: Final = litellm.completion(
            model="hopbase/glm-5.3",
            api_key="fake-test-key",
            messages=[{"role": "user", "content": "Say hello"}],
        )
    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer fake-test-key"
    assert json.loads(request.content)["model"] == "glm-5.3"
    assert response.choices[0].message.content == "hello"
    rates: Final = litellm.model_cost["hopbase/glm-5.3"]
    expected: Final = (
        20 * rates["input_cost_per_token"]
        + 80 * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


def test_hopbase_long_prompt_uses_the_upper_tier(local_model_cost_map: None) -> None:
    rates: Final = litellm.model_cost["hopbase/gpt-5.6-sol"]
    with respx.mock() as upstream:
        upstream.post("https://api.hop-base.com/v1/chat/completions").respond(
            200, json=_chat_response("gpt-5.6-sol", prompt_tokens=300_000, cached_tokens=0, completion_tokens=10)
        )
        response: Final = litellm.completion(
            model="hopbase/gpt-5.6-sol", api_key="fake-test-key", messages=[{"role": "user", "content": "hi"}]
        )
    expected: Final = (
        300_000 * rates["input_cost_per_token_above_272k_tokens"]
        + 10 * rates["output_cost_per_token_above_272k_tokens"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("when", "rate_source"),
    [
        (datetime(2026, 11, 2, 2, 0, tzinfo=timezone.utc), "peak"),  # Monday 10:00 Beijing
        (datetime(2026, 11, 2, 12, 0, tzinfo=timezone.utc), "off_peak"),  # Monday 20:00 Beijing
        (datetime(2026, 11, 1, 2, 0, tzinfo=timezone.utc), "off_peak"),  # Sunday
    ],
)
def test_hopbase_deepseek_flash_bills_off_peak_hours(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch, when: datetime, rate_source: str
) -> None:
    from litellm.litellm_core_utils.llm_cost_calc import utils as cost_utils

    monkeypatch.setattr(cost_utils, "current_billing_time", lambda: when)
    info: Final = litellm.model_cost["hopbase/deepseek-v4.1-flash-1017"]
    rates: Final = info if rate_source == "peak" else info["off_peak_pricing"]
    prompt_cost, completion_cost = litellm.cost_per_token(
        model="hopbase/deepseek-v4.1-flash-1017", prompt_tokens=1000, completion_tokens=100
    )
    assert prompt_cost == pytest.approx(1000 * rates["input_cost_per_token"])
    assert completion_cost == pytest.approx(100 * rates["output_cost_per_token"])


def test_hopbase_models_trim_to_their_input_budget(local_model_cost_map: None) -> None:
    model_names: Final = tuple(model for model in litellm.model_cost if model.startswith("hopbase/"))
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
