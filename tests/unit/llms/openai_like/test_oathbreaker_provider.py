import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


@pytest.mark.parametrize("provider_key", [None, ""])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_missing_oathbreaker_key_never_sends_openai_credentials(
    monkeypatch: pytest.MonkeyPatch, provider_key: str | None, asynchronous: bool
) -> None:
    monkeypatch.delenv("OATHBREAKER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-environment-key")
    monkeypatch.setattr(litellm, "openai_key", "fake-openai-global-key")
    monkeypatch.setattr(litellm, "api_key", None)
    with respx.mock(assert_all_called=False) as upstream:
        upstream.post("https://oathbreaker.ai/v1/chat/completions").respond(401)
        if asynchronous:
            with pytest.raises(litellm.AuthenticationError, match="OATHBREAKER_API_KEY"):
                await litellm.acompletion(
                    model="oathbreaker/oathbreaker-glm-5.3",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key=provider_key,
                    num_retries=0,
                )
        else:
            with pytest.raises(litellm.AuthenticationError, match="OATHBREAKER_API_KEY"):
                litellm.completion(
                    model="oathbreaker/oathbreaker-glm-5.3",
                    messages=[{"role": "user", "content": "Hello"}],
                    api_key=provider_key,
                    num_retries=0,
                )
        assert len(upstream.calls) == 0


@pytest.mark.parametrize("model", ["oathbreaker-glm-5.3", "oathbreaker-glm-5.3-flash"])
def test_oathbreaker_resolves_environment_and_explicit_credentials(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    monkeypatch.setenv("OATHBREAKER_API_KEY", "fake-environment-key")
    assert get_llm_provider(model=f"oathbreaker/{model}") == (
        model,
        "oathbreaker",
        "fake-environment-key",
        "https://oathbreaker.ai/v1",
    )
    assert get_llm_provider(
        model=f"oathbreaker/{model}", api_key="fake-explicit-key", api_base="https://gateway.example/v1"
    ) == (model, "oathbreaker", "fake-explicit-key", "https://gateway.example/v1")


@pytest.mark.parametrize("cached_tokens", [0, 80])
@pytest.mark.parametrize("model", ["oathbreaker-glm-5.3", "oathbreaker-glm-5.3-flash"])
def test_oathbreaker_chat_forwards_tools_and_accounts_for_cache(
    monkeypatch: pytest.MonkeyPatch, local_model_cost_map: None, cached_tokens: int, model: str
) -> None:
    monkeypatch.setenv("OATHBREAKER_API_KEY", "fake-environment-key")
    tool: Final = {
        "type": "function",
        "function": {"name": "get_number", "parameters": {"type": "object", "properties": {}}},
    }
    tool_call: Final = {
        "id": "call-test",
        "type": "function",
        "function": {"name": "get_number", "arguments": "{}"},
    }
    with respx.mock() as upstream:
        route: Final = upstream.post("https://oathbreaker.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": cached_tokens},
                },
            },
        )
        response: Final = litellm.completion(
            model=f"oathbreaker/{model}",
            messages=[{"role": "user", "content": "Get a number"}],
            max_tokens=64,
            reasoning_effort="low",
            tools=[tool],
            tool_choice="required",
            parallel_tool_calls=False,
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer fake-environment-key"
    assert body == {
        "model": model,
        "messages": [{"role": "user", "content": "Get a number"}],
        "max_tokens": 64,
        "reasoning_effort": "low",
        "tools": [tool],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }
    assert response.choices[0].message.tool_calls[0].model_dump() == tool_call
    assert response.choices[0].finish_reason == "tool_calls"
    rates: Final = litellm.model_cost[f"oathbreaker/{model}"]
    expected: Final = (
        (100 - cached_tokens) * rates["input_cost_per_token"]
        + cached_tokens * rates["cache_read_input_token_cost"]
        + 5 * rates["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected)
