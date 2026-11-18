import json
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

QSP_MODELS: Final = (
    "quicksilverpro/claude-opus-5-5",
    "quicksilverpro/claude-sonnet-5-5",
    "quicksilverpro/deepseek-v4.1-flash",
    "quicksilverpro/gemini-3.8-flash",
    "quicksilverpro/glm-5.3-flash",
    "quicksilverpro/gpt-6.1-sol",
    "quicksilverpro/mimo-v2.6-pro",
    "quicksilverpro/qwen3.8-27b",
)


@pytest.mark.parametrize("model", QSP_MODELS)
@pytest.mark.parametrize("prompt_tokens", (1000, 200001))
def test_quicksilverpro_model_cost_and_capabilities(model: str, prompt_tokens: int) -> None:
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model, prompt_tokens=prompt_tokens, completion_tokens=1000, custom_llm_provider="quicksilverpro"
    )
    model_info: Final = litellm.get_model_info(model)
    entry: Final = litellm.model_cost[model]
    input_rate: Final = (
        entry.get("input_cost_per_token_above_200k_tokens", model_info["input_cost_per_token"])
        if prompt_tokens > 200000
        else model_info["input_cost_per_token"]
    )
    output_rate: Final = (
        entry.get("output_cost_per_token_above_200k_tokens", model_info["output_cost_per_token"])
        if prompt_tokens > 200000
        else model_info["output_cost_per_token"]
    )
    assert prompt_cost == pytest.approx(input_rate * prompt_tokens)
    assert completion_cost == pytest.approx(output_rate * 1000)
    assert model_info["output_cost_per_token"] > 0
    max_output: Final = model_info.get("max_output_tokens") or model_info["max_input_tokens"]
    assert model_info["max_tokens"] == max_output <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "quicksilverpro"
    assert model_info["mode"] == "chat"
    assert litellm.supports_vision(model) is model_info["supports_vision"]


@pytest.mark.parametrize(
    ("model", "api_key", "api_base", "env_base", "expected_base", "expected_key"),
    [
        (
            "quicksilverpro/claude-sonnet-5-5",
            "explicit-test-key",
            None,
            None,
            "https://api.quicksilverpro.io/v1",
            "explicit-test-key",
        ),
        (
            "quicksilverpro/claude-sonnet-5-5",
            None,
            None,
            "https://env.example/v1",
            "https://env.example/v1",
            "environment-test-key",
        ),
        (
            "quicksilverpro/claude-sonnet-5-5",
            "explicit-test-key",
            "https://override.example/v1",
            "https://env.example/v1",
            "https://override.example/v1",
            "explicit-test-key",
        ),
        (
            "claude-sonnet-5-5",
            None,
            "https://api.quicksilverpro.io/v1",
            None,
            "https://api.quicksilverpro.io/v1",
            "environment-test-key",
        ),
    ],
    ids=["explicit-key", "environment", "explicit-overrides-environment", "url-autodetection"],
)
@pytest.mark.parametrize("use_router", (False, True), ids=["completion", "router"])
def test_quicksilverpro_chat_completion_request(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    api_key: str | None,
    api_base: str | None,
    env_base: str | None,
    expected_base: str,
    expected_key: str,
    use_router: bool,
) -> None:
    monkeypatch.setenv("QUICKSILVERPRO_API_KEY", "environment-test-key")
    if env_base is None:
        monkeypatch.delenv("QUICKSILVERPRO_API_BASE", raising=False)
    else:
        monkeypatch.setenv("QUICKSILVERPRO_API_BASE", env_base)
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{expected_base}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_quicksilverpro",
                "object": "chat.completion",
                "created": 1789550000,
                "model": "claude-sonnet-5-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from QuickSilver Pro"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = (
            litellm.Router(
                model_list=[
                    {
                        "model_name": "quicksilverpro-chat",
                        "litellm_params": {"model": model, "api_key": api_key, "api_base": api_base},
                    }
                ]
            ).completion(
                model="quicksilverpro-chat",
                messages=[{"role": "user", "content": "Say hello"}],
                max_completion_tokens=128,
            )
            if use_router
            else litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Say hello"}],
                max_completion_tokens=128,
                api_key=api_key,
                api_base=api_base,
            )
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{expected_base}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {expected_key}"
    assert body == {
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 128,
    }
    assert response.choices[0].message.content == "Hello from QuickSilver Pro"


def test_quicksilverpro_responses_request() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.quicksilverpro.io/v1/responses").respond(
            200,
            json={
                "id": "resp_quicksilverpro",
                "object": "response",
                "created_at": 1789550000,
                "model": "claude-sonnet-5-5",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_quicksilverpro",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from QuickSilver Pro", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="quicksilverpro/claude-sonnet-5-5", input="Say hello", api_key="quicksilverpro-test-key"
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.quicksilverpro.io/v1/responses"
    assert request.headers["authorization"] == "Bearer quicksilverpro-test-key"
    assert body == {"model": "claude-sonnet-5-5", "input": "Say hello"}
    assert response.output[0].content[0].text == "Hello from QuickSilver Pro"


@pytest.mark.asyncio
async def test_quicksilverpro_anthropic_messages_translates_to_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.quicksilverpro.io/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_quicksilverpro",
                "object": "chat.completion",
                "created": 1789550000,
                "model": "claude-sonnet-5-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from QuickSilver Pro"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="quicksilverpro/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="quicksilverpro-test-key",
        )
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer quicksilverpro-test-key"
    assert body == {
        "model": "claude-sonnet-5-5",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 32,
    }
    assert response["content"] == [{"type": "text", "text": "Hello from QuickSilver Pro"}]
