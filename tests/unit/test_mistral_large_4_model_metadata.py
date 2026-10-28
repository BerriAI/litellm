"""Le Chonk (Mistral Large 4) metadata and offline API-contract regressions."""

import json
from pathlib import Path

import pytest

import litellm
from litellm.llms.mistral.chat.transformation import MistralConfig

REPO_ROOT = Path(__file__).parents[2]
MODELS = ("mistral/mistral-large-4", "mistral/mistral-large-4-0")


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    """Never depend on the published cost map having this new model yet."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.get_model_info.cache_clear()
    yield
    litellm.get_model_info.cache_clear()


@pytest.mark.parametrize("model", MODELS)
def test_metadata_and_backup(model):
    main = json.loads((REPO_ROOT / "model_prices_and_context_window.json").read_text())
    backup = json.loads((REPO_ROOT / "litellm/model_prices_and_context_window_backup.json").read_text())
    assert main[model] == backup[model] == main[MODELS[0]]
    info = litellm.get_model_info(model)
    assert info["litellm_provider"] == "mistral"
    assert info["mode"] == "chat"
    assert info["max_input_tokens"] == 1048576
    assert info["input_cost_per_token"] == 6.8e-07
    assert info["output_cost_per_token"] == 2.09e-06
    assert info["cache_read_input_token_cost"] == 6.8e-08
    assert info["reasoning_effort_levels"] == ["none", "high"]
    for capability in (
        "supports_vision",
        "supports_reasoning",
        "supports_function_calling",
        "supports_tool_choice",
        "supports_response_schema",
        "supports_assistant_prefill",
        "supports_prompt_caching",
    ):
        assert info[capability] is True
    assert litellm.get_llm_provider(model=model)[:2] == (model.split("/", 1)[1], "mistral")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("effort", ["none", "high"])
def test_reasoning_effort(model, effort):
    config = MistralConfig()
    assert "reasoning_effort" in config.get_supported_openai_params(model)
    assert config.map_openai_params({"reasoning_effort": effort}, {}, model, False) == {"reasoning_effort": effort}


@pytest.mark.parametrize("model", MODELS)
def test_completion_cost(model):
    uncached_response = litellm.ModelResponse(
        model=model.split("/", 1)[1],
        usage={"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
    )
    assert litellm.completion_cost(completion_response=uncached_response, model=model) == pytest.approx(
        1000 * 6.8e-07 + 100 * 2.09e-06
    )
    response = litellm.ModelResponse(
        model=model.split("/", 1)[1],
        usage={
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": {"cached_tokens": 600},
        },
    )
    assert litellm.completion_cost(completion_response=response, model=model) == pytest.approx(
        400 * 6.8e-07 + 600 * 6.8e-08 + 100 * 2.09e-06
    )


def test_large_3_alias_is_not_retargeted():
    info = litellm.get_model_info("mistral/mistral-large-latest")
    assert info["input_cost_per_token"] == 5e-07
    assert info["max_input_tokens"] == 262144
    assert not info["supports_reasoning"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("effort", ["none", "high"])
@pytest.mark.asyncio
async def test_completion_contract(model, sync_mode, effort, respx_mock):
    """Exercise the real adapter against explicitly mocked Mistral HTTP responses."""
    api_model = model.split("/", 1)[1]
    content = "Test answer"
    if effort == "high":
        content = [
            {"type": "thinking", "thinking": [{"type": "text", "text": "Test reasoning"}]},
            {"type": "text", "text": "Test answer"},
        ]
    route = respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(
        json={
            "id": "test-large-4",
            "object": "chat.completion",
            "created": 1,
            "model": api_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": effort,
        "max_tokens": 64,
        "api_key": "test-key",
        "num_retries": 0,
    }
    response = litellm.completion(**kwargs) if sync_mode else await litellm.acompletion(**kwargs)
    assert response.choices[0].message.content == "Test answer"
    if effort == "high":
        assert response.choices[0].message.reasoning_content == "Test reasoning"
    assert route.call_count == 1
    payload = json.loads(route.calls[0].request.content)
    assert payload["model"] == api_model
    assert payload["reasoning_effort"] == effort
    assert payload["max_tokens"] == 64
    assert payload["messages"] == kwargs["messages"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_streaming_completion_contract(model, sync_mode, respx_mock):
    """Verify thinking chunks and text are normalized on the streaming path."""
    api_model = model.split("/", 1)[1]
    deltas = [
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": [{"type": "text", "text": "Test reasoning"}]}],
        },
        {"content": "Test answer"},
    ]
    events = [
        {
            "id": "test-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": api_model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    events.append(
        {
            "id": "test-stream",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": api_model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    route = respx_mock.post("https://api.mistral.ai/v1/chat/completions").respond(
        text="".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n",
        headers={"Content-Type": "text/event-stream"},
    )
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "reasoning_effort": "high",
        "stream": True,
        "api_key": "test-key",
        "num_retries": 0,
    }
    if sync_mode:
        chunks = list(litellm.completion(**kwargs))
    else:
        stream = await litellm.acompletion(**kwargs)
        chunks = [chunk async for chunk in stream]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Test answer"
    assert (
        "".join(getattr(chunk.choices[0].delta, "reasoning_content", None) or "" for chunk in chunks)
        == "Test reasoning"
    )
    assert chunks[-1].choices[0].finish_reason == "stop"
    payload = json.loads(route.calls[0].request.content)
    assert payload["model"] == api_model
    assert payload["reasoning_effort"] == "high"
    assert payload["stream"] is True
