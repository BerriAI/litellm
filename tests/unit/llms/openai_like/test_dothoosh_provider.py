import json
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

DOTHOOSH_API_BASE: Final = "https://gateway.dothoosh.com/v1"
DOTHOOSH_MODEL_ID: Final = "deepseek-flash"
DOTHOOSH_MODEL: Final = f"dothoosh/{DOTHOOSH_MODEL_ID}"


def _chat_completion_body(usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl_dothoosh",
        "object": "chat.completion",
        "created": 1_789_550_000,
        "model": DOTHOOSH_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from DotHoosh"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def test_dothoosh_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DOTHOOSH_API_KEY", "dothoosh-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=DOTHOOSH_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == DOTHOOSH_MODEL_ID
    assert provider == "dothoosh"
    assert api_key == "dothoosh-env-key"
    assert api_base == DOTHOOSH_API_BASE


def test_dothoosh_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DOTHOOSH_API_KEY", "dothoosh-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model=DOTHOOSH_MODEL,
        custom_llm_provider=None,
        api_base="https://dothoosh.internal.example/v1",
        api_key="dothoosh-explicit-key",
    )

    assert provider == "dothoosh"
    assert api_key == "dothoosh-explicit-key"
    assert api_base == "https://dothoosh.internal.example/v1"


def test_dothoosh_registers_chat_and_responses_but_not_messages():
    provider: Final = JSONProviderRegistry.get("dothoosh")
    assert provider is not None
    assert set(provider.supported_endpoints) == {"/v1/chat/completions", "/v1/responses"}
    assert "/v1/messages" not in provider.supported_endpoints


def test_dothoosh_cost_map_rows_only_list_registered_endpoints():
    provider: Final = JSONProviderRegistry.get("dothoosh")
    assert provider is not None
    registered: Final = set(provider.supported_endpoints)
    rows: Final = {
        name: info for name, info in litellm.model_cost.items() if info.get("litellm_provider") == "dothoosh"
    }

    assert rows
    assert all(set(info["supported_endpoints"]) <= registered for info in rows.values())
    assert all("/v1/messages" not in info["supported_endpoints"] for info in rows.values())


def test_dothoosh_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{DOTHOOSH_API_BASE}/chat/completions").respond(
            200, json=_chat_completion_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        )
        response: Final = litellm.completion(
            model=DOTHOOSH_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=7,
            api_key="dothoosh-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer dothoosh-test-key"
    assert body["model"] == DOTHOOSH_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 7
    assert "max_tokens" not in body
    assert response.choices[0].message.content == "Hello from DotHoosh"


def test_dothoosh_chat_completion_reads_key_and_base_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DOTHOOSH_API_KEY", "dothoosh-env-key")
    monkeypatch.setenv("DOTHOOSH_API_BASE", "https://dothoosh.internal.example/v1")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://dothoosh.internal.example/v1/chat/completions").respond(
            200, json=_chat_completion_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        )
        litellm.completion(model=DOTHOOSH_MODEL, messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer dothoosh-env-key"


def test_dothoosh_chat_completion_sends_provider_specific_params_in_the_body():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{DOTHOOSH_API_BASE}/chat/completions").respond(
            200, json=_chat_completion_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        )
        response: Final = litellm.completion(
            model=DOTHOOSH_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            top_k=5,
            chat_template_kwargs={"enable_thinking": False},
            api_key="dothoosh-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["top_k"] == 5
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert response.choices[0].message.content == "Hello from DotHoosh"


def test_dothoosh_context_length_error_raises_context_window_exceeded():
    with respx.mock() as upstream:
        upstream.post(f"{DOTHOOSH_API_BASE}/chat/completions").respond(
            400,
            json={
                "error": {
                    "message": "Input too long: 1500055 input tokens, limit is 1000000 for this model",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "context_length_exceeded",
                }
            },
        )
        with pytest.raises(litellm.ContextWindowExceededError):
            litellm.completion(
                model=DOTHOOSH_MODEL,
                messages=[{"role": "user", "content": "Say hello"}],
                api_key="dothoosh-test-key",
            )


def test_dothoosh_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{DOTHOOSH_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp_dothoosh",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": DOTHOOSH_MODEL_ID,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_dothoosh",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from DotHoosh", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=DOTHOOSH_MODEL,
            input="Say hello",
            api_key="dothoosh-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer dothoosh-test-key"
    assert body["model"] == DOTHOOSH_MODEL_ID
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from DotHoosh"


def test_dothoosh_chat_completion_cost_bills_cached_tokens_at_cache_read_rate():
    row: Final = litellm.model_cost[DOTHOOSH_MODEL]
    with respx.mock() as upstream:
        upstream.post(f"{DOTHOOSH_API_BASE}/chat/completions").respond(
            200,
            json=_chat_completion_body(
                {
                    "prompt_tokens": 1000,
                    "completion_tokens": 20,
                    "total_tokens": 1020,
                    "prompt_tokens_details": {"cached_tokens": 800},
                }
            ),
        )
        response: Final = litellm.completion(
            model=DOTHOOSH_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="dothoosh-test-key",
        )

    assert row["cache_read_input_token_cost"] < row["input_cost_per_token"]
    assert litellm.completion_cost(completion_response=response, model=DOTHOOSH_MODEL) == pytest.approx(
        200 * row["input_cost_per_token"] + 800 * row["cache_read_input_token_cost"] + 20 * row["output_cost_per_token"]
    )
