import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

INFERENCEAPIS_API_BASE: Final = "https://api.inferenceapis.com/v1"
INFERENCEAPIS_MODEL_ID: Final = "deepseek-ai/DeepSeek-V4-Flash"
INFERENCEAPIS_MODEL: Final = f"inferenceapis/{INFERENCEAPIS_MODEL_ID}"


def _chat_completion_body(usage: dict[str, object]) -> dict[str, object]:
    return {
        "id": "chatcmpl_inferenceapis",
        "object": "chat.completion",
        "created": 1_789_550_000,
        "model": INFERENCEAPIS_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from Inference APIs"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def test_inferenceapis_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCEAPIS_API_KEY", "inferenceapis-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=INFERENCEAPIS_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == INFERENCEAPIS_MODEL_ID
    assert provider == "inferenceapis"
    assert api_key == "inferenceapis-env-key"
    assert api_base == INFERENCEAPIS_API_BASE


def test_inferenceapis_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCEAPIS_API_KEY", "inferenceapis-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="inferenceapis/openai/gpt-oss-120b",
        custom_llm_provider=None,
        api_base="https://inferenceapis.internal.example/v1",
        api_key="inferenceapis-explicit-key",
    )

    assert provider == "inferenceapis"
    assert api_key == "inferenceapis-explicit-key"
    assert api_base == "https://inferenceapis.internal.example/v1"


def test_inferenceapis_provider_detected_from_api_base():
    model, provider, _, api_base = get_llm_provider(
        model="zai-org/GLM-5.3",
        custom_llm_provider=None,
        api_base=INFERENCEAPIS_API_BASE,
        api_key=None,
    )

    assert model == "zai-org/GLM-5.3"
    assert provider == "inferenceapis"
    assert api_base == INFERENCEAPIS_API_BASE


def test_inferenceapis_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    inferenceapis: Final = next(provider for provider in providers if provider["litellm_provider"] == "inferenceapis")

    assert inferenceapis["provider"] == "INFERENCEAPIS"
    assert inferenceapis["provider_display_name"] == "Inference APIs"
    assert {field["key"]: field["required"] for field in inferenceapis["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_inferenceapis_supported_endpoints():
    matrix_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    endpoints: Final = json.loads(matrix_path.read_text())["providers"]["inferenceapis"]["endpoints"]

    assert {name for name, supported in endpoints.items() if supported} == {"chat_completions", "responses"}


def test_inferenceapis_cost_map_rows_only_list_registered_endpoints():
    provider: Final = JSONProviderRegistry.get("inferenceapis")
    assert provider is not None
    registered: Final = set(provider.supported_endpoints)
    rows: Final = {
        name: info for name, info in litellm.model_cost.items() if info.get("litellm_provider") == "inferenceapis"
    }

    assert rows
    assert all(set(info["supported_endpoints"]) <= registered for info in rows.values())


def test_inferenceapis_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{INFERENCEAPIS_API_BASE}/chat/completions").respond(
            200, json=_chat_completion_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        )
        response: Final = litellm.completion(
            model=INFERENCEAPIS_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=7,
            api_key="inferenceapis-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer inferenceapis-test-key"
    assert body["model"] == INFERENCEAPIS_MODEL_ID
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 7
    assert "max_tokens" not in body
    assert response.choices[0].message.content == "Hello from Inference APIs"


def test_inferenceapis_chat_completion_reads_key_and_base_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INFERENCEAPIS_API_KEY", "inferenceapis-env-key")
    monkeypatch.setenv("INFERENCEAPIS_API_BASE", "https://inferenceapis.internal.example/v1")
    with respx.mock() as upstream:
        route: Final = upstream.post("https://inferenceapis.internal.example/v1/chat/completions").respond(
            200, json=_chat_completion_body({"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7})
        )
        litellm.completion(model=INFERENCEAPIS_MODEL, messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer inferenceapis-env-key"


def test_inferenceapis_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{INFERENCEAPIS_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp_inferenceapis",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": INFERENCEAPIS_MODEL_ID,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_inferenceapis",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Inference APIs", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=INFERENCEAPIS_MODEL,
            input="Say hello",
            api_key="inferenceapis-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer inferenceapis-test-key"
    assert body["model"] == INFERENCEAPIS_MODEL_ID
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Inference APIs"


def test_inferenceapis_chat_completion_cost_bills_cached_tokens_at_cache_read_rate():
    row: Final = litellm.model_cost[INFERENCEAPIS_MODEL]
    with respx.mock() as upstream:
        upstream.post(f"{INFERENCEAPIS_API_BASE}/chat/completions").respond(
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
            model=INFERENCEAPIS_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="inferenceapis-test-key",
        )

    assert row["cache_read_input_token_cost"] < row["input_cost_per_token"]
    assert litellm.completion_cost(completion_response=response, model=INFERENCEAPIS_MODEL) == pytest.approx(
        200 * row["input_cost_per_token"] + 800 * row["cache_read_input_token_cost"] + 20 * row["output_cost_per_token"]
    )
