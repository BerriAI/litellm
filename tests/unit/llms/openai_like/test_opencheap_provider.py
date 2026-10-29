import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm

REPO_ROOT: Final = Path(litellm.__file__).parents[1]
OPENCHEAP_MODELS: Final = (
    "opencheap/deepseek-v4-pro",
    "opencheap/deepseek-v4.1-flash",
    "opencheap/glm-5.3",
    "opencheap/glm-5.3-flash",
    "opencheap/gpt-oss-120b",
    "opencheap/kimi-k3",
    "opencheap/llama-3.3-70b",
    "opencheap/minimax-m2.5",
    "opencheap/qwen3-coder",
    "opencheap/qwen3.5-397b-a17b",
)


def test_opencheap_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPENCHEAP_API_KEY", "kl_live_env_key")
    monkeypatch.delenv("OPENCHEAP_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="opencheap/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.3-flash"
    assert provider == "opencheap"
    assert api_key == "kl_live_env_key"
    assert api_base == "https://api.opencheap.si/v1"


def test_opencheap_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPENCHEAP_API_KEY", "kl_live_env_key")
    monkeypatch.setenv("OPENCHEAP_API_BASE", "https://opencheap.internal.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="opencheap/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "opencheap"
    assert api_base == "https://opencheap.internal.example/v1"


def test_opencheap_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OPENCHEAP_API_KEY", "kl_live_env_key")

    _, provider, api_key, api_base = get_llm_provider(
        model="opencheap/glm-5.3-flash",
        custom_llm_provider=None,
        api_base="https://opencheap.internal.example/v1",
        api_key="kl_live_explicit_key",
    )

    assert provider == "opencheap"
    assert api_key == "kl_live_explicit_key"
    assert api_base == "https://opencheap.internal.example/v1"


def test_opencheap_is_detected_from_its_base_url():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider(
        model="glm-5.3-flash",
        custom_llm_provider=None,
        api_base="https://api.opencheap.si/v1",
        api_key="kl_live_explicit_key",
    )

    assert model == "glm-5.3-flash"
    assert provider == "opencheap"
    assert api_base == "https://api.opencheap.si/v1"


def test_opencheap_is_registered_as_a_json_provider():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = JSONProviderRegistry.get("opencheap")

    assert config is not None
    assert litellm.LlmProviders.OPENCHEAP.value == "opencheap"
    assert "opencheap" in litellm.provider_list
    assert config.base_url == "https://api.opencheap.si/v1"
    assert config.api_key_env == "OPENCHEAP_API_KEY"
    assert config.api_base_env == "OPENCHEAP_API_BASE"
    assert config.supported_endpoints == ["/v1/chat/completions", "/v1/responses"]


def test_opencheap_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    entries = [provider for provider in providers if provider["litellm_provider"] == "opencheap"]

    assert len(entries) == 1
    assert entries[0]["provider"] == "OPENCHEAP"
    assert entries[0]["provider_display_name"] == "OpenCheap"
    assert entries[0]["default_model_placeholder"] == "opencheap/glm-5.3-flash"
    assert {field["key"]: field["required"] for field in entries[0]["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


@pytest.mark.parametrize(
    "matrix_path", ["provider_endpoints_support.json", "litellm/provider_endpoints_support_backup.json"]
)
def test_opencheap_supported_endpoints(matrix_path: str):
    entry = json.loads((REPO_ROOT / matrix_path).read_text())["providers"]["opencheap"]

    assert entry["display_name"] == "OpenCheap (`opencheap`)"
    assert entry["url"] == "https://docs.litellm.ai/docs/providers/opencheap"
    assert entry["endpoints"]["chat_completions"] is True
    assert entry["endpoints"]["responses"] is True
    assert entry["endpoints"]["embeddings"] is False
    assert entry["endpoints"]["image_generations"] is False


def test_opencheap_chat_completion_request_and_cost():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.opencheap.si/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_opencheap",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "glm-5.3-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from OpenCheap"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            },
        )
        response: Final = litellm.completion(
            model="opencheap/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="kl_live_test_key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    cost_info: Final = litellm.model_cost["opencheap/glm-5.3-flash"]
    assert route.call_count == 1
    assert str(request.url) == "https://api.opencheap.si/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer kl_live_test_key"
    assert body["model"] == "glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from OpenCheap"
    assert response._hidden_params["response_cost"] == pytest.approx(
        1000 * cost_info["input_cost_per_token"] + 500 * cost_info["output_cost_per_token"]
    )
    assert response._hidden_params["response_cost"] > 0


def test_opencheap_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.opencheap.si/v1/responses").respond(
            200,
            json={
                "id": "resp_opencheap",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "glm-5.3-flash",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_opencheap",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from OpenCheap", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="opencheap/glm-5.3-flash",
            input="Say hello",
            api_key="kl_live_test_key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.opencheap.si/v1/responses"
    assert request.headers["authorization"] == "Bearer kl_live_test_key"
    assert body["model"] == "glm-5.3-flash"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from OpenCheap"


def test_opencheap_cost_map_rows():
    model_cost = json.loads((REPO_ROOT / "model_prices_and_context_window.json").read_text())
    rows = {name: row for name, row in model_cost.items() if row.get("litellm_provider") == "opencheap"}

    assert set(rows) == set(OPENCHEAP_MODELS)
    for name, row in rows.items():
        assert row["mode"] == "chat", name
        assert row["max_input_tokens"] >= 131072, name
        assert 0 < row["input_cost_per_token"] < 1e-6, name
        assert 0 < row["output_cost_per_token"] < 1e-6, name
        assert 0 < row["cache_read_input_token_cost"] <= row["input_cost_per_token"], name
        assert row["supports_function_calling"] is True, name
        assert row["supports_tool_choice"] is True, name
        assert row["supports_response_schema"] is True, name


def test_opencheap_cost_map_backup_matches_root():
    model_cost = json.loads((REPO_ROOT / "model_prices_and_context_window.json").read_text())
    backup = json.loads((REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json").read_text())

    for name in OPENCHEAP_MODELS:
        assert backup[name] == model_cost[name], f"{name} differs between the root and backup cost map"
