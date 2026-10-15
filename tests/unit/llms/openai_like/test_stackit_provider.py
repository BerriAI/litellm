import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_BASE_URL: Final = "https://api.openai-compat.model-serving.eu01.onstackit.cloud/v1"
_REPO_ROOT: Final = Path(litellm.__file__).parent.parent

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_stackit",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "openai/gpt-oss-120b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from STACKIT"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}

_EMBEDDING: Final = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    "model": "intfloat/e5-mistral-7b-instruct",
    "usage": {"prompt_tokens": 3, "total_tokens": 3},
}


def test_stackit_is_a_registered_provider():
    assert litellm.LlmProviders.STACKIT.value == "stackit"
    assert "stackit" in litellm.provider_list
    assert "stackit" in litellm.constants.openai_compatible_providers
    assert _BASE_URL in litellm.constants.openai_compatible_endpoints


def test_stackit_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("STACKIT_API_KEY", "stackit-test-key")
    monkeypatch.delenv("STACKIT_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="stackit/openai/gpt-oss-120b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "openai/gpt-oss-120b"
    assert provider == "stackit"
    assert api_key == "stackit-test-key"
    assert api_base == _BASE_URL


def test_stackit_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("STACKIT_API_KEY", "stackit-env-key")
    monkeypatch.setenv("STACKIT_API_BASE", "https://stackit.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="stackit/openai/gpt-oss-120b",
        custom_llm_provider=None,
        api_base="https://stackit.internal.example/v1",
        api_key="stackit-explicit-key",
    )

    assert provider == "stackit"
    assert api_key == "stackit-explicit-key"
    assert api_base == "https://stackit.internal.example/v1"


def test_stackit_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("STACKIT_API_KEY", "stackit-env-key")
    monkeypatch.setenv("STACKIT_API_BASE", "https://stackit.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="stackit/openai/gpt-oss-120b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "stackit"
    assert api_base == "https://stackit.env.example/v1"


def test_stackit_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("STACKIT_API_KEY", "stackit-env-key")

    # STACKIT model ids carry their own org prefix (openai/, google/, Qwen/, ...). "openai/..." on its own
    # would be routed to OpenAI, so autodetection is only meaningful for ids without a provider-like prefix.
    model, provider, api_key, api_base = get_llm_provider(
        model="Qwen/Qwen3.8-27B",
        custom_llm_provider=None,
        api_base=_BASE_URL,
        api_key=None,
    )

    assert model == "Qwen/Qwen3.8-27B"
    assert provider == "stackit"
    assert api_key == "stackit-env-key"
    assert api_base == _BASE_URL


def test_stackit_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    stackit = next(provider for provider in providers if provider["litellm_provider"] == "stackit")

    assert stackit["provider"] == "STACKIT"
    assert stackit["provider_display_name"] == "STACKIT"
    assert stackit["default_model_placeholder"] == "stackit/openai/gpt-oss-120b"
    assert {field["key"]: field["required"] for field in stackit["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_stackit_supported_endpoints():
    expected: Final = {
        "chat_completions": True,
        "messages": True,
        "responses": True,
        "embeddings": True,
        "image_generations": False,
        "audio_transcriptions": False,
        "audio_speech": False,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
        "interactions": False,
    }
    backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path = _REPO_ROOT / "provider_endpoints_support.json"

    assert json.loads(backup_path.read_text())["providers"]["stackit"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["stackit"]["endpoints"] == expected


def test_stackit_cost_map_entries():
    root = json.loads((_REPO_ROOT / "model_prices_and_context_window.json").read_text())
    backup = json.loads((Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json").read_text())

    for cost_map in (root, backup):
        stackit = {k: v for k, v in cost_map.items() if v.get("litellm_provider") == "stackit"}
        assert stackit, "no stackit entries in cost map"
        assert all(key.startswith("stackit/") for key in stackit)
        assert {v["mode"] for v in stackit.values()} == {"chat", "embedding"}

        # STACKIT price tiers (EUR per 1M tokens): standard 0.15/0.25, plus 0.45/0.65, premium 1.50/1.75
        gpt_oss_20b = stackit["stackit/openai/gpt-oss-20b"]
        assert gpt_oss_20b["input_cost_per_token"] == pytest.approx(0.15e-6)
        assert gpt_oss_20b["output_cost_per_token"] == pytest.approx(0.25e-6)

        gpt_oss_120b = stackit["stackit/openai/gpt-oss-120b"]
        assert gpt_oss_120b["input_cost_per_token"] == pytest.approx(0.45e-6)
        assert gpt_oss_120b["output_cost_per_token"] == pytest.approx(0.65e-6)
        assert gpt_oss_120b["supports_reasoning"] is True
        assert gpt_oss_120b["supports_function_calling"] is True

        qwen_vl = stackit["stackit/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"]
        assert qwen_vl["input_cost_per_token"] == pytest.approx(1.50e-6)
        assert qwen_vl["output_cost_per_token"] == pytest.approx(1.75e-6)
        assert qwen_vl["supports_vision"] is True

        e5 = stackit["stackit/intfloat/e5-mistral-7b-instruct"]
        assert e5["input_cost_per_token"] == pytest.approx(0.02e-6)
        assert e5["output_cost_per_token"] == 0.0
        assert e5["output_vector_size"] == 4096

        qwen_embedding = stackit["stackit/Qwen/Qwen3-VL-Embedding-8B"]
        assert qwen_embedding["input_cost_per_token"] == pytest.approx(0.08e-6)
        assert qwen_embedding["max_input_tokens"] == 32000


def test_stackit_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE_URL}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="stackit/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="stackit-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_BASE_URL}/chat/completions"
    assert request.headers["authorization"] == "Bearer stackit-test-key"
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from STACKIT"


def test_stackit_chat_completion_cost_uses_cost_map():
    with respx.mock() as upstream:
        upstream.post(f"{_BASE_URL}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="stackit/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="stackit-test-key",
        )

    # 4 prompt tokens * 0.45e-6 + 3 completion tokens * 0.65e-6
    assert response._hidden_params["response_cost"] == pytest.approx(4 * 0.45e-6 + 3 * 0.65e-6)


def test_stackit_embedding_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE_URL}/embeddings").respond(200, json=_EMBEDDING)
        response: Final = litellm.embedding(
            model="stackit/intfloat/e5-mistral-7b-instruct",
            input=["hello"],
            api_key="stackit-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer stackit-test-key"
    assert body["model"] == "intfloat/e5-mistral-7b-instruct"
    assert body["input"] == ["hello"]
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


def test_stackit_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE_URL}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="stackit/openai/gpt-oss-120b",
            input="Say hello",
            api_key="stackit-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer stackit-test-key"
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from STACKIT"


@pytest.mark.asyncio
async def test_stackit_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE_URL}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="stackit/openai/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="stackit-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer stackit-test-key"
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from STACKIT"
