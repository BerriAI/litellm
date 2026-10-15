import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.utils import Choices, Message, ModelResponse, Usage

_MODEL: Final = "gitgot/openai/gpt-oss-120b"
_UPSTREAM_MODEL: Final = "openai/gpt-oss-120b"
_DEFAULT_API_BASE: Final = "https://inference.gitgot.ai/v1"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_gitgot",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": _UPSTREAM_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from GitGot"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_gitgot_is_a_registered_provider():
    assert litellm.LlmProviders.GITGOT.value == "gitgot"
    assert "gitgot" in litellm.provider_list
    assert "gitgot" in litellm.constants.openai_compatible_providers


def test_gitgot_provider_resolution_keeps_the_vendor_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GITGOT_API_KEY", "gitgot-env-key")
    monkeypatch.delenv("GITGOT_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == _UPSTREAM_MODEL
    assert provider == "gitgot"
    assert api_key == "gitgot-env-key"
    assert api_base == _DEFAULT_API_BASE


def test_gitgot_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GITGOT_API_KEY", "gitgot-env-key")
    monkeypatch.setenv("GITGOT_API_BASE", "https://gitgot.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base="https://gitgot.internal.example/v1",
        api_key="gitgot-explicit-key",
    )

    assert provider == "gitgot"
    assert api_key == "gitgot-explicit-key"
    assert api_base == "https://gitgot.internal.example/v1"


def test_gitgot_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GITGOT_API_KEY", "gitgot-env-key")
    monkeypatch.setenv("GITGOT_API_BASE", "https://gitgot.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "gitgot"
    assert api_base == "https://gitgot.env.example/v1"


def test_gitgot_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    gitgot = next(provider for provider in providers if provider["litellm_provider"] == "gitgot")

    assert gitgot["provider"] == "GITGOT"
    assert gitgot["default_model_placeholder"] == _MODEL
    assert {field["key"]: field["required"] for field in gitgot["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_gitgot_endpoint_support_files_agree():
    backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"
    backup = json.loads(backup_path.read_text())["providers"]["gitgot"]
    root = json.loads(root_path.read_text())["providers"]["gitgot"]

    assert backup == root
    assert root["endpoints"]["chat_completions"] is True


def test_gitgot_chat_completion_request_sends_the_vendor_prefixed_model():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="gitgot-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer gitgot-test-key"
    assert body["model"] == _UPSTREAM_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from GitGot"


def test_gitgot_cost_is_derived_from_its_cost_map_entry(local_model_cost_map):
    entry: Final = litellm.model_cost[_MODEL]
    response: Final = ModelResponse(
        id="chatcmpl-test",
        choices=[Choices(finish_reason="stop", index=0, message=Message(content="hello", role="assistant"))],
        created=1_790_000_000,
        model=_MODEL,
        object="chat.completion",
        usage=Usage(prompt_tokens=1_000_000, completion_tokens=500_000, total_tokens=1_500_000),
    )

    cost: Final = litellm.completion_cost(completion_response=response)

    assert entry["litellm_provider"] == "gitgot"
    assert cost == pytest.approx(1_000_000 * entry["input_cost_per_token"] + 500_000 * entry["output_cost_per_token"])
    assert 0 < cost < 10
