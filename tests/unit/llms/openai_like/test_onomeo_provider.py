import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.proxy.public_endpoints.public_endpoints import (
    get_provider_fields,
    get_supported_endpoints,
    get_supported_providers,
)

_CHAT_URL: Final = "https://onomeo.com/v1/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_onomeo",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "deepseek-v4-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from onomeo"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_onomeo_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ONOMEO_API_KEY", "onomeo-test-key")
    monkeypatch.delenv("ONOMEO_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="onomeo/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4-flash"
    assert provider == "onomeo"
    assert api_key == "onomeo-test-key"
    assert api_base == "https://onomeo.com/v1"


def test_onomeo_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ONOMEO_API_KEY", "onomeo-env-key")
    monkeypatch.setenv("ONOMEO_API_BASE", "https://onomeo.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="onomeo/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base="https://onomeo.internal.example/v1",
        api_key="onomeo-explicit-key",
    )

    assert provider == "onomeo"
    assert api_key == "onomeo-explicit-key"
    assert api_base == "https://onomeo.internal.example/v1"


def test_onomeo_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ONOMEO_API_KEY", "onomeo-env-key")
    monkeypatch.setenv("ONOMEO_API_BASE", "https://onomeo.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="onomeo/deepseek-v4-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "onomeo"
    assert api_base == "https://onomeo.env.example/v1"


def test_onomeo_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ONOMEO_API_KEY", "onomeo-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="deepseek-v4-flash",
        custom_llm_provider=None,
        api_base="https://onomeo.com/v1",
        api_key=None,
    )

    assert model == "deepseek-v4-flash"
    assert provider == "onomeo"
    assert api_key == "onomeo-env-key"
    assert api_base == "https://onomeo.com/v1"


def test_onomeo_supported_params_include_both_token_limit_params():
    supported: Final = litellm.get_supported_openai_params(model="deepseek-v4-flash", custom_llm_provider="onomeo")

    assert {"max_tokens", "max_completion_tokens", "stream"} <= set(supported)


async def test_onomeo_is_listed_by_the_public_providers_endpoint():
    assert "onomeo" in await get_supported_providers()


async def test_onomeo_is_served_to_the_add_model_form():
    providers: Final = await get_provider_fields()
    entries: Final = [provider for provider in providers if provider.litellm_provider == "onomeo"]

    assert len(entries) == 1
    assert entries[0].provider == "ONOMEO"
    assert entries[0].provider_display_name == "onomeo"
    assert entries[0].default_model_placeholder == "onomeo/deepseek-v4-flash"
    assert {field.key: field.required for field in entries[0].credential_fields} == {
        "api_base": False,
        "api_key": True,
    }


async def test_public_endpoints_list_onomeo_for_chat_completions_only():
    response: Final = await get_supported_endpoints()
    listed_under: Final = [
        endpoint.key
        for endpoint in response.endpoints
        if "onomeo" in {provider.slug for provider in endpoint.providers}
    ]

    assert listed_under == ["chat_completions"]


def test_onomeo_endpoint_matrix_copies_agree():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"
    backup: Final = json.loads(backup_path.read_text(encoding="utf-8"))["providers"]["onomeo"]
    root: Final = json.loads(root_path.read_text(encoding="utf-8"))["providers"]["onomeo"]

    assert root == backup


def test_onomeo_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="onomeo/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="onomeo-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_URL
    assert request.headers["authorization"] == "Bearer onomeo-test-key"
    assert body["model"] == "deepseek-v4-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from onomeo"


@pytest.mark.parametrize(
    ("sent", "not_sent"),
    [("max_tokens", "max_completion_tokens"), ("max_completion_tokens", "max_tokens")],
)
def test_onomeo_forwards_the_token_limit_param_the_caller_used(sent: str, not_sent: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="onomeo/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="onomeo-test-key",
            **{sent: 32},
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body[sent] == 32
    assert not_sent not in body


def test_onomeo_forwards_non_openai_params_in_the_request_body():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="onomeo/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="onomeo-test-key",
            provider_specific_flag=True,
        )

    assert json.loads(route.calls.last.request.content)["provider_specific_flag"] is True
