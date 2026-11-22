import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import HTTPHandler

_DEFAULT_API_BASE: Final = "https://api.resetdata.ai/api/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_resetdata",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "zai/glm-5.3",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from ResetData", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
RESETDATA_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("resetdata/")))


def test_resetdata_provider_resolution_keeps_the_nested_model_slug(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESETDATA_API_KEY", "sk-rd-env")
    monkeypatch.delenv("RESETDATA_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="resetdata/zai/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "zai/glm-5.3"
    assert provider == "resetdata"
    assert api_key == "sk-rd-env"
    assert api_base == _DEFAULT_API_BASE


def test_resetdata_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESETDATA_API_KEY", "sk-rd-env")
    monkeypatch.setenv("RESETDATA_API_BASE", "https://resetdata.env.example/api/v1")

    _, provider, _, api_base = get_llm_provider(
        model="resetdata/zai/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "resetdata"
    assert api_base == "https://resetdata.env.example/api/v1"


def test_resetdata_explicit_credentials_win_over_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("RESETDATA_API_KEY", "sk-rd-env")
    monkeypatch.setenv("RESETDATA_API_BASE", "https://resetdata.env.example/api/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="resetdata/zai/glm-5.3",
        custom_llm_provider=None,
        api_base="https://resetdata.internal.example/api/v1",
        api_key="sk-rd-explicit",
    )

    assert provider == "resetdata"
    assert api_key == "sk-rd-explicit"
    assert api_base == "https://resetdata.internal.example/api/v1"


def test_resetdata_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="resetdata/zai/glm-5.3",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=64,
            api_key="sk-rd-test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer sk-rd-test"
    assert body["model"] == "zai/glm-5.3"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 64
    assert response.choices[0].message.content == "Hello from ResetData"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_resetdata_sends_max_completion_tokens_as_max_tokens():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="resetdata/google/gemma-4-26b-a4b-it",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=12,
            api_key="sk-rd-test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["max_tokens"] == 12
    assert "max_completion_tokens" not in body


def test_resetdata_chat_never_borrows_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RESETDATA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="RESETDATA_API_KEY"):
            litellm.completion(model="resetdata/zai/glm-5.3", messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 0


def test_resetdata_chat_on_the_http_handler_never_borrows_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RESETDATA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "true")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="RESETDATA_API_KEY"):
            litellm.completion(
                model="resetdata/zai/glm-5.3",
                messages=[{"role": "user", "content": "Say hello"}],
                client=HTTPHandler(),
            )

    assert route.call_count == 0


def test_resetdata_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="resetdata/zai/glm-5.3",
            input="Say hello",
            api_key="sk-rd-test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "zai/glm-5.3"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from ResetData"


@pytest.mark.usefixtures("local_model_cost_map")
def test_resetdata_wildcard_expands_to_every_resetdata_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert RESETDATA_MODELS
    assert set(get_known_models_from_wildcard("resetdata/*")) == set(RESETDATA_MODELS)


def test_resetdata_backup_maps_mirror_the_root_maps():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    cost_backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    endpoints: Final = json.loads((package_root.parent / "provider_endpoints_support.json").read_text())
    endpoints_backup: Final = json.loads((package_root / "provider_endpoints_support_backup.json").read_text())
    resetdata_rows: Final = {name: entry for name, entry in cost_map.items() if name.startswith("resetdata/")}

    assert tuple(sorted(resetdata_rows)) == RESETDATA_MODELS
    assert resetdata_rows == {name: cost_backup[name] for name in resetdata_rows}
    assert endpoints["providers"]["resetdata"] == endpoints_backup["providers"]["resetdata"]


def test_resetdata_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    resetdata: Final = next(provider for provider in providers if provider["litellm_provider"] == "resetdata")

    assert resetdata["provider"] == "RESETDATA"
    assert resetdata["default_model_placeholder"] in RESETDATA_MODELS
    assert {field["key"]: field["required"] for field in resetdata["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
