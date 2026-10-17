import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

MODEL: Final = "alpha/Qwen/Qwen3.8-27B-FP8"
DEFAULT_BASE: Final = "https://alpha.sh/v1"


def _completion_body(content: str) -> dict:
    return {
        "id": "chatcmpl-alpha",
        "object": "chat.completion",
        "created": 1_791_050_944,
        "model": "Qwen/Qwen3.8-27B-FP8",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1_000, "completion_tokens": 2_000, "total_tokens": 3_000},
    }


@pytest.fixture
def server_key(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("ALPHA_API_KEY", "alpha-server-key")
    monkeypatch.delenv("ALPHA_API_BASE", raising=False)
    return "alpha-server-key"


def test_alpha_model_uses_the_server_key_on_the_default_base(server_key: str):
    model, provider, api_key, api_base = get_llm_provider(model=MODEL, custom_llm_provider=None)

    assert model == "Qwen/Qwen3.8-27B-FP8"
    assert provider == "alpha"
    assert api_key == server_key
    assert api_base == DEFAULT_BASE


def test_alpha_explicit_credentials_win_over_the_server_key(server_key: str):
    _, provider, api_key, api_base = get_llm_provider(
        model=MODEL,
        custom_llm_provider=None,
        api_base="https://alpha.internal.example/v1",
        api_key="alpha-explicit-key",
    )

    assert provider == "alpha"
    assert api_key == "alpha-explicit-key"
    assert api_base == "https://alpha.internal.example/v1"


def test_alpha_server_key_is_not_sent_to_a_caller_chosen_base(server_key: str):
    with pytest.raises(litellm.BadRequestError, match="ALPHA_API_KEY"):
        get_llm_provider(model=MODEL, custom_llm_provider=None, api_base="https://attacker.example/v1")


def test_alpha_completion_refuses_a_caller_chosen_base_before_any_request(server_key: str):
    with respx.mock(assert_all_called=False) as upstream:
        attacker: Final = upstream.post("https://attacker.example/v1/chat/completions").respond(
            200, json=_completion_body("leaked")
        )
        with pytest.raises(litellm.BadRequestError, match="ALPHA_API_KEY"):
            litellm.completion(
                model=MODEL,
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://attacker.example/v1",
            )

    assert attacker.call_count == 0


def test_alpha_server_key_follows_the_operator_base(server_key: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ALPHA_API_BASE", "https://alpha.internal.example/v1")

    _, _, api_key, api_base = get_llm_provider(
        model=MODEL, custom_llm_provider=None, api_base="https://alpha.internal.example/v1"
    )

    assert api_key == server_key
    assert api_base == "https://alpha.internal.example/v1"


def test_alpha_chat_completion_request_is_priced_from_the_cost_map(server_key: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(DEFAULT_BASE + "/chat/completions").respond(
            200, json=_completion_body("Hello from Alpha.sh")
        )
        response: Final = litellm.completion(model=MODEL, messages=[{"role": "user", "content": "Say hello"}])

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    model_info: Final = litellm.get_model_info(MODEL)
    assert request.headers["authorization"] == "Bearer " + server_key
    assert body["model"] == "Qwen/Qwen3.8-27B-FP8"
    assert response.choices[0].message.content == "Hello from Alpha.sh"
    assert response._hidden_params["response_cost"] == pytest.approx(
        1_000 * model_info["input_cost_per_token"] + 2_000 * model_info["output_cost_per_token"]
    )


def test_alpha_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    alpha_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("alpha/")}

    assert alpha_entries
    assert alpha_entries == {name: backup[name] for name in alpha_entries}


@pytest.mark.asyncio
async def test_alpha_is_offered_in_the_add_model_form_with_a_required_key():
    from litellm.proxy.public_endpoints.public_endpoints import get_provider_fields

    providers: Final = await get_provider_fields()
    alpha: Final = next(provider for provider in providers if provider.litellm_provider == "alpha")

    assert alpha.default_model_placeholder.startswith("alpha/")
    assert {field.key: field.required for field in alpha.credential_fields} == {"api_base": False, "api_key": True}


@pytest.mark.asyncio
async def test_alpha_is_listed_for_chat_completions_only():
    from litellm.proxy.public_endpoints.public_endpoints import get_supported_endpoints

    response: Final = await get_supported_endpoints()
    serving: Final = {
        entry.key for entry in response.endpoints if any(provider.slug == "alpha" for provider in entry.providers)
    }

    assert serving == {"chat_completions"}
