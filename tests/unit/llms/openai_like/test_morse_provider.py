import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://inference.trymorse.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_morse",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "z-ai/glm-5.3-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Morse", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
MORSE_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("morse/")))


def test_morse_is_a_registered_provider():
    assert litellm.LlmProviders.MORSE.value == "morse"
    assert "morse" in litellm.provider_list
    assert "morse" not in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_morse_provider_resolution_keeps_the_lab_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MORSE_API_KEY", "morse-test-key")
    monkeypatch.delenv("MORSE_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="morse/z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "z-ai/glm-5.3-flash"
    assert provider == "morse"
    assert api_key == "morse-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_morse_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MORSE_API_KEY", "morse-env-key")
    monkeypatch.setenv("MORSE_API_BASE", "https://morse.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="morse/z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "morse"
    assert api_base == "https://morse.env.example/v1"


@pytest.mark.parametrize("model", MORSE_MODELS)
def test_morse_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import supports_function_calling, supports_reasoning, supports_vision

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="morse",
    )

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert litellm.get_model_info(model)["litellm_provider"] == "morse"
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_vision(model) is entry["supports_vision"]


def test_morse_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling

    assert supports_function_calling("morse/a-lab/a-model-not-in-the-cost-map") is True


def test_morse_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    morse_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("morse/")}

    assert morse_entries
    assert tuple(sorted(morse_entries)) == MORSE_MODELS
    assert morse_entries == {name: backup[name] for name in morse_entries}


def test_morse_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    morse: Final = next(provider for provider in providers if provider["litellm_provider"] == "morse")

    assert morse["provider_display_name"] == "Morse"
    assert morse["default_model_placeholder"] in MORSE_MODELS
    assert {field["key"]: field["required"] for field in morse["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_morse_supported_endpoints_match_backup():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["morse"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["morse"]["endpoints"]

    assert backup == root
    assert root["chat_completions"] is True
    assert root["embeddings"] is False


def test_morse_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="morse/z-ai/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="morse-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer morse-test-key"
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "Hello from Morse"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_morse_without_a_key_never_sends_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", "openai-test-key")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="MORSE_API_KEY"):
            litellm.completion(
                model="morse/z-ai/glm-5.3-flash",
                messages=[{"role": "user", "content": "Say hello"}],
            )

    assert route.call_count == 0


def _call_morse_endpoint(endpoint: str, audio_path: Path) -> None:
    if endpoint == "speech":
        litellm.speech(model="morse/a-lab/a-voice-model", voice="alloy", input="Say hello")
    elif endpoint == "transcription":
        with audio_path.open("rb") as audio:
            litellm.transcription(model="morse/a-lab/a-transcription-model", file=audio)
    else:
        litellm.text_completion(model="morse/z-ai/glm-5.3-flash", prompt="Say hello")


@pytest.fixture
def openai_key_without_morse_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MORSE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", "openai-test-key")


@pytest.mark.usefixtures("openai_key_without_morse_key")
@pytest.mark.parametrize("endpoint", ["speech", "transcription", "text_completion"])
def test_morse_other_endpoints_never_send_the_openai_key(tmp_path: Path, endpoint: str):
    audio_path: Final = tmp_path / "hello.wav"
    audio_path.write_bytes(b"RIFF0000WAVEfmt ")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.route().respond(200, json={})
        with pytest.raises(Exception, match="(?i)morse"):
            _call_morse_endpoint(endpoint, audio_path)

    assert route.call_count == 0


@pytest.mark.usefixtures("openai_key_without_morse_key")
def test_morse_image_generation_never_sends_the_openai_key():
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.route().respond(200, json={})
        response: Final = litellm.image_generation(model="morse/a-lab/an-image-model", prompt="Say hello")

    assert route.call_count == 0
    assert response.data == []
