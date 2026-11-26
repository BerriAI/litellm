"""
Tests for JSON-based provider configuration system.
"""

import json
import os
import sys
from collections.abc import Callable
from typing import Final
from unittest.mock import patch

try:
    import pytest
except ImportError:
    # pytest not available, will run as standalone script
    pytest = None

# Add workspace to path
workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)


class TestJSONProviderLoader:
    """Test JSON provider loading and configuration"""

    def test_load_json_providers(self):
        """Test that JSON providers load correctly"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        # Verify publicai is loaded
        assert JSONProviderRegistry.exists("publicai")

        # Get publicai config
        publicai = JSONProviderRegistry.get("publicai")
        assert publicai is not None
        assert publicai.base_url == "https://api.publicai.co/v1"
        assert publicai.api_key_env == "PUBLICAI_API_KEY"
        assert publicai.api_base_env == "PUBLICAI_API_BASE"
        assert publicai.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_dynamic_config_generation(self):
        """Test dynamic config class creation"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Test API info resolution
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.publicai.co/v1"

        # Test with custom base
        api_base, api_key = config.get_openai_compatible_provider_info("https://custom.api.com", "test-key")
        assert api_base == "https://custom.api.com"
        assert api_key == "test-key"

    def test_parameter_mapping(self):
        """Test parameter mapping works"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Test parameter mapping
        optional_params = {}
        non_default_params = {"max_completion_tokens": 100, "temperature": 0.7}
        result = config.map_openai_params(non_default_params, optional_params, "gpt-4", False)

        # max_completion_tokens should be mapped to max_tokens
        assert "max_tokens" in result
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result

        # temperature should be passed through
        assert result["temperature"] == 0.7

    def test_supported_params(self):
        """Test that config returns supported params"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Get supported params
        supported = config.get_supported_openai_params("gpt-4")

        # Should have standard OpenAI params
        assert isinstance(supported, list)
        assert len(supported) > 0

    def test_tool_params_excluded_when_function_calling_not_supported(self):
        """Test that tool-related params are excluded for models that don't support
        function calling. Regression test for https://github.com/BerriAI/litellm/issues/21125
        """
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Mock supports_function_calling to return False
        with patch("litellm.utils.supports_function_calling", return_value=False):
            supported = config.get_supported_openai_params("some-model-without-fc")

        tool_params = [
            "tools",
            "tool_choice",
            "function_call",
            "functions",
            "parallel_tool_calls",
        ]
        for param in tool_params:
            assert param not in supported, (
                f"'{param}' should not be in supported params when function calling is not supported"
            )

        # Non-tool params should still be present
        assert "temperature" in supported
        assert "max_tokens" in supported
        assert "stop" in supported

    def test_tool_params_included_when_function_calling_supported(self):
        """Test that tool-related params are included for models that support function calling."""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("publicai")
        config_class = create_config_class(provider)
        config = config_class()

        # Mock supports_function_calling to return True
        with patch("litellm.utils.supports_function_calling", return_value=True):
            supported = config.get_supported_openai_params("some-model-with-fc")

        assert "tools" in supported
        assert "tool_choice" in supported

    def test_provider_resolution(self):
        """Test that provider resolution finds JSON providers"""
        from litellm.litellm_core_utils.get_llm_provider_logic import (
            get_llm_provider,
        )

        model, provider, api_key, api_base = get_llm_provider(
            model="publicai/gpt-4",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gpt-4"
        assert provider == "publicai"
        assert api_base == "https://api.publicai.co/v1"

    def test_provider_config_manager(self):
        """Test that ProviderConfigManager returns JSON-based configs"""
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(model="gpt-4", provider=LlmProviders.PUBLICAI)

        assert config is not None
        assert config.custom_llm_provider == "publicai"


class TestPinstripes:
    """Tests for Pinstripes JSON-configured provider"""

    def test_pinstripes_json_config_exists(self):
        """Test that pinstripes is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("pinstripes")

        pinstripes = JSONProviderRegistry.get("pinstripes")
        assert pinstripes is not None
        assert pinstripes.base_url == "https://pinstripes.io/v1"
        assert pinstripes.api_key_env == "PINSTRIPES_API_KEY"
        assert pinstripes.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_pinstripes_provider_resolution(self):
        """Test that provider resolution finds pinstripes and returns the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="pinstripes/ps/glm-4.5-air",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "ps/glm-4.5-air"
        assert provider == "pinstripes"
        assert api_base == "https://pinstripes.io/v1"

    def test_pinstripes_dynamic_config(self):
        """Test dynamic config class creation for pinstripes"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("pinstripes")
        config_class = create_config_class(provider)
        config = config_class()

        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://pinstripes.io/v1"

        api_base, api_key = config.get_openai_compatible_provider_info("https://custom.pinstripes.io/v1", "test-key")
        assert api_base == "https://custom.pinstripes.io/v1"
        assert api_key == "test-key"

    def test_pinstripes_parameter_mapping(self):
        """Test that max_completion_tokens is mapped to max_tokens for pinstripes"""
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("pinstripes")
        config_class = create_config_class(provider)
        config = config_class()

        optional_params = {}
        non_default_params = {"max_completion_tokens": 100, "temperature": 0.7}
        result = config.map_openai_params(non_default_params, optional_params, "ps/glm-4.5-air", False)

        assert "max_tokens" in result
        assert result["max_tokens"] == 100
        assert "max_completion_tokens" not in result
        assert result["temperature"] == 0.7


class TestDarkbloom:
    def test_darkbloom_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        darkbloom = JSONProviderRegistry.get("darkbloom")
        assert darkbloom is not None
        assert darkbloom.base_url == "https://api.darkbloom.dev/v1"
        assert darkbloom.api_key_env == "DARKBLOOM_API_KEY"
        assert darkbloom.api_base_env == "DARKBLOOM_API_BASE"
        assert darkbloom.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_darkbloom_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="darkbloom/gemma-4-26b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "gemma-4-26b"
        assert provider == "darkbloom"
        assert api_key is None
        assert api_base == "https://api.darkbloom.dev/v1"

    def test_darkbloom_dynamic_config(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("darkbloom")
        config_class = create_config_class(provider)
        config = config_class()

        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.darkbloom.dev/v1"

        api_base, api_key = config.get_openai_compatible_provider_info("https://custom.darkbloom.dev/v1", "test-key")
        assert api_base == "https://custom.darkbloom.dev/v1"
        assert api_key == "test-key"

    def test_darkbloom_complete_url_appends_endpoint(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        provider = JSONProviderRegistry.get("darkbloom")
        config_class = create_config_class(provider)
        config = config_class()

        url = config.get_complete_url(
            api_base="https://api.darkbloom.dev/v1",
            api_key="test-key",
            model="darkbloom/gemma-4-26b",
            optional_params={},
            litellm_params={},
            stream=True,
        )

        assert url == "https://api.darkbloom.dev/v1/chat/completions"

    def test_darkbloom_provider_config_manager(self):
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(model="gemma-4-26b", provider=LlmProviders.DARKBLOOM)

        assert config is not None
        assert config.custom_llm_provider == "darkbloom"


def _complete(model: str) -> object:
    import litellm

    return litellm.completion(model=model, messages=[{"role": "user", "content": "hi"}])


def _transcribe(model: str) -> object:
    import litellm

    return litellm.transcription(model=model, file=("clip.wav", b"RIFF0000WAVE"))


def _speak(model: str) -> object:
    import litellm

    return litellm.speech(model=model, input="hi", voice="alloy")


def _generate_image(model: str) -> object:
    import litellm

    return litellm.image_generation(model=model, prompt="a red square")


def _moderate(model: str) -> object:
    import asyncio

    import litellm

    return asyncio.run(litellm.amoderation(model=model, input="hi"))


def _create_realtime_client_secret(model: str) -> object:
    import asyncio

    import litellm

    return asyncio.run(litellm.acreate_realtime_client_secret(model=model))


def _create_realtime_transcription_session(model: str) -> object:
    import asyncio

    import litellm

    return asyncio.run(litellm.acreate_realtime_transcription_session(model=model))


@pytest.mark.parametrize("provider", ("resetdata", "parasail"))
@pytest.mark.parametrize(
    "call",
    (
        _complete,
        _transcribe,
        _speak,
        _generate_image,
        _moderate,
        _create_realtime_client_secret,
        _create_realtime_transcription_session,
    ),
    ids=(
        "chat",
        "transcription",
        "speech",
        "image",
        "moderation",
        "realtime_client_secret",
        "realtime_transcription_session",
    ),
)
def test_json_provider_without_its_key_fails_before_sending_the_openai_key(
    monkeypatch: pytest.MonkeyPatch, provider: str, call: Callable[[str], object]
):
    import respx

    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider_config: Final = JSONProviderRegistry.get(provider)
    assert provider_config is not None
    monkeypatch.delenv(provider_config.api_key_env, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-canary")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.route().respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match=provider_config.api_key_env):
            call(f"{provider}/some-model")

    assert route.call_count == 0


@pytest.mark.parametrize("provider", ("chutes", "nano-gpt"))
def test_json_text_completion_provider_without_its_key_fails_before_sending_the_openai_key(
    monkeypatch: pytest.MonkeyPatch, provider: str
):
    import respx

    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider_config: Final = JSONProviderRegistry.get(provider)
    assert provider_config is not None
    assert provider in litellm.openai_text_completion_compatible_providers
    monkeypatch.delenv(provider_config.api_key_env, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-canary")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.route().respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match=provider_config.api_key_env):
            litellm.text_completion(model=f"{provider}/some-model", prompt="hi")

    assert route.call_count == 0


_HEADER_AUTH: Final = {"Authorization": "Bearer sk-header"}
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "some-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
}


def _complete_with_headers(**header_kwargs: dict[str, str]) -> object:
    import litellm

    return litellm.completion(model="parasail/some-model", messages=[{"role": "user", "content": "hi"}], **header_kwargs)


def _text_complete_with_headers(**header_kwargs: dict[str, str]) -> object:
    import litellm

    return litellm.text_completion(model="chutes/some-model", prompt="hi", **header_kwargs)


def _generate_image_with_headers(**header_kwargs: dict[str, str]) -> object:
    import litellm

    return litellm.image_generation(model="parasail/some-model", prompt="a red square", **header_kwargs)


def _create_realtime_client_secret_with_headers(**header_kwargs: dict[str, str]) -> object:
    import asyncio

    import litellm

    return asyncio.run(litellm.acreate_realtime_client_secret(model="parasail/some-model", **header_kwargs))


def _create_realtime_transcription_session_with_headers(**header_kwargs: dict[str, str]) -> object:
    import asyncio

    import litellm

    return asyncio.run(litellm.acreate_realtime_transcription_session(model="parasail/some-model", **header_kwargs))


@pytest.mark.parametrize(
    ("call", "header_param"),
    (
        (_complete_with_headers, "extra_headers"),
        (_complete_with_headers, "headers"),
        (_text_complete_with_headers, "extra_headers"),
        (_generate_image_with_headers, "headers"),
        (_create_realtime_client_secret_with_headers, "extra_headers"),
        (_create_realtime_transcription_session_with_headers, "extra_headers"),
    ),
    ids=(
        "chat-extra_headers",
        "chat-headers",
        "text_completion",
        "image",
        "realtime_client_secret",
        "realtime_transcription_session",
    ),
)
@pytest.mark.parametrize("header_name", ("Authorization", "authorization"))
def test_json_provider_without_its_key_sends_the_callers_authorization_header(
    monkeypatch: pytest.MonkeyPatch, call: Callable[..., object], header_param: str, header_name: str
):
    import respx

    import litellm

    monkeypatch.delenv("PARASAIL_API_KEY", raising=False)
    monkeypatch.delenv("CHUTES_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-canary")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock() as upstream:
        route: Final = upstream.route().respond(200, json=_CHAT_COMPLETION)
        call(**{header_param: {header_name: "Bearer sk-header"}})

    assert route.call_count == 1
    assert route.calls.last.request.headers.get_list("authorization") == ["Bearer sk-header"]


def test_json_provider_chat_on_the_http_handler_sends_the_callers_authorization_header(
    monkeypatch: pytest.MonkeyPatch,
):
    import respx

    import litellm
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    monkeypatch.delenv("PARASAIL_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-canary")
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "true")

    with respx.mock() as upstream:
        route: Final = upstream.route().respond(200, json=_CHAT_COMPLETION)
        _complete_with_headers(extra_headers=_HEADER_AUTH, client=HTTPHandler())

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-header"


def _transcribe_with_headers(**header_kwargs: dict[str, str]) -> object:
    import litellm

    return litellm.transcription(model="parasail/some-model", file=("clip.wav", b"RIFF0000WAVE"), **header_kwargs)


def _speak_with_headers(**header_kwargs: dict[str, str]) -> object:
    import litellm

    return litellm.speech(model="parasail/some-model", input="hi", voice="alloy", **header_kwargs)


def _moderate_with_headers(**header_kwargs: dict[str, str]) -> object:
    import asyncio

    import litellm

    return asyncio.run(litellm.amoderation(model="parasail/some-model", input="hi", **header_kwargs))


@pytest.mark.parametrize(
    "call",
    (_transcribe_with_headers, _speak_with_headers, _moderate_with_headers),
    ids=("transcription", "speech", "moderation"),
)
def test_json_provider_header_the_route_never_forwards_does_not_replace_its_key(
    monkeypatch: pytest.MonkeyPatch, call: Callable[..., object]
):
    import respx

    import litellm

    monkeypatch.delenv("PARASAIL_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-canary")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.route().respond(200, json={})
        with pytest.raises(litellm.AuthenticationError, match="PARASAIL_API_KEY"):
            call(extra_headers=_HEADER_AUTH, headers=_HEADER_AUTH)

    assert route.call_count == 0


def test_json_provider_moderation_sends_its_own_env_key(monkeypatch: pytest.MonkeyPatch):
    import asyncio

    import respx

    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider_config: Final = JSONProviderRegistry.get("parasail")
    assert provider_config is not None
    monkeypatch.delenv(provider_config.api_base_env or "", raising=False)
    monkeypatch.setenv(provider_config.api_key_env, "sk-parasail-env")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-canary")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock() as upstream:
        route: Final = upstream.post(f"{provider_config.base_url}/moderations").respond(
            200, json={"id": "modr_1", "model": "some-model", "results": [{"flagged": False}]}
        )
        asyncio.run(litellm.amoderation(model="parasail/some-model", input="hi"))

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-parasail-env"


def test_json_provider_realtime_call_without_its_key_sends_the_callers_ephemeral_key(
    monkeypatch: pytest.MonkeyPatch,
):
    import asyncio

    import respx

    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider_config: Final = JSONProviderRegistry.get("parasail")
    assert provider_config is not None
    monkeypatch.delenv(provider_config.api_key_env, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-canary")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock() as upstream:
        route: Final = upstream.route().respond(201, text="v=0", headers={"Location": "/v1/realtime/calls/rtc_1"})
        asyncio.run(
            litellm.arealtime_calls(model="parasail/some-model", openai_ephemeral_key="ek_caller", sdp_body=b"v=0")
        )

    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer ek_caller"


class TestRightNow:
    MODEL = "rightnow/deepseek-v4-1-flash-1017"

    def test_rightnow_json_config_exists(self):
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        rightnow = JSONProviderRegistry.get("rightnow")
        assert rightnow is not None
        assert rightnow.base_url == "https://api.rightnowai.co/v1"
        assert rightnow.api_key_env == "RIGHTNOW_API_KEY"
        assert rightnow.api_base_env == "RIGHTNOW_API_BASE"
        assert rightnow.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_rightnow_provider_resolution(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model=self.MODEL, custom_llm_provider=None, api_base=None, api_key=None
        )

        assert model == "deepseek-v4-1-flash-1017"
        assert provider == "rightnow"
        assert api_key is None
        assert api_base == "https://api.rightnowai.co/v1"

    def test_rightnow_dynamic_config(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        config = create_config_class(JSONProviderRegistry.get("rightnow"))()

        api_base, _ = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.rightnowai.co/v1"

        api_base, api_key = config._get_openai_compatible_provider_info("https://custom.example/v1", "test-key")
        assert api_base == "https://custom.example/v1"
        assert api_key == "test-key"

    def test_rightnow_complete_url_appends_endpoint(self):
        from litellm.llms.openai_like.dynamic_config import create_config_class
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        config = create_config_class(JSONProviderRegistry.get("rightnow"))()
        url = config.get_complete_url(
            api_base="https://api.rightnowai.co/v1",
            api_key="test-key",
            model=self.MODEL,
            optional_params={},
            litellm_params={},
            stream=True,
        )

        assert url == "https://api.rightnowai.co/v1/chat/completions"

    def test_rightnow_provider_config_manager(self):
        from litellm import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(
            model="deepseek-v4-1-flash-1017", provider=LlmProviders.RIGHTNOW
        )

        assert config is not None
        assert config.custom_llm_provider == "rightnow"

    def test_rightnow_model_cost_map(self):
        with open(os.path.join(workspace_path, "model_prices_and_context_window.json")) as f:
            model_cost = json.load(f)

        # model: (input, output, cache read) per token, max input, max output, image input
        expected_models = {
            "rightnow/deepseek-v4-1-flash": (1.4e-07, 5.8e-07, 3e-08, 1048576, 65536, False),
            "rightnow/deepseek-v4-1-flash-1017": (1.5e-07, 6e-07, 1e-08, 1048576, 65536, True),
            "rightnow/glm-5-3": (1.3e-06, 4.2e-06, 1.3e-07, 1048576, 65536, False),
            "rightnow/glm-5-3-flash": (1.1e-07, 4.5e-07, 3e-08, 1048576, 65536, True),
            "rightnow/kimi-k3-1-pro": (1.6e-06, 8e-06, 1.6e-07, 1048576, 32768, True),
            "rightnow/nemotron-3-5-lightning-30b": (5e-08, 1.5e-07, 1e-08, 262144, 65536, False),
            "rightnow/ornith-1-5-35b": (1e-07, 4e-07, 1e-08, 262144, 65536, True),
            "rightnow/qwen3-8-27b": (1e-07, 4e-07, 1e-08, 262144, 32768, True),
        }
        # Every RightNow entry is pinned here, so a retired model left in the cost map,
        # or a new one added without its prices, reddens this test.
        assert {k for k, v in model_cost.items() if v.get("litellm_provider") == "rightnow"} == set(expected_models)
        for model, (input_cost, output_cost, cache_read, max_input, max_output, vision) in expected_models.items():
            entry = model_cost[model]
            assert entry["input_cost_per_token"] == input_cost
            assert entry["output_cost_per_token"] == output_cost
            assert entry["cache_read_input_token_cost"] == cache_read
            assert entry["supports_prompt_caching"] is True
            assert entry["max_input_tokens"] == max_input
            assert entry["max_output_tokens"] == max_output
            assert entry["max_tokens"] == max_output
            assert entry["supports_function_calling"] is True
            assert entry["supports_tool_choice"] is True
            assert entry["supports_response_schema"] is True
            assert entry["supports_reasoning"] is True
            assert entry.get("supports_vision", False) is vision

        with open(os.path.join(workspace_path, "litellm", "model_prices_and_context_window_backup.json")) as f:
            backup = json.load(f)
        for model in expected_models:
            assert backup[model] == model_cost[model]
