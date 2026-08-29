"""
Apodex Anthropic Messages transformation.

Apodex implements the Anthropic protocol natively at POST /v1/messages, but only
serves the core models there. The Deep Research tiers must keep working on the
same route through LiteLLM's translation instead of being handed to a path that
would reject them.
"""

import pytest

import litellm
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

CORE_MODELS = ("apodex-1.1", "apodex-1.1-mini")
DEEP_RESEARCH_MODELS = (
    "apodex-1-1-deep-research",
    "apodex-1-1-deep-solve",
    "apodex-1-1-deep-discover",
)


@pytest.fixture(autouse=True)
def _apodex_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APODEX_API_KEY", "sk-apodex-test")
    monkeypatch.delenv("APODEX_API_BASE", raising=False)
    yield


def _messages_config(model: str):
    return ProviderConfigManager.get_provider_anthropic_messages_config(model=model, provider=LlmProviders.APODEX)


def _complete_url() -> str:
    """Resolve the endpoint the way the handler does: validate first, then build the URL.

    validate_anthropic_messages_environment returns the api_base the handler feeds
    into get_complete_url, so the two steps have to run in that order.
    """
    config = _messages_config("apodex-1.1")
    assert config is not None
    _, api_base = config.validate_anthropic_messages_environment(
        headers={}, model="apodex-1.1", messages=[], optional_params={}, litellm_params={}
    )
    return config.get_complete_url(
        api_base=api_base, api_key=None, model="apodex-1.1", optional_params={}, litellm_params={}
    )


class TestNativePassthroughRouting:
    @pytest.mark.parametrize("model", CORE_MODELS)
    def test_core_models_get_the_native_config(self, model: str):
        config = _messages_config(model)
        assert config is not None
        assert type(config).__name__ == "ApodexAnthropicMessagesConfig"
        assert config.custom_llm_provider == "apodex"
        assert config.should_strip_billing_metadata() is True

    @pytest.mark.parametrize("model", DEEP_RESEARCH_MODELS)
    def test_deep_research_models_fall_back_to_translation(self, model: str):
        """No native config means LiteLLM uses a protocol translation instead of
        forwarding to a path Apodex does not serve for these tiers."""
        assert _messages_config(model) is None

    @pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
    def test_deep_research_translation_uses_responses_api(self, monkeypatch: pytest.MonkeyPatch, stream: bool):
        from litellm.llms.anthropic.experimental_pass_through.messages import handler

        captured: dict = {}

        class ResponsesRouteSelected(Exception):
            pass

        def capture_responses_translation(**kwargs):
            captured.update(kwargs)
            raise ResponsesRouteSelected

        def reject_chat_translation(**kwargs):
            pytest.fail("Apodex Deep Research messages must not route through chat completions")

        monkeypatch.setattr(litellm, "responses", capture_responses_translation)
        monkeypatch.setattr(litellm, "completion", reject_chat_translation)

        with pytest.raises(ResponsesRouteSelected):
            handler.anthropic_messages_handler(
                max_tokens=256,
                messages=[{"role": "user", "content": "hi"}],
                model="apodex/apodex-1-1-deep-research",
                custom_llm_provider="apodex",
                stream=stream,
            )

        assert captured["model"] == "apodex-1-1-deep-research"
        assert captured["custom_llm_provider"] == "apodex"
        assert captured.get("stream", False) is stream


class TestNativePassthroughRequest:
    def test_url_targets_the_native_messages_path(self):
        assert _complete_url() == "https://api.apodex.ai/v1/messages"

    def test_url_honours_an_api_base_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("APODEX_API_BASE", "https://env.apodex.test/v1")
        assert _complete_url() == "https://env.apodex.test/v1/messages"

    def test_headers_use_the_provider_api_key(self):
        config = _messages_config("apodex-1.1")
        assert config is not None
        headers, _ = config.validate_anthropic_messages_environment(
            headers={}, model="apodex-1.1", messages=[], optional_params={}, litellm_params={}
        )
        assert headers["authorization"] == "Bearer sk-apodex-test"
        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["content-type"] == "application/json"

    def test_caller_supplied_auth_header_is_not_overwritten(self):
        config = _messages_config("apodex-1.1")
        assert config is not None
        headers, _ = config.validate_anthropic_messages_environment(
            headers={"x-api-key": "sk-caller"},
            model="apodex-1.1",
            messages=[],
            optional_params={},
            litellm_params={},
        )
        assert headers["x-api-key"] == "sk-caller"
        assert "authorization" not in headers
