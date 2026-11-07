"""
Unit tests for the Baidu Qianfan chat config.

Qianfan is OpenAI-compatible; what this provider adds is the mapping of LiteLLM's `thinking` /
`reasoning_effort` onto Qianfan's `thinking: {"type": ...}` switch and `thinking_budget`, plus
provider discovery from the `qianfan/` prefix and the default base URL
"""

import pytest

import litellm
from litellm.llms.qianfan.chat.transformation import QIANFAN_API_BASE, QianfanChatConfig

MODEL = "deepseek-v4-pro"


def _map(**non_default_params: object) -> dict:
    return QianfanChatConfig().map_openai_params(
        non_default_params=dict(non_default_params), optional_params={}, model=MODEL, drop_params=False
    )


class TestThinkingMapping:
    @pytest.mark.parametrize("state", ["enabled", "disabled"])
    def test_thinking_type_is_forwarded_as_qianfan_switch(self, state: str):
        out = _map(thinking={"type": state})
        assert out["thinking"] == {"type": state}
        assert "thinking_budget" not in out

    def test_budget_tokens_becomes_thinking_budget_only_when_enabled(self):
        enabled = _map(thinking={"type": "enabled", "budget_tokens": 2048})
        assert enabled["thinking"] == {"type": "enabled"}
        assert enabled["thinking_budget"] == 2048

        disabled = _map(thinking={"type": "disabled", "budget_tokens": 2048})
        assert disabled["thinking"] == {"type": "disabled"}
        assert "thinking_budget" not in disabled

    def test_reasoning_effort_none_turns_thinking_off(self):
        out = _map(reasoning_effort="none")
        assert out["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in out

    @pytest.mark.parametrize("effort", ["low", "high", "max"])
    def test_other_reasoning_effort_passes_through_without_a_thinking_switch(self, effort: str):
        out = _map(reasoning_effort=effort)
        assert out["reasoning_effort"] == effort
        assert "thinking" not in out

    def test_bridged_reasoning_effort_object_is_unwrapped(self):
        out = _map(reasoning_effort={"effort": "max", "summary": "auto"})
        assert out["reasoning_effort"] == "max"

    def test_explicit_thinking_wins_over_reasoning_effort_none(self):
        out = _map(thinking={"type": "enabled"}, reasoning_effort="none")
        assert out["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in out

    def test_unknown_thinking_shape_is_dropped(self):
        out = _map(thinking={"type": "auto"}, temperature=0.2)
        assert "thinking" not in out
        assert out["temperature"] == 0.2

    def test_openai_params_still_map_through_the_base_class(self):
        out = _map(max_completion_tokens=64, temperature=0.5, tool_choice="required")
        assert out["max_completion_tokens"] == 64
        assert out["temperature"] == 0.5
        assert out["tool_choice"] == "required"


class TestSupportedParams:
    def test_thinking_and_reasoning_effort_are_supported(self):
        supported = QianfanChatConfig().get_supported_openai_params(MODEL)
        assert "thinking" in supported
        assert "reasoning_effort" in supported
        assert "tools" in supported
        assert "response_format" in supported


class TestProviderResolution:
    def test_prefix_resolves_to_qianfan_with_default_base(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QIANFAN_API_KEY", "test-key")
        monkeypatch.delenv("QIANFAN_API_BASE", raising=False)
        model, provider, api_key, api_base = litellm.get_llm_provider(model=f"qianfan/{MODEL}")
        assert (model, provider, api_key, api_base) == (MODEL, "qianfan", "test-key", QIANFAN_API_BASE)

    def test_default_base_is_recognised_as_qianfan(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QIANFAN_API_KEY", "test-key")
        _, provider, _, _ = litellm.get_llm_provider(model=MODEL, api_base=QIANFAN_API_BASE)
        assert provider == "qianfan"

    def test_env_base_overrides_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("QIANFAN_API_BASE", "https://example.invalid/v2")
        api_base, _ = QianfanChatConfig().get_openai_compatible_provider_info(api_base=None, api_key=None)
        assert api_base == "https://example.invalid/v2"

    def test_complete_url_appends_chat_completions_once(self):
        config = QianfanChatConfig()
        url = config.get_complete_url(None, None, MODEL, {}, {})
        assert url == f"{QIANFAN_API_BASE}/chat/completions"
        assert config.get_complete_url(url, None, MODEL, {}, {}) == url

    def test_provider_config_manager_returns_qianfan_config(self):
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(model=MODEL, provider=LlmProviders.QIANFAN)
        assert isinstance(config, QianfanChatConfig)
