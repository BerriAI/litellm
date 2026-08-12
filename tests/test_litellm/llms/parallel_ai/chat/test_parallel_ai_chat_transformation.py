"""
Tests for Parallel AI chat transformation (OpenAI-compatible /chat/completions).

Source: litellm/llms/parallel_ai/chat/transformation.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.parallel_ai.chat.transformation import ParallelAIChatConfig

MOCK_BASIS = [
    {
        "field": "answer",
        "citations": [
            {"url": "https://example.com/a", "excerpts": ["excerpt one"]},
            {"url": "https://example.com/b"},
        ],
        "reasoning": "compared both sources",
        "confidence": "high",
    }
]


class TestParallelAIChatConfig:
    def test_provider_info_defaults(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")

        api_base, api_key = ParallelAIChatConfig()._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.parallel.ai"
        assert api_key == "pk-test"

    def test_provider_info_prefers_parallel_ai_key(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-primary")
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-fallback")

        _, api_key = ParallelAIChatConfig()._get_openai_compatible_provider_info(None, None)
        assert api_key == "pk-primary"

    def test_provider_info_explicit_args_win(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-env")

        api_base, api_key = ParallelAIChatConfig()._get_openai_compatible_provider_info(
            "https://proxy.example.com", "pk-explicit"
        )
        assert api_base == "https://proxy.example.com"
        assert api_key == "pk-explicit"

    def test_supported_params_exclude_sampling_and_tools(self):
        supported = ParallelAIChatConfig().get_supported_openai_params(model="core")
        assert "response_format" in supported
        assert "stream" in supported
        assert "temperature" not in supported
        assert "tools" not in supported
        assert "tool_choice" not in supported

    def test_get_llm_provider_routes_parallel_ai(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")
        model, provider, api_key, api_base = litellm.get_llm_provider("parallel_ai/speed")
        assert model == "speed"
        assert provider == "parallel_ai"
        assert api_key == "pk-test"
        assert api_base == "https://api.parallel.ai"


class TestParallelAICredentialSafety:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    def test_untrusted_api_base_refuses_server_key(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-server-secret")

        with pytest.raises(ValueError, match="Refusing to send"):
            ParallelAIChatConfig()._get_openai_compatible_provider_info("https://attacker.example.com", None)

    def test_untrusted_api_base_with_explicit_key_is_allowed(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-server-secret")

        api_base, api_key = ParallelAIChatConfig()._get_openai_compatible_provider_info(
            "https://proxy.example.com", "pk-caller"
        )
        assert api_base == "https://proxy.example.com"
        assert api_key == "pk-caller"

    def test_operator_env_base_is_trusted(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-server-secret")
        monkeypatch.setenv("PARALLEL_AI_API_BASE", "https://proxy.internal.example.com")

        api_base, api_key = ParallelAIChatConfig()._get_openai_compatible_provider_info(
            "https://proxy.internal.example.com", None
        )
        assert api_base == "https://proxy.internal.example.com"
        assert api_key == "pk-server-secret"

    def test_untrusted_base_without_server_key_returns_none(self):
        api_base, api_key = ParallelAIChatConfig()._get_openai_compatible_provider_info(
            "https://proxy.example.com", None
        )
        assert api_base == "https://proxy.example.com"
        assert api_key is None


class TestParallelAIProviderWiring:
    def test_validate_environment_reports_missing_key(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

        result = litellm.validate_environment(model="parallel_ai/speed")
        assert result["keys_in_environment"] is False
        assert "PARALLEL_AI_API_KEY" in result["missing_keys"]

    def test_validate_environment_accepts_either_key_name(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")

        result = litellm.validate_environment(model="parallel_ai/speed")
        assert result["keys_in_environment"] is True

    def test_get_llm_provider_detects_parallel_api_base(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")

        model, provider, api_key, api_base = litellm.get_llm_provider(model="speed", api_base="https://api.parallel.ai")
        assert provider == "parallel_ai"
        assert api_key == "pk-test"

    def test_provider_config_manager_returns_chat_config(self):
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        config = ProviderConfigManager.get_provider_chat_config(model="speed", provider=LlmProviders.PARALLEL_AI)
        assert isinstance(config, ParallelAIChatConfig)


class TestParallelAICompletionFlow:
    @pytest.mark.respx()
    def test_completion_end_to_end_preserves_basis(self, respx_mock, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")
        respx_mock.post("https://api.parallel.ai/chat/completions").respond(
            json={
                "id": "chatcmpl-789",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "core",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "grounded answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
                "basis": MOCK_BASIS,
            }
        )

        response = litellm.completion(
            model="parallel_ai/core",
            messages=[{"role": "user", "content": "question"}],
        )

        assert response.choices[0].message.content == "grounded answer"
        assert getattr(response, "basis") == MOCK_BASIS
