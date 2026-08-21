"""
Tests for Parallel AI credential and provider resolution.

Source: litellm/llms/parallel_ai/common_utils.py
"""

import pytest

import litellm
from litellm.llms.parallel_ai.common_utils import resolve_parallel_ai_credentials


class TestResolveParallelAICredentials:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_BASE", raising=False)
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")

        api_base, api_key = resolve_parallel_ai_credentials(api_base=None, api_key=None)
        assert api_base == "https://api.parallel.ai"
        assert api_key == "pk-test"

    def test_prefers_parallel_ai_key(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-primary")
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-fallback")

        _, api_key = resolve_parallel_ai_credentials(api_base=None, api_key=None)
        assert api_key == "pk-primary"

    def test_plaintext_provider_host_is_not_trusted(self, monkeypatch):
        """A host-only trust check would send the server key over plaintext HTTP."""
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-env")

        with pytest.raises(ValueError, match="Refusing to send"):
            resolve_parallel_ai_credentials(api_base="http://api.parallel.ai", api_key=None)

    def test_https_provider_host_is_trusted(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-env")

        api_base, api_key = resolve_parallel_ai_credentials(api_base="https://api.parallel.ai", api_key=None)
        assert api_base == "https://api.parallel.ai"
        assert api_key == "pk-env"

    def test_operator_plaintext_override_is_trusted(self, monkeypatch):
        """The operator's own PARALLEL_AI_API_BASE is trusted at the scheme it names."""
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-env")
        monkeypatch.setenv("PARALLEL_AI_API_BASE", "http://parallel-proxy.internal")

        _, api_key = resolve_parallel_ai_credentials(api_base="http://parallel-proxy.internal", api_key=None)
        assert api_key == "pk-env"

    def test_explicit_args_win(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-env")

        api_base, api_key = resolve_parallel_ai_credentials(api_base="https://proxy.example.com", api_key="pk-explicit")
        assert api_base == "https://proxy.example.com"
        assert api_key == "pk-explicit"

    def test_untrusted_api_base_refuses_server_key(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-server-secret")

        with pytest.raises(ValueError, match="Refusing to send"):
            resolve_parallel_ai_credentials(api_base="https://attacker.example.com", api_key=None)

    def test_operator_env_base_is_trusted(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-server-secret")
        monkeypatch.setenv("PARALLEL_AI_API_BASE", "https://proxy.internal.example.com")

        api_base, api_key = resolve_parallel_ai_credentials(api_base="https://proxy.internal.example.com", api_key=None)
        assert api_base == "https://proxy.internal.example.com"
        assert api_key == "pk-server-secret"

    def test_untrusted_base_without_server_key_returns_none(self):
        api_base, api_key = resolve_parallel_ai_credentials(api_base="https://proxy.example.com", api_key=None)
        assert api_base == "https://proxy.example.com"
        assert api_key is None


class TestParallelAIProviderWiring:
    def test_get_llm_provider_routes_parallel_ai(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")
        model, provider, api_key, api_base = litellm.get_llm_provider("parallel_ai/parallel")
        assert model == "parallel"
        assert provider == "parallel_ai"
        assert api_key == "pk-test"
        assert api_base == "https://api.parallel.ai"

    def test_get_llm_provider_detects_parallel_api_base(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")

        model, provider, api_key, api_base = litellm.get_llm_provider(
            model="parallel", api_base="https://api.parallel.ai"
        )
        assert provider == "parallel_ai"
        assert api_key == "pk-test"

    def test_validate_environment_reports_missing_key(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.delenv("PARALLEL_API_KEY", raising=False)

        result = litellm.validate_environment(model="parallel_ai/parallel")
        assert result["keys_in_environment"] is False
        assert "PARALLEL_AI_API_KEY" in result["missing_keys"]

    def test_validate_environment_accepts_either_key_name(self, monkeypatch):
        monkeypatch.delenv("PARALLEL_AI_API_KEY", raising=False)
        monkeypatch.setenv("PARALLEL_API_KEY", "pk-test")

        result = litellm.validate_environment(model="parallel_ai/parallel")
        assert result["keys_in_environment"] is True

    def test_api_base_detection_keeps_explicit_caller_key(self, monkeypatch):
        monkeypatch.setenv("PARALLEL_AI_API_KEY", "pk-env")

        model, provider, api_key, api_base = litellm.get_llm_provider(
            model="parallel", api_base="https://api.parallel.ai", api_key="pk-explicit"
        )
        assert provider == "parallel_ai"
        assert api_key == "pk-explicit"
