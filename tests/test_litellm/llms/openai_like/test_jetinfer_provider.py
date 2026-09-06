"""
Tests for JetInfer provider configuration and integration.
"""

import pytest

import litellm


@pytest.fixture
def local_model_cost(monkeypatch):
    """Assert against the cost map in this repo, not the published one.

    litellm.model_cost is populated at import time and by default comes from the cost
    map published on GitHub. A model introduced by this PR is absent from that map
    until the PR ships, so a test reading litellm.model_cost directly would only start
    passing after merge - which is the wrong way round for the test that is supposed to
    gate the merge.

    The env var has to be set *and* the map rebuilt, because the import-time read has
    already happened by the time any fixture runs. The original map is restored so this
    does not leak into tests that run afterwards in the same session.
    """
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    original = litellm.model_cost
    litellm.model_cost = litellm.get_model_cost_map(url="")
    try:
        yield litellm.model_cost
    finally:
        litellm.model_cost = original


class TestJetInferProviderConfig:
    """Test JetInfer provider configuration"""

    def test_jetinfer_in_provider_list(self):
        """Test that jetinfer is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "JETINFER")
        assert LlmProviders.JETINFER.value == "jetinfer"
        assert "jetinfer" in litellm.provider_list

    def test_jetinfer_json_config_exists(self):
        """Test that jetinfer is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("jetinfer")

        jetinfer = JSONProviderRegistry.get("jetinfer")
        assert jetinfer is not None
        assert jetinfer.base_url == "https://api.jetinfer.com/v1"
        assert jetinfer.api_key_env == "JETINFER_API_KEY"
        assert jetinfer.api_base_env == "JETINFER_API_BASE"

    def test_jetinfer_declares_no_param_mappings(self):
        """JetInfer accepts max_completion_tokens directly.

        Most JSON-configured providers map max_completion_tokens -> max_tokens because
        their upstream only accepts the legacy field. JetInfer's gateway handles both,
        so the absence of a mapping is intentional rather than an omission - this test
        exists so that nobody 'fixes' it by adding one.
        """
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        jetinfer = JSONProviderRegistry.get("jetinfer")
        assert not jetinfer.param_mappings

    def test_jetinfer_provider_resolution(self):
        """Test that provider resolution finds jetinfer and the default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="jetinfer/qwen3.8-27b",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "qwen3.8-27b"
        assert provider == "jetinfer"
        assert api_base == "https://api.jetinfer.com/v1"

    def test_jetinfer_api_base_override(self):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="jetinfer/qwen3.8-27b",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "jetinfer"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_jetinfer_model_cost_map(self, local_model_cost):
        """Test that the jetinfer model is present in the model cost map"""
        model_cost = local_model_cost

        assert "jetinfer/qwen3.8-27b" in model_cost
        info = model_cost["jetinfer/qwen3.8-27b"]
        assert info["litellm_provider"] == "jetinfer"
        assert info["mode"] == "chat"
        assert info["max_input_tokens"] == 262144
        assert info["max_output_tokens"] == 65536
        assert info["supports_function_calling"] is True
        assert info["supports_response_schema"] is True

    def test_jetinfer_cached_input_is_cheaper_than_fresh(self, local_model_cost):
        """A cached prompt token must cost less than a fresh one.

        JetInfer's cached-input rate is the reason it is worth routing agent traffic
        here, and cost estimates across the ecosystem read this file. A transposed or
        dropped digit would silently overcharge every caller, so assert the relationship
        rather than only the literal values.
        """
        info = local_model_cost["jetinfer/qwen3.8-27b"]

        assert info["input_cost_per_token"] == 3.4e-07
        assert info["output_cost_per_token"] == 2.55e-06
        assert info["cache_read_input_token_cost"] == 3.4e-08
        assert info["cache_read_input_token_cost"] < info["input_cost_per_token"]
        assert info["supports_prompt_caching"] is True

    def test_jetinfer_router_config(self):
        """Test that jetinfer can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "jetinfer-chat",
                    "litellm_params": {
                        "model": "jetinfer/qwen3.8-27b",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "jetinfer-chat"

    def test_jetinfer_supported_endpoints_matrix(self):
        """The runtime-served backup matrix (GET /public/supported_endpoints) lists jetinfer."""
        import json
        from pathlib import Path

        import litellm as _litellm

        backup_path = (
            Path(_litellm.__file__).parent / "provider_endpoints_support_backup.json"
        )
        matrix = json.loads(backup_path.read_text())

        assert "jetinfer" in matrix["providers"]
        endpoints = matrix["providers"]["jetinfer"]["endpoints"]
        assert endpoints["chat_completions"] is True
        # responses is advertised false: JetInfer serves /v1/chat/completions and
        # /v1/completions, not OpenAI's /v1/responses API.
        assert endpoints["responses"] is False
        assert endpoints["embeddings"] is False
