"""
Tests for ResetData provider configuration and integration.

ResetData is an Australian sovereign AI-inference provider with an
OpenAI-compatible API at https://api.resetdata.ai/api/v1. Model slugs are
in "provider/model" form (e.g. zai/glm-5.3), so full LiteLLM model ids are
three segments: resetdata/zai/glm-5.3.
"""

from typing import Any

import litellm
import pytest


class TestResetDataProviderConfig:
    """Test ResetData provider configuration"""

    def test_resetdata_in_provider_list(self):
        """Test that resetdata is in the provider list"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "RESETDATA")
        assert LlmProviders.RESETDATA.value == "resetdata"
        assert "resetdata" in litellm.provider_list

    def test_resetdata_json_config_exists(self):
        """Test that resetdata is configured in providers.json"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("resetdata")

        resetdata = JSONProviderRegistry.get("resetdata")
        assert resetdata is not None
        assert resetdata.base_url == "https://api.resetdata.ai/api/v1"
        assert resetdata.api_key_env == "RESETDATA_API_KEY"
        assert resetdata.api_base_env == "RESETDATA_API_BASE"
        assert resetdata.supported_endpoints == ["/v1/chat/completions"]

    def test_resetdata_serves_chat_endpoint(self):
        """The registry declares chat completions as the served endpoint."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert (
            JSONProviderRegistry.serves_endpoint(
                "resetdata", None, "/v1/chat/completions"
            )
            is True
        )
        # Only chat completions is claimed for the initial contribution.
        assert (
            JSONProviderRegistry.serves_endpoint("resetdata", None, "/v1/embeddings")
            is False
        )
        assert (
            JSONProviderRegistry.serves_endpoint("resetdata", None, "/v1/responses")
            is False
        )
        assert JSONProviderRegistry.supports_embeddings("resetdata") is False

    def test_resetdata_provider_resolution(self, monkeypatch: pytest.MonkeyPatch):
        """Test that provider resolution finds resetdata and the default base URL.

        ResetData model slugs contain a slash (provider/model form), so the
        full model id is three segments. Only the first segment (the LiteLLM
        provider prefix) must be stripped.
        """
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        # A developer machine with these exported would otherwise flip the
        # default-base-url assertion.
        monkeypatch.delenv("RESETDATA_API_KEY", raising=False)
        monkeypatch.delenv("RESETDATA_API_BASE", raising=False)

        model, provider, api_key, api_base = get_llm_provider(
            model="resetdata/zai/glm-5.3",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "zai/glm-5.3"
        assert provider == "resetdata"
        assert api_base == "https://api.resetdata.ai/api/v1"

    def test_resetdata_api_base_override(self, monkeypatch: pytest.MonkeyPatch):
        """Test that an explicit api_base / api_key overrides the default"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        monkeypatch.delenv("RESETDATA_API_KEY", raising=False)
        monkeypatch.delenv("RESETDATA_API_BASE", raising=False)

        model, provider, api_key, api_base = get_llm_provider(
            model="resetdata/zai/glm-5.3",
            custom_llm_provider=None,
            api_base="https://custom.example.com/v1",
            api_key="sk-test",
        )

        assert provider == "resetdata"
        assert api_base == "https://custom.example.com/v1"
        assert api_key == "sk-test"

    def test_resetdata_model_cost_map(self, monkeypatch: pytest.MonkeyPatch):
        """ResetData models are present in the packaged model cost map.

        The local packaged map is asserted (not the runtime litellm.model_cost,
        which may reflect the remote main-branch map) so the test is
        deterministic before and after the catalog entries merge upstream.
        """
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        local_cost: dict[str, dict[str, Any]] = litellm.get_model_cost_map(url="")

        assert "resetdata/zai/glm-5.3" in local_cost
        info = local_cost["resetdata/zai/glm-5.3"]
        assert info["litellm_provider"] == "resetdata"
        assert info["mode"] == "chat"

    def test_resetdata_router_config(self):
        """Test that resetdata can be used in Router configuration"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "resetdata-chat",
                    "litellm_params": {
                        "model": "resetdata/zai/glm-5.3",
                        "api_key": "test-key",
                    },
                }
            ]
        )

        deployments = router.registry().all()
        assert len(deployments) == 1
        assert deployments[0]["model_name"] == "resetdata-chat"

    def test_resetdata_model_modes(self, monkeypatch: pytest.MonkeyPatch):
        """All resetdata catalog entries are chat models from the resetdata provider."""
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        local_cost: dict[str, dict[str, Any]] = litellm.get_model_cost_map(url="")

        resetdata_models = [k for k in local_cost if k.startswith("resetdata/")]
        assert len(resetdata_models) > 0
        for key in resetdata_models:
            assert local_cost[key]["mode"] == "chat", key
            assert local_cost[key]["litellm_provider"] == "resetdata", key

    def test_resetdata_supported_endpoints_matrix(self):
        """The runtime-served backup matrix (GET /public/supported_endpoints) lists resetdata."""
        import json
        from pathlib import Path

        import litellm as _litellm

        backup_path = (
            Path(_litellm.__file__).parent / "provider_endpoints_support_backup.json"
        )
        matrix = json.loads(backup_path.read_text())

        assert "resetdata" in matrix["providers"]
        endpoints = matrix["providers"]["resetdata"]["endpoints"]
        assert endpoints["chat_completions"] is True
        # Only chat completions is claimed for the initial contribution:
        # ResetData documents /chat/completions as its core endpoint today.
        # Other endpoints can be enabled as they are validated.
        assert endpoints["embeddings"] is False
        assert endpoints["responses"] is False
