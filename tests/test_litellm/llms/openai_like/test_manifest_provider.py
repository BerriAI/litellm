"""
Tests for Manifest provider configuration and integration.

Manifest is a JSON-configured OpenAI-compatible gateway that resolves the
served model server-side, so callers always target the single ``auto`` model.
"""

import litellm


class TestManifestProviderConfig:
    """Test Manifest provider configuration"""

    def test_manifest_in_provider_list(self):
        """manifest is registered as a first-class provider"""
        from litellm import LlmProviders

        assert hasattr(LlmProviders, "MANIFEST")
        assert LlmProviders.MANIFEST.value == "manifest"
        assert "manifest" in litellm.provider_list

    def test_manifest_json_config(self):
        """providers.json carries the gateway endpoint and env vars"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("manifest")

        manifest = JSONProviderRegistry.get("manifest")
        assert manifest is not None
        assert manifest.base_url == "https://app.manifest.build/v1"
        assert manifest.api_key_env == "MANIFEST_API_KEY"
        assert manifest.api_base_env == "MANIFEST_API_BASE"
        assert manifest.param_mappings.get("max_completion_tokens") == "max_tokens"

    def test_manifest_supports_responses_api(self):
        """Manifest declares /v1/responses, so the Responses API is wired"""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("manifest") is True

    def test_manifest_provider_resolution(self):
        """auto model resolves to the manifest provider and default base URL"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="manifest/auto",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "auto"
        assert provider == "manifest"
        assert api_base == "https://app.manifest.build/v1"

    def test_manifest_api_base_override(self):
        """explicit api_base / api_key win over the defaults (self-hosted Manifest)"""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="manifest/auto",
            custom_llm_provider=None,
            api_base="https://manifest.internal.example.com/v1",
            api_key="mnfst_test",
        )

        assert provider == "manifest"
        assert api_base == "https://manifest.internal.example.com/v1"
        assert api_key == "mnfst_test"

    def test_manifest_auto_in_model_cost_map(self):
        """the single auto model is present and priced as a passthrough gateway"""
        model_cost = litellm.model_cost

        assert "manifest/auto" in model_cost
        info = model_cost["manifest/auto"]
        assert info["litellm_provider"] == "manifest"
        assert info["mode"] == "chat"
        assert info["input_cost_per_token"] == 0
        assert info["output_cost_per_token"] == 0
        assert info["supports_function_calling"] is True

    def test_manifest_router_config(self):
        """manifest/auto is usable from Router model_list"""
        from litellm import Router

        router = Router(
            model_list=[
                {
                    "model_name": "manifest-auto",
                    "litellm_params": {
                        "model": "manifest/auto",
                        "api_key": "mnfst_test",
                    },
                }
            ]
        )

        assert len(router.model_list) == 1
        assert router.model_list[0]["model_name"] == "manifest-auto"

    def test_manifest_supported_endpoints_matrix(self):
        """the runtime-served backup matrix advertises chat + responses, not messages"""
        import json
        from pathlib import Path

        import litellm as _litellm

        backup_path = Path(_litellm.__file__).parent / "provider_endpoints_support_backup.json"
        matrix = json.loads(backup_path.read_text())

        assert "manifest" in matrix["providers"]
        endpoints = matrix["providers"]["manifest"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert endpoints["responses"] is True
        assert endpoints["messages"] is False
