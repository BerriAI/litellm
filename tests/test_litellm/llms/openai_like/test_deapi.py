"""
Tests for deAPI provider configuration and integration.

deAPI is a non-chat OpenAI-compatible provider serving embeddings,
image generation/edits, and audio (TTS + transcription).
"""

import os
import sys
from unittest.mock import MagicMock, patch

try:
    import pytest
except ImportError:
    pytest = None

workspace_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, workspace_path)

import litellm


class TestDeAPIProviderConfig:
    """Test deAPI provider configuration (mock-only, no live calls)."""

    def test_deapi_json_config_exists(self):
        """deAPI is registered in providers.json with the expected base_url/env."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.exists("deapi")

        cfg = JSONProviderRegistry.get("deapi")
        assert cfg is not None
        assert cfg.base_url == "https://oai.deapi.ai/v1"
        assert cfg.api_key_env == "DEAPI_API_KEY"
        assert cfg.api_base_env == "DEAPI_API_BASE"

    def test_deapi_supported_endpoints(self):
        """supported_endpoints declares the non-chat endpoints deAPI serves."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        cfg = JSONProviderRegistry.get("deapi")
        endpoints = set(cfg.supported_endpoints)

        assert "/v1/embeddings" in endpoints
        assert "/v1/images/generations" in endpoints
        assert "/v1/images/edits" in endpoints
        assert "/v1/audio/speech" in endpoints
        assert "/v1/audio/transcriptions" in endpoints
        # deAPI does not serve chat completions
        assert "/v1/chat/completions" not in endpoints
        assert "/v1/responses" not in endpoints

    def test_deapi_provider_resolution(self):
        """get_llm_provider resolves deapi/<model> to base_url + provider slug."""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="deapi/Bge_M3_FP16",
            custom_llm_provider=None,
            api_base=None,
            api_key=None,
        )

        assert model == "Bge_M3_FP16"
        assert provider == "deapi"
        assert api_base == "https://oai.deapi.ai/v1"

    def test_deapi_api_key_env_resolution(self):
        """DEAPI_API_KEY env var is read when no explicit key is passed."""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        with patch.dict(os.environ, {"DEAPI_API_KEY": "dpn-sk-test-key"}):
            _, _, api_key, _ = get_llm_provider(
                model="deapi/Bge_M3_FP16",
                custom_llm_provider=None,
                api_base=None,
                api_key=None,
            )
            assert api_key == "dpn-sk-test-key"

    def test_deapi_does_not_advertise_responses_api(self):
        """JSON registry should NOT report Responses API support for deAPI."""
        from litellm.llms.openai_like.json_loader import JSONProviderRegistry

        assert JSONProviderRegistry.supports_responses_api("deapi") is False


if __name__ == "__main__":
    test = TestDeAPIProviderConfig()
    test.test_deapi_json_config_exists()
    test.test_deapi_supported_endpoints()
    test.test_deapi_provider_resolution()
    test.test_deapi_api_key_env_resolution()
    test.test_deapi_does_not_advertise_responses_api()
    print("All config tests passed.")
