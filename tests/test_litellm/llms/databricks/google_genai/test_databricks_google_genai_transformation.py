import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.databricks.google_genai.transformation import (
    DatabricksGoogleGenAIConfig,
)
from litellm.utils import ProviderConfigManager

HOST = "https://my-ws.cloud.databricks.com"
MODEL = "databricks/databricks-gemini-2-5-pro"


class TestProviderRegistration:
    def test_databricks_gemini_uses_native_config(self):
        config = ProviderConfigManager.get_provider_google_genai_generate_content_config(
            model=MODEL,
            provider=litellm.LlmProviders.DATABRICKS,
        )
        assert isinstance(config, DatabricksGoogleGenAIConfig)

    def test_databricks_non_gemini_returns_none(self):
        config = ProviderConfigManager.get_provider_google_genai_generate_content_config(
            model="databricks/databricks-claude-3-7-sonnet",
            provider=litellm.LlmProviders.DATABRICKS,
        )
        assert config is None


class TestUrlBuilding:
    def _url(self, api_base, stream=False):
        headers, url = DatabricksGoogleGenAIConfig().sync_get_auth_token_and_url(
            api_base=api_base,
            model=MODEL,
            litellm_params={"api_key": "dapi-test", "api_base": api_base},
            stream=stream,
        )
        return url

    def test_bare_host_non_stream(self):
        assert self._url(HOST) == (
            f"{HOST}/ai-gateway/gemini/v1beta/models/"
            "databricks-gemini-2-5-pro:generateContent"
        )

    def test_stream_url_has_sse(self):
        assert self._url(HOST, stream=True) == (
            f"{HOST}/ai-gateway/gemini/v1beta/models/"
            "databricks-gemini-2-5-pro:streamGenerateContent?alt=sse"
        )

    def test_serving_endpoints_base_rewritten_to_gateway(self):
        assert self._url(f"{HOST}/serving-endpoints") == (
            f"{HOST}/ai-gateway/gemini/v1beta/models/"
            "databricks-gemini-2-5-pro:generateContent"
        )


class TestAuth:
    def test_pat_sets_bearer(self):
        config = DatabricksGoogleGenAIConfig()
        headers = config.validate_environment(
            api_key="dapi-secret",
            headers=None,
            model=MODEL,
            litellm_params={"api_base": HOST},
        )
        assert headers["Authorization"] == "Bearer dapi-secret"
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers

    def test_inherits_native_gemini_transforms(self):
        # Sanity: the native param mapping is inherited from GoogleGenAIConfig.
        config = DatabricksGoogleGenAIConfig()
        params = config.get_supported_generate_content_optional_params(MODEL)
        assert "temperature" in params
        assert "tools" in params
