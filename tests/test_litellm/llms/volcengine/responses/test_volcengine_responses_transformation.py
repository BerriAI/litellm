"""
Tests for Volcengine Responses API transformation config.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import litellm
from litellm.llms.volcengine.responses.transformation import (
    VolcEngineResponsesAPIConfig,
)
from litellm.llms.volcengine.common_utils import VolcEngineError
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestVolcengineResponsesAPIConfig:
    """Unit tests for Volcengine Responses API configuration."""

    def setup_method(self):
        self.config = VolcEngineResponsesAPIConfig()

    def test_provider_registration(self):
        """ProviderConfigManager should return Volcengine config."""
        cfg = ProviderConfigManager.get_provider_responses_api_config(
            provider=LlmProviders.VOLCENGINE,
            model="volcengine/doubao-seed-1.6",
        )
        assert isinstance(cfg, VolcEngineResponsesAPIConfig)
        assert cfg.custom_llm_provider == LlmProviders.VOLCENGINE

    def test_get_complete_url_variants(self):
        """Ensure Volcengine endpoint construction handles different bases."""
        default_url = self.config.get_complete_url(api_base=None, litellm_params={})
        assert (
            default_url == "https://ark.cn-beijing.volces.com/api/v3/responses"
        ), default_url

        url_with_api = self.config.get_complete_url(
            api_base="https://ark.cn-beijing.volces.com/api/v3", litellm_params={}
        )
        assert (
            url_with_api == "https://ark.cn-beijing.volces.com/api/v3/responses"
        ), url_with_api

        custom_base = self.config.get_complete_url(
            api_base="https://custom.example.com/", litellm_params={}
        )
        assert (
            custom_base == "https://custom.example.com/api/v3/responses"
        ), custom_base

        already_full = self.config.get_complete_url(
            api_base="https://foo/api/v3/responses", litellm_params={}
        )
        assert already_full == "https://foo/api/v3/responses"

    def test_validate_environment_with_api_key(self):
        """Headers should include Volcengine auth info when api_key is provided."""
        headers = {"User-Agent": "pytest"}
        litellm_params = GenericLiteLLMParams(api_key="test-volc-key")

        result = self.config.validate_environment(
            headers=headers,
            model="volcengine/doubao",
            litellm_params=litellm_params,
        )

        assert result["Authorization"] == "Bearer test-volc-key"
        assert result["Content-Type"] == "application/json"
        assert result["User-Agent"] == "pytest"

    def test_validate_environment_env_fallback(self, monkeypatch):
        """Fallback to VOLCENGINE_API_KEY environment variable."""
        monkeypatch.setenv("VOLCENGINE_API_KEY", "env-key")
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None, raising=False)
        result = self.config.validate_environment(
            headers={},
            model="volcengine/doubao",
            litellm_params=GenericLiteLLMParams(),
        )
        assert result["Authorization"] == "Bearer env-key"
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)

    def test_validate_environment_missing_key(self, monkeypatch):
        """Error when no API key is available."""
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.setattr(litellm, "api_key", None, raising=False)
        with pytest.raises(VolcEngineError):
            self.config.validate_environment(
                headers={},
                model="volcengine/doubao",
                litellm_params=GenericLiteLLMParams(),
            )

    def test_validate_environment_with_global_litellm_key(self, monkeypatch):
        """使用 litellm.api_key 作为默认秘钥。"""
        monkeypatch.setattr(litellm, "api_key", "global-key", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
        result = self.config.validate_environment(
            headers={},
            model="volcengine/doubao",
            litellm_params=GenericLiteLLMParams(),
        )
        assert result["Authorization"] == "Bearer global-key"

    def test_validate_environment_with_ark_key(self, monkeypatch):
        """优先回退到 ARK_API_KEY。"""
        monkeypatch.delenv("VOLCENGINE_API_KEY", raising=False)
        monkeypatch.setenv("ARK_API_KEY", "ark-key")
        monkeypatch.setattr(litellm, "api_key", None, raising=False)
        result = self.config.validate_environment(
            headers={},
            model="volcengine/doubao",
            litellm_params=GenericLiteLLMParams(),
        )
        assert result["Authorization"] == "Bearer ark-key"
        monkeypatch.delenv("ARK_API_KEY", raising=False)

    def test_transform_response_api_response_success(self):
        """Successful responses should be converted to ResponsesAPIResponse."""
        payload = {
            "id": "resp_test",
            "object": "response",
            "created_at": 1234567890,
            "status": "completed",
            "model": "volcengine/doubao",
            "output": [],
            "parallel_tool_calls": False,
            "metadata": {},
        }
        raw_response = MagicMock()
        raw_response.json.return_value = payload
        raw_response.text = json.dumps(payload)
        raw_response.headers = {}
        raw_response.status_code = 200
        logging_obj = MagicMock()

        result = self.config.transform_response_api_response(
            model="volcengine/doubao",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert isinstance(result, ResponsesAPIResponse)
        logging_obj.post_call.assert_called()

    def test_transform_response_api_response_error(self):
        """Errors from super class should surface as VolcEngineError."""
        raw_response = MagicMock()
        raw_response.json.side_effect = Exception("boom")
        raw_response.text = "bad response"
        raw_response.headers = {}
        raw_response.status_code = 500
        logging_obj = MagicMock()

        with pytest.raises(VolcEngineError):
            self.config.transform_response_api_response(
                model="volcengine/doubao",
                raw_response=raw_response,
                logging_obj=logging_obj,
            )
