import pytest

from litellm.llms.byteplus.responses.transformation import BytePlusResponsesAPIConfig
from litellm.types.utils import LlmProviders


class TestBytePlusResponsesAPIConfig:
    def test_custom_llm_provider(self):
        config = BytePlusResponsesAPIConfig()
        assert config.custom_llm_provider == LlmProviders.BYTEPLUS

    def test_get_complete_url(self):
        config = BytePlusResponsesAPIConfig()
        url = config.get_complete_url(
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            litellm_params={},
        )
        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/responses"

    def test_validate_environment_missing_key(self, monkeypatch):
        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        config = BytePlusResponsesAPIConfig()
        with pytest.raises(ValueError, match="BytePlus API key is required"):
            config.validate_environment(headers={}, model="byteplus/seed-2-0-lite", litellm_params=None)

    def test_validate_environment_success(self, monkeypatch):
        monkeypatch.setenv("BYTEPLUS_API_KEY", "test-key")
        config = BytePlusResponsesAPIConfig()
        headers = config.validate_environment(headers={}, model="byteplus/seed-2-0-lite", litellm_params=None)
        assert headers.get("Authorization") == "Bearer test-key"

    def test_get_error_class(self):
        config = BytePlusResponsesAPIConfig()
        err = config.get_error_class("responses error", 500, headers={"content-type": "application/json"})
        assert err.status_code == 500
        assert "responses error" in err.message

    def test_get_complete_url_variations(self):
        config = BytePlusResponsesAPIConfig()
        url1 = config.get_complete_url("https://custom.com/responses", litellm_params={})
        assert url1 == "https://custom.com/responses"

        url2 = config.get_complete_url("https://custom.com", litellm_params={})
        assert url2 == "https://custom.com/api/v3/responses"

    def test_provider_config_manager_responses(self):
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_responses_api_config(
            model="byteplus/seed-2-0-lite",
            provider=LlmProviders.BYTEPLUS,
        )
        assert isinstance(cfg, BytePlusResponsesAPIConfig)
