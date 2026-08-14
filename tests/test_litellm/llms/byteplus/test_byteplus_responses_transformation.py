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
