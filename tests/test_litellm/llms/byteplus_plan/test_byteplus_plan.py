from unittest.mock import MagicMock, patch

from openai import OpenAI

from litellm import completion
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.byteplus_plan.chat.transformation import BytePlusPlanChatConfig
from litellm.types.utils import ModelResponse
from litellm.utils import get_optional_params


class TestBytePlusPlanConfig:
    def test_inherits_volcengine_config(self):
        from litellm.llms.volcengine.chat.transformation import VolcEngineChatConfig

        assert issubclass(BytePlusPlanChatConfig, VolcEngineChatConfig)

    def test_get_supported_openai_params(self):
        config = BytePlusPlanChatConfig()
        params = config.get_supported_openai_params(model="kimi-k2.5")
        assert "thinking" in params
        assert "tools" in params
        assert "stream" in params
        assert "temperature" in params

    def test_thinking_parameter_handling(self):
        config = BytePlusPlanChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled"}},
            optional_params={},
            model="kimi-k2.5",
            drop_params=False,
        )
        assert result == {"extra_body": {"thinking": {"type": "enabled"}}}


class TestBytePlusPlanProviderResolution:
    def test_provider_resolution(self):
        model, provider, key, api_base = get_llm_provider(
            model="byteplus_plan/kimi-k2.5",
            api_base=None,
            api_key=None,
        )
        assert model == "kimi-k2.5"
        assert provider == "byteplus_plan"
        assert api_base == "https://ark.ap-southeast.bytepluses.com/api/coding/v3"

    def test_shares_api_key_with_byteplus(self):
        """byteplus_plan should use BYTEPLUS_API_KEY"""
        import os

        with patch.dict(os.environ, {"BYTEPLUS_API_KEY": "bp-shared-key"}, clear=False):
            _, _, key, _ = get_llm_provider(
                model="byteplus_plan/kimi-k2.5",
                api_base=None,
                api_key=None,
            )
            assert key == "bp-shared-key"

    def test_get_optional_params(self):
        params = get_optional_params(
            model="kimi-k2.5",
            custom_llm_provider="byteplus_plan",
            thinking={"type": "enabled"},
            drop_params=False,
        )
        assert "thinking" in params["extra_body"]

    def test_e2e_completion(self):
        client = OpenAI(api_key="test_api_key")
        mock_raw_response = MagicMock()
        mock_raw_response.headers = {
            "x-request-id": "123",
            "openai-organization": "org-123",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        mock_raw_response.parse.return_value = ModelResponse()

        with patch.object(
            client.chat.completions.with_raw_response, "create", mock_raw_response
        ) as mock_create:
            completion(
                model="byteplus_plan/kimi-k2.5",
                messages=[{"role": "user", "content": "hello"}],
                client=client,
            )
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "kimi-k2.5"
