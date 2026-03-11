from unittest.mock import MagicMock, patch

from openai import OpenAI

from litellm import completion
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.volcengine_plan.chat.transformation import VolcEnginePlanChatConfig
from litellm.types.utils import ModelResponse
from litellm.utils import get_optional_params


class TestVolcEnginePlanConfig:
    def test_inherits_volcengine_config(self):
        from litellm.llms.volcengine.chat.transformation import VolcEngineChatConfig

        assert issubclass(VolcEnginePlanChatConfig, VolcEngineChatConfig)

    def test_get_supported_openai_params(self):
        config = VolcEnginePlanChatConfig()
        params = config.get_supported_openai_params(model="ark-code-latest")
        assert "thinking" in params
        assert "tools" in params
        assert "stream" in params
        assert "temperature" in params

    def test_thinking_parameter_handling(self):
        config = VolcEnginePlanChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled"}},
            optional_params={},
            model="ark-code-latest",
            drop_params=False,
        )
        assert result == {"extra_body": {"thinking": {"type": "enabled"}}}

    def test_thinking_disabled(self):
        config = VolcEnginePlanChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model="ark-code-latest",
            drop_params=False,
        )
        assert result == {"extra_body": {"thinking": {"type": "disabled"}}}

    def test_thinking_none_ignored(self):
        config = VolcEnginePlanChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": None},
            optional_params={},
            model="ark-code-latest",
            drop_params=False,
        )
        assert result == {}


class TestVolcEnginePlanProviderResolution:
    def test_provider_resolution(self):
        model, provider, key, api_base = get_llm_provider(
            model="volcengine_plan/ark-code-latest",
            api_base=None,
            api_key=None,
        )
        assert model == "ark-code-latest"
        assert provider == "volcengine_plan"
        assert api_base == "https://ark.cn-beijing.volces.com/api/coding/v3"

    def test_shares_api_key_with_volcengine(self):
        """volcengine_plan should use VOLCENGINE_API_KEY / ARK_API_KEY"""
        import os

        with patch.dict(os.environ, {"VOLCENGINE_API_KEY": "test-shared-key"}, clear=False):
            _, _, key, _ = get_llm_provider(
                model="volcengine_plan/ark-code-latest",
                api_base=None,
                api_key=None,
            )
            assert key == "test-shared-key"

    def test_get_optional_params(self):
        params = get_optional_params(
            model="ark-code-latest",
            custom_llm_provider="volcengine_plan",
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
                model="volcengine_plan/ark-code-latest",
                messages=[{"role": "user", "content": "hello"}],
                client=client,
            )
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "ark-code-latest"
