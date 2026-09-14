from litellm.llms.byteplus.chat.transformation import BytePlusChatConfig


class TestBytePlusChatConfig:
    def test_get_supported_openai_params(self):
        config = BytePlusChatConfig()
        params = config.get_supported_openai_params("byteplus/seed-2-0-lite-260228")
        assert "max_completion_tokens" in params
        assert "thinking" in params
        assert "stream" in params

    def test_map_openai_params_max_completion_tokens(self):
        config = BytePlusChatConfig()
        non_default = {"max_completion_tokens": 100}
        optional = {}
        res = config.map_openai_params(
            non_default_params=non_default,
            optional_params=optional,
            model="byteplus/seed-2-0-lite-260228",
            drop_params=False,
        )
        assert res.get("max_tokens") == 100

    def test_map_openai_params_thinking(self):
        config = BytePlusChatConfig()
        non_default = {"thinking": {"type": "disabled"}}
        optional = {}
        res = config.map_openai_params(
            non_default_params=non_default,
            optional_params=optional,
            model="byteplus/seed-2-0-lite-260228",
            drop_params=False,
        )
        assert res.get("extra_body", {}).get("thinking") == {"type": "disabled"}

    def test_map_openai_params_thinking_bool(self):
        config = BytePlusChatConfig()
        res_enabled = config.map_openai_params(
            non_default_params={"thinking": True},
            optional_params={},
            model="byteplus/seed-2-0-lite-260228",
            drop_params=False,
        )
        assert res_enabled.get("extra_body", {}).get("thinking") == {"type": "enabled"}

        res_disabled = config.map_openai_params(
            non_default_params={"thinking": False},
            optional_params={},
            model="byteplus/seed-2-0-lite-260228",
            drop_params=False,
        )
        assert res_disabled.get("extra_body", {}).get("thinking") == {"type": "disabled"}

    def test_get_supported_openai_params_core_util(self):
        from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params

        params = get_supported_openai_params("byteplus/seed-2-0-lite-260228", custom_llm_provider="byteplus")
        assert "thinking" in params
        assert "max_completion_tokens" in params

    def test_get_llm_provider_logic(self):
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, dynamic_key, api_base = get_llm_provider("byteplus/ep-20250101", api_key="test-key")
        assert provider == "byteplus"
        assert model == "ep-20250101"
        assert dynamic_key == "test-key"
        assert api_base == "https://ark.ap-southeast.bytepluses.com/api/v3"

    def test_provider_config_manager_chat(self):
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        cfg = ProviderConfigManager.get_provider_chat_config(
            model="byteplus/seed-2-0-lite", provider=LlmProviders.BYTEPLUS
        )
        assert isinstance(cfg, BytePlusChatConfig)

    def test_litellm_completion_byteplus(self):
        from unittest.mock import MagicMock, patch
        from openai import OpenAI
        from litellm import completion
        from litellm.types.utils import Choices, Message, ModelResponse

        client = OpenAI(api_key="test_api_key")
        mock_raw_response = MagicMock()
        mock_raw_response.headers = {"x-request-id": "123"}
        mock_raw_response.parse.return_value = ModelResponse(
            id="chatcmpl-123",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello from BytePlus!", role="assistant"),
                )
            ],
            model="byteplus/seed-2-0-lite",
        )

        mock_create = MagicMock(return_value=mock_raw_response)
        with patch.object(client.chat.completions.with_raw_response, "create", mock_create):
            res = completion(
                model="byteplus/seed-2-0-lite",
                messages=[{"role": "user", "content": "Hello"}],
                client=client,
            )
            mock_create.assert_called_once()
            assert res.choices[0].message.content == "Hello from BytePlus!"

    def test_litellm_completion_byteplus_make_request(self):
        from unittest.mock import patch
        from litellm import completion
        from litellm.types.utils import Choices, Message, ModelResponse

        model_response = ModelResponse(
            id="chatcmpl-123",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello from BytePlus!", role="assistant"),
                )
            ],
            model="byteplus/seed-2-0-lite",
        )

        with patch(
            "litellm.llms.openai.openai.OpenAIChatCompletion.make_sync_openai_chat_completion_request",
            return_value=({}, model_response),
        ):
            res = completion(
                model="byteplus/seed-2-0-lite",
                messages=[{"role": "user", "content": "hello"}],
                api_key="test-api-key",
            )
            assert res.choices[0].message.content == "Hello from BytePlus!"
