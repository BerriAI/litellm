import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.siliconflow.chat.transformation import SiliconFlowChatConfig


class TestSiliconFlowChatConfig:
    def setup_method(self):
        self.config = SiliconFlowChatConfig()

    def test_get_complete_url_defaults_to_chat_completions(self):
        assert (
            self.config.get_complete_url(
                api_base=None,
                api_key=None,
                model="deepseek-ai/DeepSeek-V4-Flash",
                optional_params={},
                litellm_params={},
            )
            == "https://api.siliconflow.cn/v1/chat/completions"
        )

    def test_get_complete_url_preserves_full_endpoint(self):
        assert (
            self.config.get_complete_url(
                api_base="https://api.siliconflow.cn/v1/chat/completions",
                api_key=None,
                model="deepseek-ai/DeepSeek-V4-Flash",
                optional_params={},
                litellm_params={},
            )
            == "https://api.siliconflow.cn/v1/chat/completions"
        )

    def test_validate_environment_sets_bearer_auth(self):
        headers = self.config.validate_environment(
            headers={},
            model="deepseek-ai/DeepSeek-V4-Flash",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    def test_supported_params_include_reasoning_effort(self):
        supported_params = self.config.get_supported_openai_params(
            "deepseek-ai/DeepSeek-V4-Flash"
        )
        assert "reasoning_effort" in supported_params
