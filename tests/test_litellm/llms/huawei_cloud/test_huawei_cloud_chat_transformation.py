"""
Unit tests for Huawei Cloud ModelArts MaaS chat integration.
"""

import os
import sys

import pytest

from litellm.llms.huawei_cloud.common_utils import HuaweiCloudException
from litellm.utils import get_optional_params

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.huawei_cloud.chat.transformation import HuaweiCloudChatConfig

config = HuaweiCloudChatConfig()
model = "huawei_cloud/DeepSeek-V3"


class TestHuaweiCloudChatConfig:
    def test_custom_llm_provider(self):
        """Test that custom_llm_provider returns the correct value."""
        assert config.custom_llm_provider == "huawei_cloud"

    def test_get_complete_url_default(self):
        """Test that the default base URL is set correctly."""
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api-ap-southeast-1.modelarts-maas.com/openai/v1/chat/completions"

    def test_get_complete_url_custom_base(self):
        """Test that a custom api_base is respected."""
        url = config.get_complete_url(
            api_base="https://api-ap-southeast-1.modelarts-maas.com/v1",
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api-ap-southeast-1.modelarts-maas.com/v1/chat/completions"

    def test_get_complete_url_already_has_path(self):
        """Test that a base URL already containing /chat/completions is not duplicated."""
        url = config.get_complete_url(
            api_base="https://api-ap-southeast-1.modelarts-maas.com/openai/v1/chat/completions",
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api-ap-southeast-1.modelarts-maas.com/openai/v1/chat/completions"

    def test_get_complete_url_trailing_slash(self):
        """Test that trailing slashes on api_base are stripped."""
        url = config.get_complete_url(
            api_base="https://api-ap-southeast-1.modelarts-maas.com/openai/v1/",
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api-ap-southeast-1.modelarts-maas.com/openai/v1/chat/completions"

    def test_transform_request_basic(self):
        """Test basic request transformation."""
        transformed = config.transform_request(
            model,
            messages=[{"role": "user", "content": "Hello, world!"}],
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert transformed["model"] == model
        assert transformed["messages"] == [{"role": "user", "content": "Hello, world!"}]

    def test_transform_request_with_extra_body(self):
        """Test that extra_body parameters are merged into the request."""
        transformed = config.transform_request(
            model,
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
            litellm_params={},
            headers={},
        )

        assert transformed["chat_template_kwargs"] == {"enable_thinking": True}
        assert "extra_body" not in transformed

    def test_map_openai_params(self):
        """Test OpenAI parameter mapping passes standard params through."""
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
        }

        mapped = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
            drop_params=False,
        )

        assert mapped["temperature"] == 0.7
        assert mapped["max_tokens"] == 100
        assert mapped["top_p"] == 0.9

    def test_get_error_class(self):
        """Test that get_error_class returns a HuaweiCloudException."""
        error = config.get_error_class(
            error_message="Test error",
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

        assert isinstance(error, HuaweiCloudException)
        assert error.message == "Test error"
        assert error.status_code == 400

    @pytest.mark.parametrize(
        "model_name",
        [
            "DeepSeek-V3",
            "deepseek-v3.2",
            "Qwen3-32B-0.02",
            "glm-5",
            "some-future-model-not-in-cost-map",
        ],
    )
    def test_tools_not_filtered_by_static_model_map(self, model_name):
        """
        Huawei Cloud MaaS is OpenAI-compatible; tools/tool_choice must pass
        through for any model name. The server rejects unsupported calls —
        LiteLLM must not strip them based on a stale static catalog.
        """
        params = get_optional_params(
            model=model_name,
            custom_llm_provider="huawei_cloud",
            tools=[
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
            tool_choice="auto",
        )

        assert "tools" in params
        assert "tool_choice" in params

    def test_openai_compatible_provider_info_env_fallback(self, monkeypatch):
        """Test that HUAWEI_CLOUD_API_BASE env var is used when api_base is None."""
        monkeypatch.setenv("HUAWEI_CLOUD_API_BASE", "https://custom-endpoint.example.com/v1")
        base, _ = config._get_openai_compatible_provider_info(None, None)
        assert base == "https://custom-endpoint.example.com/v1"

    def test_openai_compatible_provider_info_arg_takes_priority(self, monkeypatch):
        """Test that explicit api_base takes priority over env var."""
        monkeypatch.setenv("HUAWEI_CLOUD_API_BASE", "https://env-endpoint.example.com/v1")
        base, _ = config._get_openai_compatible_provider_info(
            "https://explicit-endpoint.example.com/v1", None
        )
        assert base == "https://explicit-endpoint.example.com/v1"


def test_huawei_cloud_integration():
    """
    Live integration test — requires HUAWEI_CLOUD_API_KEY to be set.
    Run with: pytest -k test_huawei_cloud_integration -s
    """
    from litellm import completion

    api_key = os.getenv("HUAWEI_CLOUD_API_KEY")

    if not api_key:
        pytest.skip("HUAWEI_CLOUD_API_KEY not set, skipping integration test")

    from litellm.types.utils import ModelResponse

    response = completion(
        model,
        messages=[{"role": "user", "content": "Say hello in one word"}],
        api_key=api_key,
        max_tokens=10,
        temperature=0.7,
    )

    assert isinstance(response, ModelResponse)
    assert response.choices[0].message.content
    assert len(response.choices[0].message.content.strip()) > 0
    assert response.model
    assert response.usage
    assert response.usage.total_tokens > 0


def test_huawei_cloud_streaming_integration():
    """
    Live streaming integration test — requires HUAWEI_CLOUD_API_KEY to be set.
    Run with: pytest -k test_huawei_cloud_streaming_integration -s
    """
    from litellm import completion

    api_key = os.getenv("HUAWEI_CLOUD_API_KEY")

    if not api_key:
        pytest.skip("HUAWEI_CLOUD_API_KEY not set, skipping streaming integration test")

    response = completion(
        model,
        messages=[{"role": "user", "content": "Count from 1 to 3"}],
        api_key=api_key,
        max_tokens=30,
        stream=True,
    )

    chunks = []
    content_parts = []

    for chunk in response:
        chunks.append(chunk)
        delta = chunk.choices[0].delta  # type: ignore[union-attr]
        if delta.content:
            content_parts.append(delta.content)

    assert len(chunks) > 0, "Should receive at least one chunk"
    assert len(content_parts) > 0, "Should receive content in chunks"

    full_content = "".join(content_parts)
    assert len(full_content.strip()) > 0, "Should have non-empty content"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
