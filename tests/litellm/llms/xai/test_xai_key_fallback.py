import os
import litellm
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.llms.xai.common_utils import XAIModelInfo
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig


def test_get_api_key_priority():
    """
    Test the fallback order of XAI API key resolution:
    1. api_key parameter
    2. litellm.xai_key
    3. XAI_API_KEY environment variable
    4. litellm.api_key
    5. None
    """

    original_env = os.environ.get("XAI_API_KEY")
    had_env = "XAI_API_KEY" in os.environ
    original_xai_key = litellm.xai_key
    original_api_key = litellm.api_key

    try:
        # Case 1: api_key parameter is passed
        litellm.xai_key = "xai_key_value"
        litellm.api_key = "common_api_key"
        os.environ["XAI_API_KEY"] = "env_api_key"
        result = XAIModelInfo.get_api_key("param_api_key")
        assert result == "param_api_key"

        # Case 2: api_key not passed, use litellm.xai_key
        result = XAIModelInfo.get_api_key(None)
        assert result == "xai_key_value"

        # Case 3: api_key and xai_key not set, prefer XAI_API_KEY over litellm.api_key
        litellm.xai_key = None
        result = XAIModelInfo.get_api_key(None)
        assert result == "env_api_key"

        # Case 4: Empty XAI_API_KEY falls through to litellm.api_key
        os.environ["XAI_API_KEY"] = ""
        result = XAIModelInfo.get_api_key(None)
        assert result == "common_api_key"

        # Case 5: None of the above, return None
        os.environ.pop("XAI_API_KEY", None)
        litellm.api_key = None
        result = XAIModelInfo.get_api_key(None)
        assert result is None
    finally:
        if had_env:
            os.environ["XAI_API_KEY"] = original_env
        else:
            os.environ.pop("XAI_API_KEY", None)
        litellm.xai_key = original_xai_key
        litellm.api_key = original_api_key


def test_chat_config_uses_xai_key_fallback():
    original_env = os.environ.get("XAI_API_KEY")
    had_env = "XAI_API_KEY" in os.environ
    original_xai_key = litellm.xai_key
    original_api_key = litellm.api_key

    try:
        litellm.xai_key = "xai_key_value"
        litellm.api_key = None
        os.environ.pop("XAI_API_KEY", None)
        _, api_key = XAIChatConfig()._get_openai_compatible_provider_info(None, None)
        assert api_key == "xai_key_value"
    finally:
        if had_env:
            os.environ["XAI_API_KEY"] = original_env
        else:
            os.environ.pop("XAI_API_KEY", None)
        litellm.xai_key = original_xai_key
        litellm.api_key = original_api_key


def test_responses_config_uses_xai_key_fallback():
    original_env = os.environ.get("XAI_API_KEY")
    had_env = "XAI_API_KEY" in os.environ
    original_xai_key = litellm.xai_key
    original_api_key = litellm.api_key

    try:
        litellm.xai_key = "xai_key_value"
        litellm.api_key = None
        os.environ.pop("XAI_API_KEY", None)
        headers = XAIResponsesAPIConfig().validate_environment(
            {}, "xai/grok-3-mini", None
        )
        assert headers["Authorization"] == "Bearer xai_key_value"
    finally:
        if had_env:
            os.environ["XAI_API_KEY"] = original_env
        else:
            os.environ.pop("XAI_API_KEY", None)
        litellm.xai_key = original_xai_key
        litellm.api_key = original_api_key
