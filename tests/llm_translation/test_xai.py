import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse, EmbeddingResponse, Usage
from litellm import completion
from unittest.mock import patch
from litellm.llms.xai.chat.xai_transformation import XAIChatConfig, XAI_API_BASE


def test_xai_chat_config_get_openai_compatible_provider_info():
    config = XAIChatConfig()

    # Test with default values
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=None
    )
    assert api_base == XAI_API_BASE
    assert api_key == os.environ.get("XAI_API_KEY")

    # Test with custom API key
    custom_api_key = "test_api_key"
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=custom_api_key
    )
    assert api_base == XAI_API_BASE
    assert api_key == custom_api_key

    # Test with custom environment variables for api_base and api_key
    with patch.dict(
        "os.environ",
        {"XAI_API_BASE": "https://env.x.ai/v1", "XAI_API_KEY": "env_api_key"},
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://env.x.ai/v1"
        assert api_key == "env_api_key"


def test_completion_xai():
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="xai/grok-beta",
            messages=messages,
        )
        print(response)

        assert response is not None
        assert isinstance(response, litellm.ModelResponse)
        assert response.choices[0].message.content is not None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
