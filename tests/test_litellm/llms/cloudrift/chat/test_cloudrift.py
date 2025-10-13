import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add litellm to path
sys.path.insert(0, os.path.abspath("../../../.."))
import litellm


def test_deepseek_supported_openai_params():
    """
    Test "reasoning_effort" is an openai param supported for the DeepSeek model on deepinfra
    """
    from litellm.llms.cloudrift.chat.transformation import CloudRiftChatConfig

    supported_openai_params = CloudRiftChatConfig().get_supported_openai_params(model="cloudrift/moonshotai/Kimi-K2-Instruct")
    print(supported_openai_params)
    assert "temperature" in supported_openai_params


def test_max_completion_tokens_support():
    """Test that max_completion_tokens parameter is supported"""

    from litellm.llms.cloudrift.chat.transformation import CloudRiftChatConfig
    config = CloudRiftChatConfig(max_completion_tokens=150)
    assert config.max_tokens == 150