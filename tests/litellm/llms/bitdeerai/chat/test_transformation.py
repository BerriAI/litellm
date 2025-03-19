import os
import sys
from typing import Dict,List,Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.bitdeerai.chat.transformation import BitdeerAIChatConfig

import pytest

class TestBitdeerAIChatConfig:

    def test_inheritance(self):
        """Test proper inheritance from OpenAIGPTConfig"""
        config = BitdeerAIChatConfig()

        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
        assert isinstance(config, OpenAIGPTConfig)
        assert hasattr(config, "get_supported_openai_params") 