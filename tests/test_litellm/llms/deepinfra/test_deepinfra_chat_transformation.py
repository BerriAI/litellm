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
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig

    # Ensure we're using the local model cost map
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    supported_openai_params = DeepInfraConfig().get_supported_openai_params(model="deepinfra/deepseek-ai/DeepSeek-V3.1")
    print(supported_openai_params)
    assert "reasoning_effort" in supported_openai_params
