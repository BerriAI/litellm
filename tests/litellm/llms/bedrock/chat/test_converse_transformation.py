import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.types.llms.bedrock import ConverseTokenUsageBlock


def test_transform_usage():
    usage = ConverseTokenUsageBlock(
        **{
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 10,
            "cacheCreationInputTokenCount": 0,
            "cacheCreationInputTokens": 0,
            "inputTokens": 12,
            "outputTokens": 56,
            "totalTokens": 78,
        }
    )
    config = AmazonConverseConfig()
    openai_usage = config._transform_usage(usage)
    assert openai_usage.prompt_tokens == 22
    assert openai_usage.completion_tokens == 56
    assert openai_usage.total_tokens == 78
    assert openai_usage.prompt_tokens_details.cached_tokens == 10
    assert openai_usage._cache_creation_input_tokens == 0
    assert openai_usage._cache_read_input_tokens == 10
