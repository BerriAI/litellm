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
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 1789,
            "cacheWriteInputTokens": 1789,
            "inputTokens": 3,
            "outputTokens": 401,
            "totalTokens": 2193,
        }
    )
    config = AmazonConverseConfig()
    openai_usage = config._transform_usage(usage)
    assert (
        openai_usage.prompt_tokens
        == usage["inputTokens"]
        + usage["cacheWriteInputTokens"]
        + usage["cacheReadInputTokens"]
    )
    assert openai_usage.completion_tokens == usage["outputTokens"]
    assert openai_usage.total_tokens == usage["totalTokens"]
    assert (
        openai_usage.prompt_tokens_details.cached_tokens
        == usage["cacheReadInputTokens"]
    )
    assert openai_usage._cache_creation_input_tokens == usage["cacheWriteInputTokens"]
    assert openai_usage._cache_read_input_tokens == usage["cacheReadInputTokens"]
