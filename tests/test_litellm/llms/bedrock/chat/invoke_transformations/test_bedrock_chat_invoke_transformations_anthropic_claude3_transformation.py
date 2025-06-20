import asyncio
import json
import os
import sys

import pytest

# Ensure the project root is on the import path so `litellm` can be imported when
# tests are executed from any working directory.
sys.path.insert(0, os.path.abspath("../../../../../.."))

from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaude3Config,
)


def test_get_supported_params_thinking():
    config = AmazonAnthropicClaude3Config()
    params = config.get_supported_openai_params(
        model="anthropic.claude-sonnet-4-20250514-v1:0"
    )
    assert "thinking" in params
