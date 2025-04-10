import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.anthropic.chat.transformation import AnthropicConfig


def test_response_format_transformation_unit_test():
    config = AnthropicConfig()

    response_format_json_schema = {
        "description": 'Progress report for the thinking process\n\nThis model represents a snapshot of the agent\'s current progress during\nthe thinking process, providing a brief description of the current activity.\n\nAttributes:\n    agent_doing: Brief description of what the agent is currently doing.\n                Should be kept under 10 words. Example: "Learning about home automation"',
        "properties": {"agent_doing": {"title": "Agent Doing", "type": "string"}},
        "required": ["agent_doing"],
        "title": "ThinkingStep",
        "type": "object",
        "additionalProperties": False,
    }

    result = config._create_json_tool_call_for_response_format(
        json_schema=response_format_json_schema
    )

    assert result["input_schema"]["properties"] == {
        "agent_doing": {"title": "Agent Doing", "type": "string"}
    }
    print(result)


def test_calculate_usage():
    """
    Do not include cache_creation_input_tokens in the prompt_tokens

    Fixes https://github.com/BerriAI/litellm/issues/9812
    """
    config = AnthropicConfig()

    usage_object = {
        "input_tokens": 3,
        "cache_creation_input_tokens": 12304,
        "cache_read_input_tokens": 0,
        "output_tokens": 550,
    }
    usage = config.calculate_usage(usage_object=usage_object, reasoning_content=None)
    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 550
    assert usage.total_tokens == 3 + 550
    assert usage.prompt_tokens_details.cached_tokens == 0
    assert usage._cache_creation_input_tokens == 12304
    assert usage._cache_read_input_tokens == 0
