import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.databricks.chat.transformation import DatabricksConfig


def test_transform_choices():
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "i'm thinking.",
                                "signature": "ErcBCkgIAhABGAIiQMadog2CAJc8YJdce2Cmqvk0MFB+gGt4OyaH4c3l9p9v+0TKhYcNGliFkxddhCVkYR8zz8oaO1f3cHaEmYXN5SISDGAaomDR7CaTrhZxURoMbOR7AfFuHcIdVXFSIjC9ZamSyhzMg3maOtq2QHLXr6Z7tv0dut2S0Icdqk4g7MOFTSnCc0jA7lvnJyjI0wMqHR05PoVXEDSQjAV6NcUFkzFzp34z0xVMaK/VatCT",
                            }
                        ],
                    },
                    {"type": "text", "text": "# 5 Question and Answer Pairs"},
                ],
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert len(choices) == 1
    assert choices[0].message.content == "# 5 Question and Answer Pairs"
    assert choices[0].message.reasoning_content == "i'm thinking."
    assert choices[0].message.thinking_blocks is not None
    assert choices[0].message.tool_calls is None


def test_transform_choices_without_signature():
    """
    Test that the transformation works correctly when the signature field is missing
    from the summary, which occurs with new Databricks Foundation Models like
    databricks-gpt-oss-20b and databricks-gpt-oss-120b.
    """
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "summary": [
                            {
                                "type": "summary_text",
                                "text": "i'm thinking without signature.",
                                # Note: no signature field here
                            }
                        ],
                    },
                    {"type": "text", "text": "Response without signature"},
                ],
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    # This should not raise a KeyError for missing signature
    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert len(choices) == 1
    assert choices[0].message.content == "Response without signature"
    assert choices[0].message.reasoning_content == "i'm thinking without signature."
    assert choices[0].message.thinking_blocks is not None
    assert len(choices[0].message.thinking_blocks) == 1
    
    # Verify the thinking block was created successfully without signature
    thinking_block = choices[0].message.thinking_blocks[0]
    assert thinking_block["type"] == "thinking"
    assert thinking_block["thinking"] == "i'm thinking without signature."


def test_map_openai_params_preserves_max_completion_tokens_not_max_tokens():
    config = DatabricksConfig()
    non_default_params = {"max_completion_tokens": 20000}
    optional_params = {}
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="databricks/se-gpt-5-mini",
        drop_params=False,
    )
    assert result.get("max_completion_tokens") == 20000
    assert "max_tokens" not in result


def test_map_openai_params_preserves_max_tokens_not_max_completion_tokens():
    config = DatabricksConfig()
    non_default_params = {"max_tokens": 1024}
    optional_params = {}
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="databricks/databricks-dbrx-instruct",
        drop_params=False,
    )
    assert result.get("max_tokens") == 1024
    assert "max_completion_tokens" not in result


def test_map_openai_params_prefers_max_completion_tokens_when_both_provided():
    config = DatabricksConfig()
    non_default_params = {"max_tokens": 512, "max_completion_tokens": 20000}
    optional_params = {}
    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="databricks/se-gpt-5-mini",
        drop_params=False,
    )
    # Should keep only max_completion_tokens
    assert result.get("max_completion_tokens") == 20000
    assert "max_tokens" not in result
