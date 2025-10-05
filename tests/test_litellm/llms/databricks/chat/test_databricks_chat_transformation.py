import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

from litellm.llms.databricks.chat.transformation import (
    DatabricksChatResponseIterator,
    DatabricksConfig,
)


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

def test_convert_anthropic_tool_to_databricks_tool_with_description():
    config = DatabricksConfig()
    anthropic_tool = {
        "name": "test_tool",
        "description": "test description",
        "input_schema": {"type": "object", "properties": {"test": {"type": "string"}}}
    }

    databricks_tool = config.convert_anthropic_tool_to_databricks_tool(anthropic_tool)

    assert databricks_tool is not None
    assert databricks_tool["type"] == "function"
    assert databricks_tool["function"]["description"] == "test description"


def test_convert_anthropic_tool_to_databricks_tool_without_description():
    config = DatabricksConfig()
    anthropic_tool = {
        "name": "test_tool",
        "input_schema": {"type": "object", "properties": {"test": {"type": "string"}}}
    }

    databricks_tool = config.convert_anthropic_tool_to_databricks_tool(anthropic_tool)

    assert databricks_tool is not None
    assert databricks_tool["type"] == "function"
    assert databricks_tool["function"].get("description") is None

def test_transform_choices_with_citations():
    config = DatabricksConfig()
    databricks_choices = [
        {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Blue",
                        "citations": [
                            {
                                "type": "char_location",
                                "cited_text": "The sky is blue.",
                                "document_index": 0,
                                "document_title": "My Document",
                                "start_char_index": 0,
                                "end_char_index": 50,
                            }
                        ],
                    }
                ],
            },
            "index": 0,
            "finish_reason": "stop",
        }
    ]

    choices = config._transform_dbrx_choices(choices=databricks_choices)

    assert choices[0].message.provider_specific_fields == {
        "citations": [
            [
                {
                    "type": "char_location",
                    "cited_text": "The sky is blue.",
                    "document_index": 0,
                    "document_title": "My Document",
                    "start_char_index": 0,
                    "end_char_index": 50,
                    "supported_text": "Blue",
                }
            ]
        ]
    }


def test_chunk_parser_with_citation():
    iterator = DatabricksChatResponseIterator(None, sync_stream=True)
    chunk = {
        "id": "1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test",
        "choices": [
            {
                "delta": {
                    "content": [
                        {
                            "type": "text",
                            "text": "",
                            "citations": [
                                {
                                    "type": "char_location",
                                    "cited_text": "The sky is blue.",
                                    "document_index": 0,
                                    "document_title": "My Document",
                                    "start_char_index": 0,
                                    "end_char_index": 50,
                                }
                            ],
                        }
                    ],
                },
                "index": 0,
                "finish_reason": None,
            }
        ],
    }

    parsed = iterator.chunk_parser(chunk)
    assert parsed.choices[0].delta.provider_specific_fields == {
        "citation": {
            "type": "char_location",
            "cited_text": "The sky is blue.",
            "document_index": 0,
            "document_title": "My Document",
            "start_char_index": 0,
            "end_char_index": 50,
        }
    }


def test_is_foundational_model():
    """Test the _is_foundational_model method to ensure it correctly identifies Databricks foundational models."""
    config = DatabricksConfig()
    
    # Mock get_model_info to return foundational model info
    mock_model_info_foundational = {
        "foundational_model": True,
        "litellm_provider": "databricks",
        "mode": "chat"
    }
    
    mock_model_info_non_foundational = {
        "foundational_model": False,
        "litellm_provider": "databricks", 
        "mode": "chat"
    }
    
    with patch('litellm.llms.databricks.chat.transformation.get_model_info') as mock_get_model_info:
        # Test with a foundational model
        mock_get_model_info.side_effect = [
            mock_model_info_foundational,  # databricks-claude-3-7-sonnet
            mock_model_info_foundational,  # databricks-llama-4-maverick
            mock_model_info_non_foundational,  # databricks-llama-2-70b-chat
            Exception("Model not found")  # unknown model
        ]
        
        foundational_model = "databricks-claude-3-7-sonnet"
        result = config._is_foundational_model(foundational_model)
        assert result is True, f"Expected {foundational_model} to be identified as foundational"
        
        # Test with another foundational model
        foundational_model2 = "databricks-llama-4-maverick"
        result2 = config._is_foundational_model(foundational_model2)
        assert result2 is True, f"Expected {foundational_model2} to be identified as foundational"
        
        # Test with a non-foundational model
        non_foundational_model = "databricks-llama-2-70b-chat"
        result3 = config._is_foundational_model(non_foundational_model)
        assert result3 is False, f"Expected {non_foundational_model} to NOT be identified as foundational"
        
        # Test with unknown model (should return False due to exception)
        unknown_model = "some-random-model-name"
        result4 = config._is_foundational_model(unknown_model)
        assert result4 is False, f"Expected {unknown_model} to NOT be identified as foundational"


def test_get_complete_url_foundational_model():
    """Test that get_complete_url returns the correct URL for foundational models."""
    config = DatabricksConfig()
    
    # Mock get_model_info to return foundational model info
    mock_model_info_foundational = {
        "foundational_model": True,
        "litellm_provider": "databricks",
        "mode": "chat"
    }
    
    mock_model_info_non_foundational = {
        "foundational_model": False,
        "litellm_provider": "databricks",
        "mode": "chat"
    }
    
    # Mock the _get_api_base method to return a test API base
    with patch.object(config, '_get_api_base', return_value="https://test.databricks.com"), \
         patch('litellm.llms.databricks.chat.transformation.get_model_info') as mock_get_model_info:
        
        # Test with a foundational model
        mock_get_model_info.return_value = mock_model_info_foundational
        url = config.get_complete_url(
            api_base="https://test.databricks.com",
            api_key="test-key",
            model="databricks-claude-3-7-sonnet",
            optional_params={},
            litellm_params={}
        )
        expected_url = "https://test.databricks.com/serving-endpoints/databricks-claude-3-7-sonnet/invocations"
        assert url == expected_url, f"Expected {expected_url}, got {url}"
        
        # Test with a non-foundational model
        mock_get_model_info.return_value = mock_model_info_non_foundational
        url2 = config.get_complete_url(
            api_base="https://test.databricks.com",
            api_key="test-key",
            model="llama-2-70b-chat",
            optional_params={},
            litellm_params={}
        )
        expected_url2 = "https://test.databricks.com/serving-endpoints/chat/completions"
        assert url2 == expected_url2, f"Expected {expected_url2}, got {url2}"
