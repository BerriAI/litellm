"""
Test that model names in batch JSONL files are correctly resolved for Vertex AI.

Regression test for: Model names not being resolved in batch files
"""

import json
import pytest
from unittest.mock import patch

from litellm.llms.vertex_ai.files.transformation import _openai_batch_jsonl_entries_to_vertex_wrapped_requests


class TestBatchModelNameResolution:
    """Test that model names in batch files are correctly resolved"""

    def test_model_name_resolved_from_litellm_format(self):
        """
        Test that model names in litellm format (e.g., 'gemini/3-5-flash:vertex')
        are correctly resolved to Vertex AI format (e.g., 'gemini-3.5-flash')
        """
        openai_jsonl_content = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gemini/3-5-flash:vertex",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello world!"},
                    ],
                    "max_tokens": 10,
                },
            }
        ]

        def mock_map_openai_to_vertex_params(openai_request_body):
            return {}

        with patch(
            "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider"
        ) as mock_get_llm_provider:
            mock_get_llm_provider.return_value = (
                "gemini-3.5-flash",
                "vertex_ai",
                None,
                None,
            )

            result = _openai_batch_jsonl_entries_to_vertex_wrapped_requests(
                openai_jsonl_content=openai_jsonl_content,
                map_openai_to_vertex_params=mock_map_openai_to_vertex_params,
            )

            assert len(result) == 1
            assert "request" in result[0]
            mock_get_llm_provider.assert_called_once_with(
                model="gemini/3-5-flash:vertex"
            )

    def test_model_name_resolved_with_provider_prefix(self):
        """
        Test that model names with provider prefix (e.g., 'vertex_ai/gemini-2.0-flash')
        are correctly resolved to Vertex AI format (e.g., 'gemini-2.0-flash')
        """
        openai_jsonl_content = [
            {
                "custom_id": "request-2",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "vertex_ai/gemini-2.0-flash",
                    "messages": [
                        {"role": "user", "content": "Test message"},
                    ],
                },
            }
        ]

        def mock_map_openai_to_vertex_params(openai_request_body):
            return {}

        with patch(
            "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider"
        ) as mock_get_llm_provider:
            mock_get_llm_provider.return_value = (
                "gemini-2.0-flash",
                "vertex_ai",
                None,
                None,
            )

            result = _openai_batch_jsonl_entries_to_vertex_wrapped_requests(
                openai_jsonl_content=openai_jsonl_content,
                map_openai_to_vertex_params=mock_map_openai_to_vertex_params,
            )

            assert len(result) == 1
            assert "request" in result[0]
            mock_get_llm_provider.assert_called_once_with(
                model="vertex_ai/gemini-2.0-flash"
            )

    def test_model_name_preserved_if_resolution_fails(self):
        """
        Test that if model name resolution fails, the original model name is preserved
        """
        openai_jsonl_content = [
            {
                "custom_id": "request-3",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "unknown-model",
                    "messages": [
                        {"role": "user", "content": "Test message"},
                    ],
                },
            }
        ]

        def mock_map_openai_to_vertex_params(openai_request_body):
            return {}

        with patch(
            "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider"
        ) as mock_get_llm_provider:
            mock_get_llm_provider.side_effect = Exception("Model not found")

            result = _openai_batch_jsonl_entries_to_vertex_wrapped_requests(
                openai_jsonl_content=openai_jsonl_content,
                map_openai_to_vertex_params=mock_map_openai_to_vertex_params,
            )

            assert len(result) == 1
            assert "request" in result[0]
            mock_get_llm_provider.assert_called_once_with(model="unknown-model")

    def test_multiple_requests_with_different_models(self):
        """
        Test that multiple requests with different model names are all correctly resolved
        """
        openai_jsonl_content = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gemini/3-5-flash:vertex",
                    "messages": [
                        {"role": "user", "content": "Message 1"},
                    ],
                },
            },
            {
                "custom_id": "request-2",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "vertex_ai/gemini-2.5-pro",
                    "messages": [
                        {"role": "user", "content": "Message 2"},
                    ],
                },
            },
        ]

        def mock_map_openai_to_vertex_params(openai_request_body):
            return {}

        with patch(
            "litellm.litellm_core_utils.get_llm_provider_logic.get_llm_provider"
        ) as mock_get_llm_provider:
            mock_get_llm_provider.side_effect = [
                ("gemini-3.5-flash", "vertex_ai", None, None),
                ("gemini-2.5-pro", "vertex_ai", None, None),
            ]

            result = _openai_batch_jsonl_entries_to_vertex_wrapped_requests(
                openai_jsonl_content=openai_jsonl_content,
                map_openai_to_vertex_params=mock_map_openai_to_vertex_params,
            )

            assert len(result) == 2
            assert mock_get_llm_provider.call_count == 2
            mock_get_llm_provider.assert_any_call(model="gemini/3-5-flash:vertex")
            mock_get_llm_provider.assert_any_call(model="vertex_ai/gemini-2.5-pro")
