import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.vertex_ai.vertex_ai_partner_models.llama3.transformation import (
    VertexAILlama3Config,
    VertexAILlama3StreamingHandler,
)


class TestVertexAILlama3Config:
    def test_transform_choices(self):
        """
        Relevant Issue: https://github.com/BerriAI/litellm/issues/10441#issuecomment-2844975599
        """
        config = VertexAILlama3Config()

        choices = [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": '{"type": "function", "name": "get_weather", "parameters": {"location": "Boston, MA"}}',
                    "role": "assistant",
                },
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current temperature for a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and country e.g. Bogotá, Colombia",
                            }
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        optional_params = {"tools": tools}
        response = config._transform_choices(
            choices=choices, json_mode=False, optional_params=optional_params
        )
        assert response[0].message.tool_calls is not None
        assert response[0].finish_reason == "tool_calls"


class TestVertexAILlama3StreamingHandler:
    def test_first_chunk_has_role_assistant_when_missing(self):
        """
        Vertex AI Llama streaming may return chunks without role in delta.
        The handler should inject role='assistant' on the first chunk.
        """
        handler = VertexAILlama3StreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
        )
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "meta/llama-4-scout-17b-16e-instruct-maas",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None, "role": None},
                    "finish_reason": None,
                }
            ],
        }
        result = handler.chunk_parser(chunk)
        assert result.choices[0].delta.role == "assistant"

    def test_subsequent_chunks_no_role_override(self):
        """
        Only the first chunk should have role injected.
        """
        handler = VertexAILlama3StreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
        )
        first_chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "meta/llama-4-scout-17b-16e-instruct-maas",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hello", "role": None},
                    "finish_reason": None,
                }
            ],
        }
        second_chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "meta/llama-4-scout-17b-16e-instruct-maas",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": " world", "role": None},
                    "finish_reason": None,
                }
            ],
        }
        first_result = handler.chunk_parser(first_chunk)
        second_result = handler.chunk_parser(second_chunk)
        assert first_result.choices[0].delta.role == "assistant"
        assert second_result.choices[0].delta.role is None

    def test_first_chunk_preserves_existing_role(self):
        """
        If the API already provides role, don't overwrite it.
        """
        handler = VertexAILlama3StreamingHandler(
            streaming_response=iter([]),
            sync_stream=True,
        )
        chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 123,
            "model": "meta/llama-4-scout-17b-16e-instruct-maas",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": None, "role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        result = handler.chunk_parser(chunk)
        assert result.choices[0].delta.role == "assistant"
