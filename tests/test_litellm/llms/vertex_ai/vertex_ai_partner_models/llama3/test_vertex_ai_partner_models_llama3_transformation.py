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
        # response = config.transform_response(
