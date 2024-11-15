import asyncio
import httpx
import json
import pytest
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm.exceptions import BadRequestError
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.utils import (
    CustomStreamWrapper,
    get_supported_openai_params,
    get_optional_params,
)

# test_example.py
from abc import ABC, abstractmethod


class BaseLLMChatTest(ABC):
    """
    Abstract base test class that enforces a common test across all test classes.
    """

    @abstractmethod
    def get_base_completion_call_args(self) -> dict:
        """Must return the base completion call args"""
        pass

    def test_content_list_handling(self):
        """Check if content list is supported by LLM API"""
        base_completion_call_args = self.get_base_completion_call_args()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            }
        ]
        response = litellm.completion(
            **base_completion_call_args,
            messages=messages,
        )
        assert response is not None

        # for OpenAI the content contains the JSON schema, so we need to assert that the content is not None
        assert response.choices[0].message.content is not None

    def test_message_with_name(self):
        base_completion_call_args = self.get_base_completion_call_args()
        messages = [
            {"role": "user", "content": "Hello", "name": "test_name"},
        ]
        response = litellm.completion(**base_completion_call_args, messages=messages)
        assert response is not None

    def test_json_response_format(self):
        """
        Test that the JSON response format is supported by the LLM API
        """
        base_completion_call_args = self.get_base_completion_call_args()
        litellm.set_verbose = True

        messages = [
            {
                "role": "system",
                "content": "Your output should be a JSON object with no additional properties.  ",
            },
            {
                "role": "user",
                "content": "Respond with this in json. city=San Francisco, state=CA, weather=sunny, temp=60",
            },
        ]

        response = litellm.completion(
            **base_completion_call_args,
            messages=messages,
            response_format={"type": "json_object"},
        )

        print(response)

        # OpenAI guarantees that the JSON schema is returned in the content
        # relevant issue: https://github.com/BerriAI/litellm/issues/6741
        assert response.choices[0].message.content is not None

    def test_json_response_format_stream(self):
        """
        Test that the JSON response format with streaming is supported by the LLM API
        """
        base_completion_call_args = self.get_base_completion_call_args()
        litellm.set_verbose = True

        messages = [
            {
                "role": "system",
                "content": "Your output should be a JSON object with no additional properties.  ",
            },
            {
                "role": "user",
                "content": "Respond with this in json. city=San Francisco, state=CA, weather=sunny, temp=60",
            },
        ]

        response = litellm.completion(
            **base_completion_call_args,
            messages=messages,
            response_format={"type": "json_object"},
            stream=True,
        )

        print(response)

        content = ""
        for chunk in response:
            content += chunk.choices[0].delta.content or ""

        print("content=", content)

        # OpenAI guarantees that the JSON schema is returned in the content
        # relevant issue: https://github.com/BerriAI/litellm/issues/6741
        # we need to assert that the JSON schema was returned in the content, (for Anthropic we were returning it as part of the tool call)
        assert content is not None
        assert len(content) > 0

    @pytest.fixture
    def pdf_messages(self):
        import base64

        import requests

        # URL of the file
        url = "https://storage.googleapis.com/cloud-samples-data/generative-ai/pdf/2403.05530.pdf"

        response = requests.get(url)
        file_data = response.content

        encoded_file = base64.b64encode(file_data).decode("utf-8")
        url = f"data:application/pdf;base64,{encoded_file}"

        image_content = [
            {"type": "text", "text": "What's this file about?"},
            {
                "type": "image_url",
                "image_url": {"url": url},
            },
        ]

        image_messages = [{"role": "user", "content": image_content}]

        return image_messages
