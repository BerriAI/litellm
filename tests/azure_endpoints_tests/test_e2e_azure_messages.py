import os
import json
import requests
import sys

sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env.test
load_dotenv(
    dotenv_path=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../.env.test")
    ),
    override=True,
)
PROXY_URL = os.environ.get("LITELLM_PROXY_URL", "http://localhost:4000")
AZURE_API_BASE = os.environ.get("AZURE_API_BASE")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")


class TestAzureMessages:
    def _validate_anthropic_message_response(
        self, data: Dict[str, Any], strict: bool = True
    ) -> None:
        """
        Validate Anthropic-style /messages response structure.

        Args:
            data: JSON response dictionary
            strict: If True, enforces strict validation. If False, allows more flexible validation.

        Raises:
            AssertionError: If response doesn't match Anthropic messages schema
        """
        assert isinstance(data, dict), f"Response should be dict, got {type(data)}"

        # Required fields for Anthropic messages response
        required_fields = [
            "id",
            "type",
            "role",
            "model",
            "content",
            "stop_reason",
            "usage",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Validate specific field types
        assert data["type"] == "message", f"Expected type 'message', got {data['type']}"
        assert (
            data["role"] == "assistant"
        ), f"Expected role 'assistant', got {data['role']}"
        assert isinstance(
            data["content"], list
        ), f"content should be list, got {type(data['content'])}"
        assert len(data["content"]) > 0, "content should not be empty"

        # Validate content structure (should be array of content blocks)
        for content_block in data["content"]:
            assert isinstance(
                content_block, dict
            ), f"content block should be dict, got {type(content_block)}"
            assert "type" in content_block, "content block missing 'type'"

            # Different content block types have different required fields
            if content_block["type"] == "text":
                assert (
                    "text" in content_block
                ), "text content block missing 'text' field"
            elif content_block["type"] == "tool_use":
                assert (
                    "id" in content_block
                ), "tool_use content block missing 'id' field"
                assert (
                    "name" in content_block
                ), "tool_use content block missing 'name' field"
                assert (
                    "input" in content_block
                ), "tool_use content block missing 'input' field"

        # Validate usage object
        assert isinstance(
            data["usage"], dict
        ), f"usage should be dict, got {type(data['usage'])}"
        assert "input_tokens" in data["usage"], "usage missing 'input_tokens'"
        assert "output_tokens" in data["usage"], "usage missing 'output_tokens'"

    def test_azure_messages_happy_path(self):
        """
        /messages happy path (Anthropic format)
        Input: valid model, valid messages in Anthropic format
        Output: 200, valid Anthropic response structure
        """
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Say hello in a brief response."}],
            "max_tokens": 50,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/messages"
        response = requests.post(url, json=payload, headers=headers)

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # Validate Anthropic response format
        self._validate_anthropic_message_response(data, strict=True)

        # Ensure content exists
        content_text = data["content"][0]["text"]
        assert len(content_text.strip()) > 0, "Response content is empty"

    def test_azure_messages_malformed_payload(self):
        """
        /messages with malformed payload
        Input: messages as string instead of array
        Output: Error response
        """
        payload = {
            "model": "gpt-4",
            "messages": "not-a-list",  # INTENTIONALLY WRONG
            "max_tokens": 50,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/messages"
        response = requests.post(url, json=payload, headers=headers)

        # Should return an error status code
        assert (
            response.status_code >= 400
        ), f"Expected error status code, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_azure_messages_streaming(self):
        """
        /messages with stream=True (Anthropic streaming format)
        Input: valid model, messages, stream=True
        Output: multiple streaming chunks in Anthropic format
        """
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "List three colors, one per line."}
            ],
            "max_tokens": 50,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/messages"
        response = requests.post(url, json=payload, headers=headers, stream=True)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Collect chunks
        chunks = []
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(data_str)
                    chunks.append(chunk_data)
                except json.JSONDecodeError:
                    continue

        # Verify we actually got streaming (multiple chunks, not single response)
        assert (
            len(chunks) > 1
        ), f"Should receive multiple chunks for streaming, got {len(chunks)}"

    def test_azure_messages_tool_calling(self):
        """
        /messages with tool/function calling (Anthropic format)
        Input: valid model, messages, tools field
        Output: 200, tool calls or direct response
        """
        payload = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": "I need to check the weather in Seattle, Washington. Please use the get_weather function to find this information.",
                }
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "state": {
                                "type": "string",
                                "description": "State abbreviation",
                            },
                        },
                        "required": ["city"],
                    },
                }
            ],
            "max_tokens": 100,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/messages"
        response = requests.post(url, json=payload, headers=headers)

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        self._validate_anthropic_message_response(
            data, strict=False
        )  # Allow tool_use without text content

        # Check if response has tool_use or text content
        has_tool_use = any(block.get("type") == "tool_use" for block in data["content"])
        has_text = any(
            block.get("type") == "text" and block.get("text")
            for block in data["content"]
        )

        assert (
            has_tool_use or has_text
        ), "Response should have either tool_use or text content"

    def test_azure_messages_invalid_role(self):
        """
        /messages with invalid role
        Input: message with invalid role
        Output: Error response
        """
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "invalid-role", "content": "This should fail"}],
            "max_tokens": 50,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/messages"
        response = requests.post(url, json=payload, headers=headers)

        # Should return an error status code
        assert (
            response.status_code >= 400
        ), f"Expected error status code, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_azure_messages_error_shape(self):
        """
        /messages error response structure
        Input: various invalid requests
        Output: consistent error structure
        """
        invalid_payloads = [
            {
                "model": "non-existent-model",
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 10,
            },
            {"model": "gpt-4", "messages": [], "max_tokens": 10},  # empty messages
            {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}],
            },  # missing max_tokens (required for Anthropic)
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/messages"

        for payload in invalid_payloads:
            response = requests.post(url, json=payload, headers=headers)

            # Should return error status
            assert response.status_code >= 400, f"Expected error for payload {payload}"

            data = response.json()
            assert "error" in data, "Response should have error field"

            # Basic error structure
            error = data["error"]
            assert "message" in error, "Error should have message field"
            assert isinstance(error["message"], str), "Error message should be string"
