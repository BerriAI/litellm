import os
import json
import requests
import sys

sys.path.append(os.path.dirname(__file__))
from dotenv import load_dotenv

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


# Test cases for Azure OpenAI Chat Completions API via LiteLLM proxy
class TestAzureChatCompletions:
    """Test Azure OpenAI Chat Completions API endpoint via LiteLLM proxy"""

    def _validate_chat_completion_response(self, data, strict=True):
        """
        Validate OpenAI chat completion response structure.

        Args:
            data: JSON response dictionary
            strict: If True, requires content in message. If False, allows tool_calls without content.

        Raises:
            AssertionError: If response doesn't match OpenAI schema
        """
        # Required top-level fields
        assert isinstance(
            data, dict
        ), f"Response must be a dictionary, got {type(data)}"

        required_fields = ["id", "object", "created", "model", "choices"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Validate field types
        assert isinstance(
            data["id"], str
        ), f"'id' must be string, got {type(data['id'])}"
        assert (
            data["object"] == "chat.completion"
        ), f"'object' must be 'chat.completion', got {data['object']}"
        assert isinstance(
            data["created"], int
        ), f"'created' must be integer, got {type(data['created'])}"
        assert isinstance(
            data["model"], str
        ), f"'model' must be string, got {type(data['model'])}"
        assert isinstance(
            data["choices"], list
        ), f"'choices' must be list, got {type(data['choices'])}"
        assert len(data["choices"]) > 0, "Response must have at least one choice"

        # Validate usage field (should be present for non-streaming)
        if "usage" in data:
            self._validate_usage_object(data["usage"])

        # Validate each choice
        for i, choice in enumerate(data["choices"]):
            self._validate_choice_object(choice, i, strict=strict)

    def _validate_usage_object(self, usage):
        """Validate OpenAI usage object structure."""
        assert isinstance(usage, dict), f"usage must be dict, got {type(usage)}"

        required_fields = ["prompt_tokens", "completion_tokens", "total_tokens"]
        for field in required_fields:
            assert field in usage, f"usage missing required field: {field}"
            assert isinstance(
                usage[field], int
            ), f"usage[{field}] must be int, got {type(usage[field])}"
            assert (
                usage[field] >= 0
            ), f"usage[{field}] must be non-negative, got {usage[field]}"

        # Validate total equals sum
        expected_total = usage["prompt_tokens"] + usage["completion_tokens"]
        assert (
            usage["total_tokens"] == expected_total
        ), f"usage total_tokens mismatch: expected {expected_total}, got {usage['total_tokens']}"

    def _validate_choice_object(self, choice, index, strict=True):
        """Validate OpenAI choice object structure."""
        assert isinstance(
            choice, dict
        ), f"choice[{index}] must be dict, got {type(choice)}"

        required_fields = ["index", "message", "finish_reason"]
        for field in required_fields:
            assert field in choice, f"choice[{index}] missing required field: {field}"

        assert (
            choice["index"] == index
        ), f"choice[{index}] index mismatch: expected {index}, got {choice['index']}"
        assert isinstance(
            choice["finish_reason"], (str, type(None))
        ), f"choice[{index}] finish_reason must be string or null, got {type(choice['finish_reason'])}"

        self._validate_message_object(
            choice["message"], strict=strict, finish_reason=choice["finish_reason"]
        )

    def _validate_message_object(self, message, strict=True, finish_reason=None):
        """Validate OpenAI message object structure."""
        assert isinstance(message, dict), f"message must be dict, got {type(message)}"

        required_fields = ["role"]
        for field in required_fields:
            assert field in message, f"message missing required field: {field}"

        assert (
            message["role"] == "assistant"
        ), f"message role must be 'assistant', got {message['role']}"

        # Content validation based on finish_reason and strict mode
        has_content = "content" in message and message["content"] is not None
        has_tool_calls = "tool_calls" in message and message["tool_calls"]

        if strict and finish_reason == "stop":
            assert has_content, "message must have content when finish_reason is 'stop'"

        if has_tool_calls:
            assert isinstance(message["tool_calls"], list), "tool_calls must be list"
            for i, tool_call in enumerate(message["tool_calls"]):
                assert isinstance(tool_call, dict), f"tool_calls[{i}] must be dict"

                # Basic tool call validation
                required_tool_fields = ["id", "type", "function"]
                for field in required_tool_fields:
                    assert (
                        field in tool_call
                    ), f"tool_calls[{i}] missing required field: {field}"

                assert (
                    tool_call["type"] == "function"
                ), f"tool_calls[{i}] type must be 'function'"

                function = tool_call["function"]
                assert isinstance(
                    function, dict
                ), f"tool_calls[{i}].function must be dict"
                assert "name" in function, f"tool_calls[{i}].function missing 'name'"
                assert (
                    "arguments" in function
                ), f"tool_calls[{i}].function missing 'arguments'"

    def test_azure_chat_completions_happy_path(self):
        """
        /chat/completions happy path
        Input: valid model, valid messages
        Output: 200, valid response structure
        """
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Say hello in a brief response."}],
            "temperature": 0,
            "max_tokens": 50,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/chat/completions"
        response = requests.post(url, json=payload, headers=headers)

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        # Basic validation
        self._validate_chat_completion_response(data, strict=True)

        # Ensure content exists
        content = data["choices"][0]["message"]["content"]
        assert len(content.strip()) > 0, "Response content is empty"

    def test_azure_chat_completions_malformed_payload(self):
        """
        /chat/completions with malformed payload
        Input: messages as string instead of array
        Output: Error response
        """
        payload = {
            "model": "gpt-4",
            "messages": "not-a-list",  # INTENTIONALLY WRONG
            "temperature": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/chat/completions"
        response = requests.post(url, json=payload, headers=headers)

        # Should return an error status code
        assert (
            response.status_code >= 400
        ), f"Expected error status code, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_azure_chat_completions_streaming(self):
        """
        /chat/completions with stream=True
        Input: valid model, messages, stream=True
        Output: multiple streaming chunks (not single response)
        """
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "List three colors, one per line."}
            ],
            "temperature": 0,
            "stream": True,
            "max_tokens": 50,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/chat/completions"
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

    def test_azure_chat_completions_tool_calling(self):
        """
        /chat/completions with tool/function calling
        Input: valid model, messages, tools field
        Output: 200, function calls or direct response
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
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
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
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/chat/completions"
        response = requests.post(url, json=payload, headers=headers)

        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()

        self._validate_chat_completion_response(
            data, strict=False
        )  # Allow tool_calls without content

        message = data["choices"][0]["message"]

        # Should have either tool_calls or content
        has_tool_calls = "tool_calls" in message and message["tool_calls"]
        has_content = "content" in message and message["content"]

        assert (
            has_tool_calls or has_content
        ), "Response should have either tool_calls or content"

    def test_azure_chat_completions_invalid_role(self):
        """
        /chat/completions with invalid role
        Input: message with invalid role
        Output: Error response
        """
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "invalid-role", "content": "This should fail"}],
            "temperature": 0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/chat/completions"
        response = requests.post(url, json=payload, headers=headers)

        # Should return an error status code
        assert (
            response.status_code >= 400
        ), f"Expected error status code, got {response.status_code}"

        data = response.json()
        assert "error" in data, "Response should contain error field"

    def test_azure_chat_completions_error_shape(self):
        """
        /chat/completions error response structure
        Input: various invalid requests
        Output: consistent error structure
        """
        invalid_payloads = [
            {
                "model": "non-existent-model",
                "messages": [{"role": "user", "content": "test"}],
            },
            {"model": "gpt-4", "messages": []},  # empty messages
            {"model": "gpt-4", "temperature": 5000},  # invalid temperature
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-1234",
        }
        url = f"{PROXY_URL}/v1/chat/completions"

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
