"""
Integration tests for OpenAI custom tool types (GPT-5.x+).

These tests verify that custom tools work correctly with OpenAI's API.
Tests are skipped if OPENAI_API_KEY is not set.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest

import litellm


class TestOpenAICustomTools:
    """Integration tests for OpenAI custom tool types"""

    def test_custom_tool_with_gpt5(self):
        """
        Test that custom tools can be passed to GPT-5.x models.

        Custom tools are a GPT-5.x+ feature that allows free-form text
        inputs/outputs instead of JSON schema parameters.
        """
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        # Custom tool definition
        custom_tool = {
            "type": "custom",
            "custom": {
                "name": "code_exec",
                "description": "Executes arbitrary python code and returns the result"
            }
        }

        # Also include a standard function tool to test mixed tool lists
        function_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }

        try:
            response = litellm.completion(
                model="gpt-5.2",
                messages=[
                    {
                        "role": "user",
                        "content": "What is 2 + 2? You can use the code_exec tool to calculate this."
                    }
                ],
                tools=[custom_tool, function_tool],
                tool_choice="auto"
            )

            # Verify we got a response
            assert response is not None
            assert response.choices is not None
            assert len(response.choices) > 0

            # The model should either respond with text or make a tool call
            choice = response.choices[0]
            assert choice.message is not None

            # Log for debugging
            print(f"Response: {response}")
            print(f"Message: {choice.message}")
            if choice.message.tool_calls:
                print(f"Tool calls: {choice.message.tool_calls}")

        except litellm.BadRequestError as e:
            # If the model doesn't support custom tools, that's expected for older models
            # But for GPT-5.2, it should work
            if "custom" in str(e).lower() or "tool" in str(e).lower():
                pytest.fail(f"GPT-5.2 should support custom tools but got error: {e}")
            raise

    def test_custom_tool_only(self):
        """
        Test passing only a custom tool (no function tools).
        """
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        custom_tool = {
            "type": "custom",
            "custom": {
                "name": "python_interpreter",
                "description": "Executes Python code and returns stdout/stderr"
            }
        }

        try:
            response = litellm.completion(
                model="gpt-5.2",
                messages=[
                    {
                        "role": "user",
                        "content": "Hello, can you help me with a simple task?"
                    }
                ],
                tools=[custom_tool]
            )

            assert response is not None
            assert response.choices is not None
            assert len(response.choices) > 0

            print(f"Response with custom tool only: {response}")

        except litellm.BadRequestError as e:
            if "custom" in str(e).lower():
                pytest.fail(f"GPT-5.2 should support custom tools but got error: {e}")
            raise

    @pytest.mark.asyncio
    async def test_custom_tool_async(self):
        """
        Test custom tools with async completion.
        """
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        custom_tool = {
            "type": "custom",
            "custom": {
                "name": "code_sandbox",
                "description": "Run code in a sandboxed environment"
            }
        }

        try:
            response = await litellm.acompletion(
                model="gpt-5.2",
                messages=[
                    {
                        "role": "user",
                        "content": "What's the result of 10 * 5?"
                    }
                ],
                tools=[custom_tool]
            )

            assert response is not None
            assert response.choices is not None
            assert len(response.choices) > 0

            print(f"Async response: {response}")

        except litellm.BadRequestError as e:
            if "custom" in str(e).lower():
                pytest.fail(f"GPT-5.2 should support custom tools but got error: {e}")
            raise

    def test_custom_tool_streaming(self):
        """
        Test custom tools with streaming completion.
        """
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        custom_tool = {
            "type": "custom",
            "custom": {
                "name": "calculator",
                "description": "Perform mathematical calculations"
            }
        }

        try:
            response = litellm.completion(
                model="gpt-5.2",
                messages=[
                    {
                        "role": "user",
                        "content": "Say hello"
                    }
                ],
                tools=[custom_tool],
                stream=True
            )

            # Collect streaming chunks
            chunks = list(response)
            assert len(chunks) > 0

            print(f"Streaming chunks count: {len(chunks)}")
            for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks
                print(f"Chunk {i}: {chunk}")

        except litellm.BadRequestError as e:
            if "custom" in str(e).lower():
                pytest.fail(f"GPT-5.2 should support custom tools but got error: {e}")
            raise


class TestOpenAIBuiltinTools:
    """Integration tests for OpenAI built-in tools via Responses API"""

    def test_web_search_tool_responses_api(self):
        """
        Test web_search tool with the Responses API.
        """
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            response = litellm.responses(
                model="gpt-4o",
                input="What is the current weather in San Francisco?",
                tools=[{"type": "web_search_preview"}]
            )

            assert response is not None
            print(f"Web search response: {response}")

        except litellm.BadRequestError as e:
            # web_search may not be available on all models/accounts
            print(f"Web search not available: {e}")
            pytest.skip(f"Web search tool not available: {e}")

    def test_file_search_tool_responses_api(self):
        """
        Test file_search tool structure with the Responses API.
        Note: This test verifies the tool is passed correctly,
        but may fail if no vector store is configured.
        """
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        try:
            response = litellm.responses(
                model="gpt-4o",
                input="Hello",
                tools=[{"type": "file_search"}]
            )

            assert response is not None
            print(f"File search response: {response}")

        except litellm.BadRequestError as e:
            # file_search requires a vector store to be configured
            print(f"File search error (expected if no vector store): {e}")
            pytest.skip(f"File search requires vector store configuration: {e}")
