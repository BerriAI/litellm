"""
Base test class for Anthropic Messages API input_examples E2E tests.

Tests that input_examples works correctly via litellm.anthropic.messages interface
by making actual API calls and validating that the beta headers are correctly passed.

Supported providers:
- Anthropic API: advanced-tool-use-2025-11-20
- Microsoft Foundry: advanced-tool-use-2025-11-20
- Vertex AI: tool-examples-2025-10-29 (all models)
- Bedrock Invoke: tool-examples-2025-10-29 (Claude Opus 4.5 only)

Reference: https://docs.anthropic.com/en/docs/build-with-claude/tool-use#providing-tool-use-examples
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
import litellm


def get_weather_tool_with_input_examples() -> Dict[str, Any]:
    """
    Returns a tool with input_examples field.
    This demonstrates how to provide examples for tool inputs.
    """
    return {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature"
                }
            },
            "required": ["location"]
        },
        "input_examples": [
            {
                "location": "San Francisco, CA",
                "unit": "fahrenheit"
            },
            {
                "location": "Tokyo, Japan",
                "unit": "celsius"
            },
            {
                "location": "New York, NY"  # unit is optional
            }
        ]
    }


def get_calculate_tool_with_input_examples() -> Dict[str, Any]:
    """
    Returns a calculator tool with input_examples field.
    """
    return {
        "name": "calculate",
        "description": "Perform basic arithmetic operations",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                }
            },
            "required": ["operation", "a", "b"]
        },
        "input_examples": [
            {
                "operation": "add",
                "a": 5,
                "b": 3
            },
            {
                "operation": "multiply",
                "a": 10,
                "b": 2
            }
        ]
    }


class BaseAnthropicMessagesInputExamplesTest(ABC):
    """
    Base test class for input_examples E2E tests across different providers.
    
    Subclasses must implement:
    - get_model(): Returns the model string to use for tests
    - get_extra_headers(): Returns extra headers (including anthropic-beta)
    
    Tests validate that input_examples are correctly handled and the appropriate
    beta headers are passed to downstream providers.
    """

    @abstractmethod
    def get_model(self) -> str:
        """
        Returns the model string to use for tests.
        
        Examples:
        - "anthropic/claude-sonnet-4-5-20250929"
        - "vertex_ai/claude-opus-4-5@20251101"
        - "bedrock/invoke/us.anthropic.claude-opus-4-5-20251101-v1:0"
        """
        pass

    @abstractmethod
    def get_extra_headers(self) -> Dict[str, str]:
        """
        Returns extra headers to pass with the request.
        Includes the anthropic-beta header for input examples.
        
        Different providers use different beta headers:
        - Anthropic API: "advanced-tool-use-2025-11-20"
        - Bedrock: "tool-examples-2025-10-29" (auto-injected by LiteLLM)
        - Vertex AI: "tool-examples-2025-10-29" (auto-injected by LiteLLM)
        """
        pass

    def get_tools_with_input_examples(self) -> List[Dict[str, Any]]:
        """
        Returns tools list with input_examples.
        """
        return [
            get_weather_tool_with_input_examples(),
            get_calculate_tool_with_input_examples()
        ]

    @pytest.mark.asyncio
    async def test_input_examples_with_calculation(self):
        """
        E2E test: Input examples should work with different tool types.
        
        This validates that the model can use the calculate tool with
        input_examples correctly.
        """
        litellm._turn_on_debug()
        
        tools = self.get_tools_with_input_examples()
        messages = [
            {
                "role": "user",
                "content": "What is 15 plus 27? Use the calculate tool."
            }
        ]
        
        response = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            tools=tools,
            max_tokens=1024,
            extra_headers=self.get_extra_headers(),
        )
        
        print(f"Response: {json.dumps(response, indent=2, default=str)}")
        
        # Validate response
        assert "content" in response, "Response should contain content"
        
        content = response.get("content", [])
        tool_uses = [block for block in content if block.get("type") == "tool_use"]
        
        # If the model decides to use a tool, it should be calculate
        if tool_uses:
            tool_names = [t.get("name") for t in tool_uses]
            print(f"Tools used: {tool_names}")
            
            # Check if calculate was used
            calc_tool_uses = [t for t in tool_uses if t.get("name") == "calculate"]
            if calc_tool_uses:
                tool_input = calc_tool_uses[0].get("input", {})
                print(f"Calculate tool input: {tool_input}")
                # Validate the input has required fields
                assert "operation" in tool_input, "Tool input should have operation"
                assert "a" in tool_input, "Tool input should have a"
                assert "b" in tool_input, "Tool input should have b"
