"""
Test for OpenRouter + GPT-5.4/GPT-5.3-Codex tool calls with cache_control.

This test verifies the fix for Issue #23803:
- cache_control is properly stripped from tools for models that don't support it
- Tools maintain valid structure (type: "function" and valid function object)
- No malformed tool entries are created

Related Issue: https://github.com/BerriAI/litellm/issues/23803
"""
import os
import sys
import copy

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.openrouter.chat.transformation import OpenrouterConfig


class TestOpenRouterToolCallsWithCacheControl:
    """Tests for tool handling with cache_control for different models."""

    def test_tools_cache_control_stripped_for_gpt54(self):
        """
        Test that cache_control is stripped from tools for GPT-5.4.
        
        Issue #23803: OpenRouter + GPT-5.4 tool calls were failing because
        cache_control was not being stripped from tools for models that
        don't support it, causing malformed tool entries.
        """
        config = OpenrouterConfig()
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                },
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time information",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                },
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Calculate expression",
                    "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search information",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
                }
            }
        ]
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"tools": copy.deepcopy(tools)}
        
        result = config.transform_request(
            model="openrouter/openai/gpt-5.4",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Verify all tools are present and valid
        assert "tools" in result
        assert len(result["tools"]) == 4
        
        for i, tool in enumerate(result["tools"]):
            # Each tool must have type="function"
            assert tool.get("type") == "function", f"Tool {i}: type should be 'function', got {tool.get('type')}"
            # Each tool must have a valid function object
            assert isinstance(tool.get("function"), dict), f"Tool {i}: function should be a dict, got {type(tool.get('function'))}"
            assert "name" in tool["function"], f"Tool {i}: function should have 'name'"
            # cache_control should be stripped for GPT-5.4
            assert "cache_control" not in tool, f"Tool {i}: cache_control should be stripped for GPT-5.4"

    def test_tools_cache_control_stripped_for_gpt53_codex(self):
        """
        Test that cache_control is stripped from tools for GPT-5.3-Codex.
        
        Issue #23803 also affected GPT-5.3-Codex models.
        """
        config = OpenrouterConfig()
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "code_tool",
                    "description": "Execute code",
                    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}
                },
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"tools": copy.deepcopy(tools)}
        
        result = config.transform_request(
            model="openrouter/openai/gpt-5.3-codex",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0].get("type") == "function"
        assert isinstance(result["tools"][0].get("function"), dict)
        assert "cache_control" not in result["tools"][0]

    def test_tools_cache_control_preserved_for_claude(self):
        """
        Test that cache_control is preserved for Claude models that support it.
        """
        config = OpenrouterConfig()
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                },
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"tools": copy.deepcopy(tools)}
        
        result = config.transform_request(
            model="openrouter/anthropic/claude-3.5-sonnet",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        assert "tools" in result
        assert len(result["tools"]) == 1
        # cache_control should be preserved for Claude models
        assert result["tools"][0].get("cache_control") == {"type": "ephemeral"}

    def test_tools_mixed_with_and_without_cache_control(self):
        """
        Test handling of tools where some have cache_control and some don't.
        
        This was the specific scenario reported in Issue #23803 where
        tool at index 3 (the 4th tool) was malformed.
        """
        config = OpenrouterConfig()
        
        # Mix of tools with and without cache_control
        tools = [
            {
                "type": "function",
                "function": {"name": "tool0", "description": "Tool 0"},
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "function",
                "function": {"name": "tool1", "description": "Tool 1"}
            },
            {
                "type": "function",
                "function": {"name": "tool2", "description": "Tool 2"}
            },
            {
                "type": "function",
                "function": {"name": "tool3", "description": "Tool 3"}
            }
        ]
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"tools": copy.deepcopy(tools)}
        
        result = config.transform_request(
            model="openrouter/openai/gpt-5.4",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Verify all 4 tools are valid - this was failing before the fix
        for i, tool in enumerate(result["tools"]):
            assert tool.get("type") == "function", f"Tool {i}: type should be 'function'"
            assert isinstance(tool.get("function"), dict), f"Tool {i}: function should be a dict"
            assert "name" in tool["function"], f"Tool {i}: function.name is required"
            # cache_control should be stripped for all tools for GPT-5.4
            assert "cache_control" not in tool, f"Tool {i}: cache_control should be stripped"

    def test_empty_tools_array(self):
        """Test handling of empty tools array."""
        config = OpenrouterConfig()
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {"tools": []}
        
        result = config.transform_request(
            model="openrouter/openai/gpt-5.4",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        assert "tools" in result
        assert result["tools"] == []

    def test_no_tools_parameter(self):
        """Test handling when no tools parameter is provided."""
        config = OpenrouterConfig()
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {}
        
        result = config.transform_request(
            model="openrouter/openai/gpt-5.4",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        assert "tools" not in result

    def test_extra_body_tools_should_not_overwrite_processed_tools(self):
        """
        Test that tools in extra_body don't overwrite correctly processed tools.
        
        Issue #23803: When extra_body contains a 'tools' key with malformed/invalid
        tools, it was overwriting the correctly processed tools from the parent
        transform_request method. This caused tool calls to fail for OpenRouter
        with 'expected function' errors.
        
        This test verifies that even if extra_body contains malformed tools,
        the correctly processed tools are preserved.
        """
        config = OpenrouterConfig()
        
        # Valid tools with cache_control that should be processed
        valid_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                },
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        # Malformed tools in extra_body that should NOT overwrite processed tools
        malformed_tools = [
            {
                # Missing required fields - this would cause "expected function" error
                "type": "not_function",
                "function": None
            }
        ]
        
        messages = [{"role": "user", "content": "Hello"}]
        optional_params = {
            "tools": copy.deepcopy(valid_tools),
            "extra_body": {"tools": malformed_tools}
        }
        
        result = config.transform_request(
            model="openrouter/openai/gpt-5.4",
            messages=messages,
            optional_params=optional_params,
            litellm_params={},
            headers={}
        )
        
        # Verify the correctly processed tools are preserved, not the malformed ones
        assert "tools" in result
        assert len(result["tools"]) == 1
        # Should have the processed tool, not the malformed one from extra_body
        assert result["tools"][0].get("type") == "function"
        assert isinstance(result["tools"][0].get("function"), dict)
        assert result["tools"][0]["function"].get("name") == "get_weather"
        # cache_control should be stripped since GPT-5.4 doesn't support it
        assert "cache_control" not in result["tools"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
