"""
Unit tests for Tool Permission Guardrail (OpenAI tool_calls semantics)
"""

import os
import sys
from unittest.mock import patch

from litellm.caching.dual_cache import DualCache
import pytest

sys.path.insert(0, os.path.abspath("../../../../../.."))

from fastapi import HTTPException
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.tool_permission import (
    ToolPermissionGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
    PermissionError,
)
from litellm.types.utils import (
    ModelResponse,
    Choices,
    ChatCompletionMessageToolCall,
)


class TestToolPermissionGuardrail:
    """Test class for Tool Permission Guardrail functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_rules = [
            {"id": "allow_bash", "tool_name": "Bash", "decision": "allow"},
            {"id": "allow_github", "tool_name": "mcp__github_*", "decision": "allow"},
            {
                "id": "allow_documentation",
                "tool_name": "mcp__aws-documentation_*_documentation",
                "decision": "allow",
            },
            {"id": "deny_read", "tool_name": "Read", "decision": "deny"},
            {"id": "deny_get", "tool_name": "*_get", "decision": "deny"},
        ]

        self.guardrail = ToolPermissionGuardrail(
            guardrail_name="test-tool-permission",
            rules=self.test_rules,
            default_action="deny",
            on_disallowed_action="block",
        )

    def test_initialization(self):
        """Test guardrail initialization"""
        assert self.guardrail.guardrail_name == "test-tool-permission"
        assert len(self.guardrail.rules) == 5
        assert self.guardrail.default_action == "deny"
        assert self.guardrail.on_disallowed_action == "block"
        assert GuardrailEventHooks.post_call in (
            self.guardrail.supported_event_hooks or []
        )

    def test_pattern_matching_exact(self):
        """Test exact pattern matching"""
        assert self.guardrail._matches_pattern("Read", "Read") is True
        assert self.guardrail._matches_pattern("Write", "Read") is False

    def test_pattern_matching_wildcards(self):
        """Test wildcard pattern matching"""
        assert (
            self.guardrail._matches_pattern(
                "mcp__github_add_issue_comment", "mcp__github_*"
            )
            is True
        )
        assert (
            self.guardrail._matches_pattern(
                "mcp__github_add_issue_comment", "mcp__github_*_comment"
            )
            is True
        )
        assert (
            self.guardrail._matches_pattern(
                "mcp__github_add_issue_comment", "*_comment"
            )
            is True
        )
        assert (
            self.guardrail._matches_pattern(
                "mcp__git_add_issue_comment", "mcp__github_*"
            )
            is False
        )
        assert (
            self.guardrail._matches_pattern(
                "mcp__github_assign_copilot_to_issue", "mcp__github_*_comment"
            )
            is False
        )
        assert (
            self.guardrail._matches_pattern(
                "mcp__github_assign_copilot_to_issue", "*_comment"
            )
            is False
        )

    def test_check_tool_permission_allow(self):
        is_allowed, rule_id, msg = self.guardrail._check_tool_permission("Bash")
        assert is_allowed is True
        assert rule_id == "allow_bash"
        assert "allowed" in (msg or "")

        is_allowed, rule_id, _ = self.guardrail._check_tool_permission(
            "mcp__github_add_issue_comment"
        )
        assert is_allowed is True
        assert rule_id == "allow_github"

    def test_check_tool_permission_deny(self):
        is_allowed, rule_id, msg = self.guardrail._check_tool_permission("Read")
        assert is_allowed is False
        assert rule_id == "deny_read"
        assert "denied" in (msg or "")

        is_allowed, rule_id, msg = self.guardrail._check_tool_permission("UnknownTool")
        assert is_allowed is False
        assert rule_id is None
        assert "default" in (msg or "")

    def test_check_tool_permission_custom_template(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="custom-template",
            rules=self.test_rules,
            default_action="deny",
            violation_message_template="custom {tool_name} {rule_id} :: {default_message}",
        )

        _, rule_id, message = guardrail._check_tool_permission("Read")
        assert rule_id == "deny_read"
        assert message.startswith("custom Read deny_read")
        assert "Tool 'Read' denied" in message

        _, rule_id, message = guardrail._check_tool_permission("UnknownTool")
        assert rule_id is None
        assert message.startswith("custom UnknownTool None")
        assert "Tool 'UnknownTool' denied by default action" in message

    def test_extract_tool_calls_openai_format(self):
        tool_call = {
            "id": "call_123",
            "function": {
                "name": "Read",
                "arguments": '{"file_path": "/test/file.txt"}',
            },
            "type": "function",
        }
        response = ModelResponse(
            choices=[
                Choices(
                    message={
                        "tool_calls": [tool_call],
                    }
                )
            ]
        )

        tool_calls = self.guardrail._extract_tool_calls_from_response(response)
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ChatCompletionMessageToolCall)
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].function.name == "Read"

    def test_extract_tool_calls_empty_response(self):
        response = ModelResponse(choices=[])
        tool_calls = self.guardrail._extract_tool_calls_from_response(response)
        assert len(tool_calls) == 0

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_no_tools(self):
        response = ModelResponse(choices=[Choices(message={})])
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["test-tool-permission"]}

        with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
            result = await self.guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_with_allowed_tools(self):
        tool_call = {
            "function": {"name": "Bash", "arguments": "{}"},
            "type": "function",
        }
        response = ModelResponse(choices=[Choices(message={"tool_calls": [tool_call]})])
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["test-tool-permission"]}

        with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
            result = await self.guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_with_denied_tools_raises(self):
        tool_call = {
            "function": {"name": "Read", "arguments": "{}"},
            "type": "function",
        }
        response = ModelResponse(choices=[Choices(message={"tool_calls": [tool_call]})])
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["test-tool-permission"]}

        with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await self.guardrail.async_post_call_success_hook(
                    data=data, user_api_key_dict=user_api_key_dict, response=response
                )

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_block_mode(self):
        data = {
            "tools": [
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "Read"}},
            ]
        }
        user_api_key_dict = UserAPIKeyAuth()
        cache = DualCache(default_in_memory_ttl=1)

        with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(HTTPException) as excinfo:
                await self.guardrail.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=cache,
                    data=data,
                    call_type="completion",
                )
        assert excinfo.value.status_code == 400

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_uses_custom_template(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="custom-template",
            rules=self.test_rules,
            default_action="deny",
            on_disallowed_action="block",
            violation_message_template="blocked {tool_name} by policy",
        )

        data = {
            "tools": [
                {"type": "function", "function": {"name": "Read"}},
            ]
        }
        user_api_key_dict = UserAPIKeyAuth()
        cache = DualCache(default_in_memory_ttl=1)

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(HTTPException) as excinfo:
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=user_api_key_dict,
                    cache=cache,
                    data=data,
                    call_type="completion",
                )

        assert excinfo.value.status_code == 400
        assert (
            excinfo.value.detail.get("detection_message")
            == "blocked Read by policy"
        )

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_rewrite_mode(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="test-tool-permission",
            rules=self.test_rules,
            default_action="deny",
            on_disallowed_action="rewrite",
        )
        data = {
            "tools": [
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "Read"}},
            ]
        }
        user_api_key_dict = UserAPIKeyAuth()
        cache = DualCache(default_in_memory_ttl=1)

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            new_data = await guardrail.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type="completion",
            )

        assert isinstance(new_data, dict)
        assert "tools" in new_data
        tool_names = [t["function"]["name"] for t in new_data["tools"]]
        assert "Bash" in tool_names
        assert "Read" not in tool_names

    def test_modify_response_with_permission_errors(self):
        # Setup a response with one tool_call
        tool_call = ChatCompletionMessageToolCall(
            function={"name": "Read", "arguments": "{}"}, id="call_123"
        )
        response = ModelResponse(
            choices=[Choices(message={"tool_calls": [tool_call], "content": ""})]
        )

        # Denied tools tuple of (tool_call, PermissionError)
        denied_tools = [
            (
                tool_call,
                PermissionError(
                    tool_name="Read",
                    rule_id="deny_read",
                    message="Tool 'Bash' denied by rule 'deny_read'",
                ),
            )
        ]

        # Apply modifications
        self.guardrail._modify_response_with_permission_errors(response, denied_tools)

        # Verify: tool_calls removed and content contains error message
        choice = response.choices[0]
        assert isinstance(choice, Choices)
        assert choice.message.tool_calls is None or choice.message.tool_calls == []
        assert isinstance(choice.message.content, str)
        assert "Permission denied" in choice.message.content


class TestToolPermissionGuardrailIntegration:
    """Integration tests for Tool Permission Guardrail"""

    def test_default_action_allow(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="test-allow-default",
            rules=[{"id": "deny_read", "tool_name": "Read", "decision": "deny"}],
            default_action="allow",
        )

        is_allowed, rule_id, message = guardrail._check_tool_permission("UnknownTool")
        assert is_allowed is True
        assert rule_id is None
        assert "default" in (message or "")

        is_allowed, rule_id, _ = guardrail._check_tool_permission("Read")
        assert is_allowed is False
        assert rule_id == "deny_read"

    def test_empty_rules(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="test-no-rules",
            rules=[],
            default_action="allow",
        )

        is_allowed, rule_id, message = guardrail._check_tool_permission("AnyTool")
        assert is_allowed is True
        assert rule_id is None
        assert "default" in (message or "")
