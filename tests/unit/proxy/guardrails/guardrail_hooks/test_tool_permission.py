"""
Unit tests for Tool Permission Guardrail (OpenAI tool_calls semantics)
"""

import copy
import json
import logging
import re
from itertools import chain
from typing import Final, Literal
from unittest.mock import patch

import pytest

from litellm.caching.dual_cache import DualCache


from fastapi import HTTPException

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.tool_permission import (
    ToolPermissionGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
    PermissionError,
)
from litellm.types.utils import (
    CallTypesLiteral,
    ChatCompletionCustomToolCallPayload,
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Function,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
)


class TestToolPermissionGuardrail:
    """Test class for Tool Permission Guardrail functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.test_rules = [
            {"id": "allow_bash", "tool_name": r"^Bash$", "decision": "allow"},
            {
                "id": "allow_github",
                "tool_name": r"^mcp__github_.*$",
                "decision": "allow",
            },
            {
                "id": "allow_documentation",
                "tool_name": r"^mcp__aws-documentation_.*_documentation$",
                "decision": "allow",
            },
            {"id": "deny_read", "tool_name": r"^Read$", "decision": "deny"},
            {"id": "deny_get", "tool_name": r".*_get$", "decision": "deny"},
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

    def test_matches_regex_helper(self):
        pattern = re.compile(r"^Read$")
        assert self.guardrail._matches_regex(pattern, "Read") is True
        assert self.guardrail._matches_regex(pattern, "Write") is False
        assert self.guardrail._matches_regex(None, "Any") is True
        assert self.guardrail._matches_regex(pattern, None) is False

    def test_rule_matches_tool_with_type_only(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="type-only",
            rules=[
                {
                    "id": "allow_functions",
                    "tool_type": r"^function$",
                    "decision": "allow",
                }
            ],
            default_action="deny",
            on_disallowed_action="block",
        )

        is_allowed, rule_id, _ = guardrail._check_tool_permission("AnyTool", "function")
        assert is_allowed is True
        assert rule_id == "allow_functions"

        is_allowed, rule_id, _ = guardrail._check_tool_permission("AnyTool", "custom")
        assert is_allowed is False
        assert rule_id is None

    def test_rule_matches_tool_with_name_and_type(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="name-type",
            rules=[
                {
                    "id": "allow_specific",
                    "tool_name": r"^Bash$",
                    "tool_type": r"^function$",
                    "decision": "allow",
                }
            ],
            default_action="deny",
            on_disallowed_action="block",
        )

        is_allowed, rule_id, _ = guardrail._check_tool_permission("Bash", "function")
        assert is_allowed is True
        assert rule_id == "allow_specific"

        is_allowed, rule_id, _ = guardrail._check_tool_permission("Bash", "custom")
        assert is_allowed is False
        assert rule_id is None

    def test_rule_requires_name_or_type(self):
        with pytest.raises(ValueError, match='validation error for ToolPermissionRule'):
            ToolPermissionGuardrail(
                guardrail_name="invalid-rule",
                rules=[{"id": "no_target", "decision": "allow"}],
                default_action="deny",
                on_disallowed_action="block",
            )

    def test_type_only_rule_skips_param_patterns(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="type-param",
            rules=[
                {
                    "id": "allow_type_only",
                    "tool_type": r"^function$",
                    "decision": "allow",
                    "allowed_param_patterns": {"foo": r"^bar$"},
                }
            ],
            default_action="deny",
            on_disallowed_action="block",
        )

        tool_call = ChatCompletionMessageToolCall(
            function={"name": "AnyTool", "arguments": "{}"},
            type="function",
        )

        is_allowed, rule_id, _ = guardrail._get_permission_for_tool_call(tool_call)
        assert is_allowed is True
        assert rule_id == "allow_type_only"

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

    def test_extract_tool_calls_legacy_function_call_format(self):
        response = ModelResponse(
            choices=[
                Choices(
                    message={
                        "function_call": {
                            "name": "Read",
                            "arguments": '{"file_path": "/test/file.txt"}',
                        },
                    }
                )
            ]
        )

        tool_calls = self.guardrail._extract_tool_calls_from_response(response)
        assert len(tool_calls) == 1
        assert isinstance(tool_calls[0], ChatCompletionMessageToolCall)
        assert tool_calls[0].id == "legacy_function_call_0"
        assert tool_calls[0].function.name == "Read"
        assert tool_calls[0].function.arguments == '{"file_path": "/test/file.txt"}'

    def test_extract_tool_calls_empty_response(self):
        response = ModelResponse(choices=[])
        tool_calls = self.guardrail._extract_tool_calls_from_response(response)
        assert len(tool_calls) == 0

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_no_tools(self):
        """Test that async_post_call_success_hook returns response when no tool calls are present."""
        response = ModelResponse(choices=[Choices(message={})])
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["test-tool-permission"]}

        with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
            result = await self.guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )
        assert result is response

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_with_allowed_tools(self):
        """Test that async_post_call_success_hook returns response when tool calls are allowed."""
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
        assert result is response

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
    async def test_async_post_call_success_hook_with_denied_legacy_function_call_raises(
        self,
    ):
        response = ModelResponse(
            choices=[
                Choices(
                    message={
                        "function_call": {
                            "name": "Read",
                            "arguments": "{}",
                        },
                    }
                )
            ]
        )
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["test-tool-permission"]}

        with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await self.guardrail.async_post_call_success_hook(
                    data=data, user_api_key_dict=user_api_key_dict, response=response
                )

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_param_patterns_allow(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="mail-guardrail",
            rules=[
                {
                    "id": "allow_mail",
                    "tool_name": r"^mail_mcp-send_email$",
                    "decision": "allow",
                    "allowed_param_patterns": {
                        "to[]": r"^.+@berri\.ai$",
                        "subject": r"^.{1,120}$",
                    },
                }
            ],
            default_action="deny",
            on_disallowed_action="block",
        )

        tool_call = {
            "function": {
                "name": "mail_mcp-send_email",
                "arguments": '{"to": ["owner@berri.ai"], "subject": "Hi"}',
            },
            "type": "function",
        }
        response = ModelResponse(choices=[Choices(message={"tool_calls": [tool_call]})])
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["mail-guardrail"]}

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            await guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_param_patterns_block(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="mail-guardrail",
            rules=[
                {
                    "id": "allow_mail",
                    "tool_name": r"^mail_mcp-send_email$",
                    "decision": "allow",
                    "allowed_param_patterns": {"to[]": r"^.+@berri\.ai$"},
                }
            ],
            default_action="deny",
            on_disallowed_action="block",
        )

        tool_call = {
            "function": {
                "name": "mail_mcp-send_email",
                "arguments": '{"to": ["intruder@example.com"]}',
            },
            "type": "function",
        }
        response = ModelResponse(choices=[Choices(message={"tool_calls": [tool_call]})])
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["mail-guardrail"]}

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await guardrail.async_post_call_success_hook(
                    data=data, user_api_key_dict=user_api_key_dict, response=response
                )

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_param_patterns_rewrite(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="mail-guardrail",
            rules=[
                {
                    "id": "allow_mail",
                    "tool_name": r"^mail_mcp-send_email$",
                    "decision": "allow",
                    "allowed_param_patterns": {"to[]": r"^.+@berri\.ai$"},
                }
            ],
            default_action="deny",
            on_disallowed_action="rewrite",
        )

        tool_call = {
            "id": "call_berri",
            "function": {
                "name": "mail_mcp-send_email",
                "arguments": '{"to": ["visitor@example.com"]}',
            },
            "type": "function",
        }
        response = ModelResponse(choices=[Choices(message={"tool_calls": [tool_call]})])
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["mail-guardrail"]}

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            await guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=user_api_key_dict, response=response
            )

        choice = response.choices[0]
        assert isinstance(choice, Choices)
        assert not choice.message.tool_calls
        assert isinstance(choice.message.content, str)
        assert "berri" in choice.message.content

    @pytest.mark.asyncio
    async def test_rewrite_denies_only_the_choice_whose_call_is_denied_when_choices_share_a_call_id(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="choice-guardrail",
            rules=[{"id": "deny_bash", "tool_name": "Bash", "decision": "deny"}],
            default_action="allow",
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
        )
        response = ModelResponse(
            choices=[
                Choices(
                    index=index,
                    finish_reason="tool_calls",
                    message={
                        "tool_calls": [
                            {"id": "call_shared", "type": "function", "function": {"name": name, "arguments": "{}"}}
                        ]
                    },
                )
                for index, name in enumerate(("Bash", "Read"))
            ]
        )

        await guardrail.async_post_call_success_hook(
            data={"guardrails": ["choice-guardrail"]}, user_api_key_dict=UserAPIKeyAuth(), response=response
        )

        kept: Final = [
            [tool_call.function.name for tool_call in choice.message.tool_calls or ()]
            for choice in response.choices
            if isinstance(choice, Choices)
        ]
        assert kept == [[], ["Read"]]

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_missing_arguments_blocks_param_rule(
        self,
    ):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="mail-guardrail",
            rules=[
                {
                    "id": "deny_gmail",
                    "tool_name": r"^mail_mcp-send_email$",
                    "decision": "deny",
                    "allowed_param_patterns": {"to[]": r"^.+@gmail\.com$"},
                }
            ],
            default_action="allow",
            on_disallowed_action="block",
        )

        tool_call = {
            "function": {
                "name": "mail_mcp-send_email",
            },
            "type": "function",
        }
        response = ModelResponse(choices=[Choices(message={"tool_calls": [tool_call]})])
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["mail-guardrail"]}

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await guardrail.async_post_call_success_hook(
                    data=data, user_api_key_dict=user_api_key_dict, response=response
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "arguments",
        [
            "{not-json",
            '["owner@berri.ai"]',
        ],
    )
    async def test_async_post_call_success_hook_malformed_arguments_blocks_param_rule(
        self, arguments
    ):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="mail-guardrail",
            rules=[
                {
                    "id": "deny_gmail",
                    "tool_name": r"^mail_mcp-send_email$",
                    "decision": "deny",
                    "allowed_param_patterns": {"to[]": r"^.+@gmail\.com$"},
                }
            ],
            default_action="allow",
            on_disallowed_action="block",
        )

        tool_call = {
            "function": {
                "name": "mail_mcp-send_email",
                "arguments": arguments,
            },
            "type": "function",
        }
        response = ModelResponse(choices=[Choices(message={"tool_calls": [tool_call]})])
        user_api_key_dict = UserAPIKeyAuth()
        data = {"guardrails": ["mail-guardrail"]}

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await guardrail.async_post_call_success_hook(
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
    async def test_async_pre_call_hook_without_tools_logs_skip_at_debug(self, caplog):
        data = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}

        with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
            with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
                result = await self.guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(),
                    cache=DualCache(default_in_memory_ttl=1),
                    data=data,
                    call_type="completion",
                )

        assert result is data
        skip_levels = [r.levelno for r in caplog.records if "No tools or functions in data" in r.getMessage()]
        assert skip_levels == [logging.DEBUG]
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_denied_tool_logs_at_info(self, caplog):
        data = {"tools": [{"type": "function", "function": {"name": "Read"}}]}

        with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
            with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
                with pytest.raises(HTTPException):
                    await self.guardrail.async_pre_call_hook(
                        user_api_key_dict=UserAPIKeyAuth(),
                        cache=DualCache(default_in_memory_ttl=1),
                        data=data,
                        call_type="completion",
                    )

        denied_levels = [
            r.levelno
            for r in caplog.records
            if r.getMessage() == "Tool Permission Guardrail: Tool 'Read' denied by rule 'deny_read'"
        ]
        assert denied_levels == [logging.INFO]
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.asyncio
    async def test_async_post_call_success_hook_denied_tool_logs_at_info(self, caplog):
        tool_call = {"function": {"name": "Read", "arguments": "{}"}, "type": "function"}
        response = ModelResponse(choices=[Choices(message={"tool_calls": [tool_call]})])

        with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
            with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
                with pytest.raises(GuardrailRaisedException):
                    await self.guardrail.async_post_call_success_hook(
                        data={"guardrails": ["test-tool-permission"]},
                        user_api_key_dict=UserAPIKeyAuth(),
                        response=response,
                    )

        denied_levels = [
            r.levelno
            for r in caplog.records
            if r.getMessage() == "Tool Permission Guardrail: Tool 'Read' denied by rule 'deny_read'"
        ]
        assert denied_levels == [logging.INFO]
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []

    def test_parse_tool_call_arguments_malformed_json_logs_warning(self, caplog):
        tool_call = ChatCompletionMessageToolCall(function={"name": "Bash", "arguments": "{not json"}, id="call_1")

        with caplog.at_level(logging.DEBUG, logger="LiteLLM Proxy"):
            parsed, error = self.guardrail._parse_tool_call_arguments(tool_call)

        assert parsed is None
        assert error == "arguments could not be parsed"
        warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_messages) == 1
        assert warning_messages[0].startswith("Tool Permission Guardrail: Failed to decode arguments for tool Bash")

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_blocks_legacy_functions(self):
        data = {
            "functions": [
                {"name": "Bash", "description": "allowed"},
                {"name": "Read", "description": "denied"},
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
    async def test_async_pre_call_hook_blocks_named_legacy_function_call(self):
        data = {
            "functions": [{"name": "Bash"}],
            "function_call": {"name": "Read"},
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
    async def test_async_pre_call_hook_blocks_named_tool_choice(self):
        data = {
            "tools": [{"type": "function", "function": {"name": "Bash"}}],
            "tool_choice": {"type": "function", "function": {"name": "Read"}},
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
        assert excinfo.value.detail.get("detection_message") == "blocked Read by policy"

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

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_rewrite_mode_filters_legacy_functions(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="test-tool-permission",
            rules=self.test_rules,
            default_action="deny",
            on_disallowed_action="rewrite",
        )
        data = {
            "functions": [
                {"name": "Bash", "description": "allowed"},
                {"name": "Read", "description": "denied"},
            ],
            "function_call": {"name": "Read"},
            "tools": [
                {"type": "function", "function": {"name": "Bash"}},
            ],
            "tool_choice": {"type": "function", "function": {"name": "Read"}},
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
        assert [function["name"] for function in new_data["functions"]] == ["Bash"]
        assert new_data["function_call"] == "none"
        assert new_data["tool_choice"] == "none"

    @staticmethod
    def _tool_search_turn(*loaded_names: str) -> dict:
        return {
            "role": "assistant",
            "content": None,
            "provider_specific_fields": {
                "tool_search_items": [
                    {"type": "tool_search_call", "call_id": "ts_1", "execution": "client"},
                    {
                        "type": "tool_search_output",
                        "call_id": "ts_1",
                        "tools": [{"type": "function", "name": name} for name in loaded_names],
                    },
                ]
            },
        }

    def _replayed_run_shell_request(self) -> dict:
        return {
            "tools": [{"type": "function", "function": {"name": "get_weather"}}, {"type": "tool_search"}],
            "messages": [
                {"role": "user", "content": "check the weather, then clean up the disk"},
                self._tool_search_turn("get_weather"),
                self._tool_search_turn("run_shell"),
            ],
        }

    @staticmethod
    def _deny_run_shell(on_disallowed_action: str) -> ToolPermissionGuardrail:
        return ToolPermissionGuardrail(
            guardrail_name="test-tool-permission",
            rules=[{"id": "deny_shell", "tool_name": r"^run_shell$", "decision": "deny"}],
            default_action="allow",
            on_disallowed_action=on_disallowed_action,
        )

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_blocks_tool_loaded_by_replayed_tool_search(self):
        guardrail = self._deny_run_shell("block")

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(HTTPException) as excinfo:
                await guardrail.async_pre_call_hook(
                    user_api_key_dict=UserAPIKeyAuth(),
                    cache=DualCache(default_in_memory_ttl=1),
                    data=self._replayed_run_shell_request(),
                    call_type="completion",
                )

        assert excinfo.value.status_code == 400
        assert "run_shell" in json.dumps(excinfo.value.detail)

    @pytest.mark.asyncio
    async def test_async_pre_call_hook_rewrite_drops_replayed_tool_search_loading_denied_tool(self):
        guardrail = self._deny_run_shell("rewrite")
        data = self._replayed_run_shell_request()

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            new_data = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(default_in_memory_ttl=1),
                data=data,
                call_type="completion",
            )

        assert isinstance(new_data, dict)
        user_turn, weather_search, shell_search = new_data["messages"]
        assert user_turn == {"role": "user", "content": "check the weather, then clean up the disk"}
        assert weather_search == self._tool_search_turn("get_weather")
        assert shell_search == {"role": "assistant", "content": None, "provider_specific_fields": {}}
        assert [tool.get("function", {}).get("name") for tool in new_data["tools"]] == ["get_weather", None]

    @pytest.mark.asyncio
    async def test_async_post_call_streaming_iterator_hook_plain_text_yields_chunks(
        self,
    ):
        """Regression test: hook must re-emit chunks when LLM replies with plain text.

        Before the fix, the `if not tool_calls:` branch did a bare `return` inside
        the async generator, which yielded nothing.  Clients received only
        `data: [DONE]` with no content.
        """
        text_chunk = ModelResponseStream(
            id="chatcmpl-plain-text",
            created=1700000000,
            model="gpt-4",
            object="chat.completion.chunk",
            choices=[],
        )

        async def _fake_stream():
            yield text_chunk

        assembled = ModelResponse(
            choices=[Choices(message={"content": "Hello, world!"})]
        )

        with patch("litellm.main.stream_chunk_builder", return_value=assembled):
            chunks = []
            async for chunk in self.guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                response=_fake_stream(),
                request_data={},
            ):
                chunks.append(chunk)

        assert len(chunks) >= 1, (
            "Hook must yield at least one chunk for plain-text responses; "
            "got none — bare return bug"
        )
        assert chunks[0].choices[0].delta.content == "Hello, world!", (
            "Hook must preserve the original response content; "
            f"got: {chunks[0].choices[0].delta.content!r}"
        )

    @pytest.mark.asyncio
    async def test_async_post_call_streaming_iterator_hook_re_emits_each_choice_of_an_n2_tool_call_stream(self):
        def chunk(choice_index: int, tool_call: ChatCompletionDeltaToolCall) -> ModelResponseStream:
            return ModelResponseStream(
                id="chatcmpl-n2-tools",
                created=1700000000,
                model="gpt-4.1-mini",
                object="chat.completion.chunk",
                choices=[StreamingChoices(index=choice_index, delta=Delta(tool_calls=[tool_call]))],
            )

        def fragment(arguments: str, name: str | None = None) -> ChatCompletionDeltaToolCall:
            return ChatCompletionDeltaToolCall(
                id="call_shared" if name else None,
                index=0,
                type="function",
                function=Function(name=name, arguments=arguments),
            )

        async def _stream():
            for streamed in (
                chunk(0, fragment("", name="Bash")),
                chunk(1, fragment("", name="Bash")),
                chunk(0, fragment('{"command": "ls"}')),
                chunk(1, fragment('{"command": "pwd"}')),
            ):
                yield streamed

        re_emitted: Final = [
            chunk
            async for chunk in self.guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(), response=_stream(), request_data={}
            )
        ]

        assert [
            (choice.index, [(call.function.name, call.function.arguments) for call in choice.delta.tool_calls])
            for choice in chain.from_iterable(streamed.choices for streamed in re_emitted)
        ] == [(0, [("Bash", '{"command": "ls"}')]), (1, [("Bash", '{"command": "pwd"}')])]

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

    def test_modify_response_resets_finish_reason_when_every_tool_call_is_denied(self):
        tool_call = ChatCompletionMessageToolCall(function={"name": "Read", "arguments": "{}"}, id="call_123")
        response = ModelResponse(
            choices=[Choices(finish_reason="tool_calls", message={"tool_calls": [tool_call], "content": ""})]
        )
        denied_tools = [
            (
                tool_call,
                PermissionError(tool_name="Read", rule_id="deny_read", message="Tool 'Read' denied by rule 'deny_read'"),
            )
        ]

        self.guardrail._modify_response_with_permission_errors(response, denied_tools)

        choice = response.choices[0]
        assert isinstance(choice, Choices)
        assert choice.finish_reason == "stop", (
            "keeping finish_reason tool_calls with no surviving tool calls leaves the client waiting on a tool"
        )

    def test_modify_response_with_permission_errors_filters_legacy_function_call(self):
        response = ModelResponse(
            choices=[
                Choices(
                    message={
                        "function_call": {
                            "name": "Read",
                            "arguments": "{}",
                        },
                        "content": "",
                    }
                )
            ]
        )
        tool_call = self.guardrail._extract_tool_calls_from_response(response)[0]
        denied_tools = [
            (
                tool_call,
                PermissionError(
                    tool_name="Read",
                    rule_id="deny_read",
                    message="Tool 'Read' denied by rule 'deny_read'",
                ),
            )
        ]

        self.guardrail._modify_response_with_permission_errors(response, denied_tools)

        choice = response.choices[0]
        assert isinstance(choice, Choices)
        assert choice.message.function_call is None
        assert isinstance(choice.message.content, str)
        assert "Permission denied" in choice.message.content


    def test_modify_response_with_permission_errors_keeps_custom_tool_call_in_another_choice(self):
        denied_call = ChatCompletionMessageToolCall(
            id="call_bash", type="function", function=Function(name="Bash", arguments="{}")
        )
        custom_call = ChatCompletionMessageCustomToolCall(
            id="call_patch", custom=ChatCompletionCustomToolCallPayload(name="apply_patch", input="*** Begin Patch")
        )
        response = ModelResponse(
            choices=[
                Choices(index=0, message={"role": "assistant", "content": None, "tool_calls": [denied_call]}),
                Choices(index=1, message={"role": "assistant", "content": None, "tool_calls": [custom_call]}),
            ]
        )
        denied_tools = [
            (
                denied_call,
                PermissionError(tool_name="Bash", rule_id="deny_bash", message="Tool 'Bash' denied by rule 'deny_bash'"),
            )
        ]

        self.guardrail._modify_response_with_permission_errors(response, denied_tools)

        kept = [
            [call.id for call in choice.message.tool_calls or []]
            for choice in response.choices
            if isinstance(choice, Choices)
        ]
        assert kept == [[], ["call_patch"]]

class TestToolPermissionGuardrailIntegration:
    """Integration tests for Tool Permission Guardrail"""

    def test_default_action_allow(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="test-allow-default",
            rules=[{"id": "deny_read", "tool_name": r"^Read$", "decision": "deny"}],
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

    def test_case_insensitive_default_action(self):
        """Test that default_action accepts capitalized values and normalizes them"""
        # Test capitalized 'Deny'
        guardrail = ToolPermissionGuardrail(
            guardrail_name="test-case-insensitive",
            rules=[],
            default_action="Deny",  # Should be normalized to 'deny'
        )
        assert guardrail.default_action == "deny"

        # Test capitalized 'Allow'
        guardrail2 = ToolPermissionGuardrail(
            guardrail_name="test-case-insensitive2",
            rules=[],
            default_action="Allow",  # Should be normalized to 'allow'
        )
        assert guardrail2.default_action == "allow"

        # Test uppercase 'DENY'
        guardrail3 = ToolPermissionGuardrail(
            guardrail_name="test-case-insensitive3",
            rules=[],
            default_action="DENY",  # Should be normalized to 'deny'
        )
        assert guardrail3.default_action == "deny"

    def test_case_insensitive_on_disallowed_action(self):
        """Test that on_disallowed_action accepts capitalized values and normalizes them"""
        # Test capitalized 'Block'
        guardrail = ToolPermissionGuardrail(
            guardrail_name="test-on-disallowed",
            rules=[],
            default_action="deny",
            on_disallowed_action="Block",  # Should be normalized to 'block'
        )
        assert guardrail.on_disallowed_action == "block"

        # Test capitalized 'Rewrite'
        guardrail2 = ToolPermissionGuardrail(
            guardrail_name="test-on-disallowed2",
            rules=[],
            default_action="deny",
            on_disallowed_action="Rewrite",  # Should be normalized to 'rewrite'
        )
        assert guardrail2.on_disallowed_action == "rewrite"

    def test_case_insensitive_decision_in_rules(self):
        """Test that decision field in rules accepts capitalized values and normalizes them"""
        guardrail = ToolPermissionGuardrail(
            guardrail_name="test-decision-case",
            rules=[
                {
                    "id": "allow_bash",
                    "tool_name": r"^Bash$",
                    "decision": "Allow",
                },  # Capitalized
                {
                    "id": "deny_read",
                    "tool_name": r"^Read$",
                    "decision": "DENY",
                },  # Uppercase
            ],
            default_action="deny",
        )

        # Verify rules are normalized
        assert guardrail.rules[0].decision == "allow"
        assert guardrail.rules[1].decision == "deny"

        # Verify functionality still works
        is_allowed, rule_id, _ = guardrail._check_tool_permission("Bash")
        assert is_allowed is True
        assert rule_id == "allow_bash"

        is_allowed, rule_id, _ = guardrail._check_tool_permission("Read")
        assert is_allowed is False
        assert rule_id == "deny_read"


class TestToolPermissionGuardrailInMemoryUpdate:
    """Regression: an in-memory params update (PUT /guardrails path) must rebuild
    the compiled rule maps, not just self.rules, so the new rules are enforced
    without reinitializing the guardrail."""

    def _bash(self, command):
        return ChatCompletionMessageToolCall(
            function={"name": "Bash", "arguments": json.dumps({"command": command})},
            type="function",
        )

    def test_update_in_memory_recompiles_added_param_pattern(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="tp",
            rules=[{"id": "native-bash", "tool_name": r"^Bash$", "decision": "allow"}],
            default_action="deny",
            on_disallowed_action="block",
        )
        # No pattern yet: any Bash command is allowed.
        assert (
            guardrail._get_permission_for_tool_call(self._bash("echo blockme"))[0]
            is True
        )

        guardrail.update_in_memory_litellm_params(
            LitellmParams(
                guardrail="tool_permission",
                mode=["pre_call", "post_call"],
                default_action="deny",
                on_disallowed_action="block",
                rules=[
                    {
                        "id": "native-bash",
                        "tool_name": r"^Bash$",
                        "decision": "allow",
                        "allowed_param_patterns": {
                            "command": r"^(?!(echo blockme)$).*$"
                        },
                    }
                ],
            )
        )

        # The compiled map must be rebuilt, and enforcement must reflect it.
        assert "command" in guardrail._compiled_rule_patterns.get("native-bash", {})
        assert (
            guardrail._get_permission_for_tool_call(self._bash("echo blockme"))[0]
            is False
        )
        assert (
            guardrail._get_permission_for_tool_call(self._bash("echo hello"))[0] is True
        )

    def test_update_in_memory_recompiles_tool_name_target(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="tp",
            rules=[],
            default_action="allow",
            on_disallowed_action="block",
        )
        # No rules: default_action allow lets Bash through.
        assert guardrail._get_permission_for_tool_call(self._bash("echo x"))[0] is True

        guardrail.update_in_memory_litellm_params(
            LitellmParams(
                guardrail="tool_permission",
                mode=["pre_call", "post_call"],
                default_action="allow",
                on_disallowed_action="block",
                rules=[{"id": "deny-bash", "tool_name": r"^Bash$", "decision": "deny"}],
            )
        )

        # A newly added deny rule (new id) must match -> its compiled target was rebuilt.
        assert "deny-bash" in guardrail._compiled_rule_targets
        assert guardrail._get_permission_for_tool_call(self._bash("echo x"))[0] is False

    def test_update_in_memory_preserves_rules_when_rules_absent(self):
        guardrail = ToolPermissionGuardrail(
            guardrail_name="tp",
            rules=[
                {
                    "id": "native-bash",
                    "tool_name": r"^Bash$",
                    "decision": "allow",
                    "allowed_param_patterns": {"command": r"^(?!(echo blockme)$).*$"},
                }
            ],
            default_action="deny",
            on_disallowed_action="block",
        )
        assert "command" in guardrail._compiled_rule_patterns.get("native-bash", {})

        # A partial update that does not carry `rules` must NOT wipe the existing
        # ruleset / compiled maps.
        guardrail.update_in_memory_litellm_params(
            LitellmParams(
                guardrail="tool_permission",
                mode=["pre_call", "post_call"],
                default_action="deny",
                on_disallowed_action="block",
            )
        )

        assert len(guardrail.rules) == 1
        assert "command" in guardrail._compiled_rule_patterns.get("native-bash", {})
        assert (
            guardrail._get_permission_for_tool_call(self._bash("echo blockme"))[0]
            is False
        )

    def test_update_in_memory_rejects_invalid_regex_and_keeps_previous_rules(self):
        """Regression: a live update whose rules contain an invalid regex must be
        rejected atomically. The bad rule must not leak in as a compiled-target
        wildcard (match-all), and the previously enforced ruleset must survive."""
        guardrail = ToolPermissionGuardrail(
            guardrail_name="tp",
            rules=[{"id": "deny-secret", "tool_name": r"^Secret$", "decision": "deny"}],
            default_action="allow",
            on_disallowed_action="block",
        )
        # Baseline: only "Secret" is denied; any other tool is allowed.
        assert guardrail._check_tool_permission("Secret")[0] is False
        assert guardrail._check_tool_permission("Other")[0] is True

        with pytest.raises(ValueError, match="Invalid regex for tool_name in rule 'bad': unterminated"):
            guardrail.update_in_memory_litellm_params(
                LitellmParams(
                    guardrail="tool_permission",
                    mode=["pre_call", "post_call"],
                    default_action="allow",
                    on_disallowed_action="block",
                    rules=[
                        {
                            "id": "deny-secret",
                            "tool_name": r"^Secret$",
                            "decision": "deny",
                        },
                        {"id": "bad", "tool_name": "[unclosed", "decision": "deny"},
                    ],
                )
            )

        # The bad rule must not have leaked in, and the prior ruleset must hold.
        assert "bad" not in guardrail._compiled_rule_targets
        assert all(rule.id != "bad" for rule in guardrail.rules)
        assert guardrail._check_tool_permission("Other")[0] is True
        assert guardrail._check_tool_permission("Secret")[0] is False


class TestToolPermissionGuardrailAnthropicMessages:
    """LIT-5250: /v1/messages responses arrive as Anthropic content blocks, not a
    ModelResponse. Before the fix the hooks early-returned on that shape, so every
    tool call an Anthropic-native client made bypassed the rules entirely.
    """

    def setup_method(self):
        self.rules = [
            {"id": "allow_bash", "tool_name": r"^Bash$", "decision": "allow"},
            {"id": "deny_read", "tool_name": r"^Read$", "decision": "deny"},
        ]
        self.blocking = ToolPermissionGuardrail(
            guardrail_name="anthropic-block",
            rules=self.rules,
            default_action="deny",
            on_disallowed_action="block",
        )
        self.rewriting = ToolPermissionGuardrail(
            guardrail_name="anthropic-rewrite",
            rules=self.rules,
            default_action="deny",
            on_disallowed_action="rewrite",
        )

    def _response(self, *blocks):
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": list(blocks),
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    def _tool_use(self, name, tool_id="tu_1"):
        return {"type": "tool_use", "id": tool_id, "name": name, "input": {"command": "ls"}}

    def _always_on_pre_call(self, on_disallowed_action: Literal["block", "rewrite"]) -> ToolPermissionGuardrail:
        return ToolPermissionGuardrail(
            guardrail_name=f"anthropic-pre-call-{on_disallowed_action}",
            rules=self.rules,
            default_action="deny",
            on_disallowed_action=on_disallowed_action,
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "denied_tool",
        [
            {"name": "Read", "input_schema": {"type": "object", "properties": {}}},
            {"type": "custom", "name": "Read", "input_schema": {"type": "object", "properties": {}}},
            {"type": "function", "name": "Read", "parameters": {"type": "object", "properties": {}}},
        ],
        ids=["anthropic", "anthropic_custom_type", "responses_api_flat_function"],
    )
    async def test_pre_call_blocks_denied_request_tool_in_flat_format(self, denied_tool: dict[str, object]) -> None:
        data = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}], "tools": [denied_tool]}

        with pytest.raises(HTTPException) as excinfo:
            await self._always_on_pre_call("block").async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(default_in_memory_ttl=1),
                data=data,
                call_type="anthropic_messages",
            )

        assert excinfo.value.status_code == 400
        assert excinfo.value.detail["detection_message"] == "Tool 'Read' denied by rule 'deny_read'"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_fields", "rule_tool_type", "call_type"),
        [
            ({"format": {"type": "text"}}, r"^custom$", "responses"),
            ({"input_schema": {"type": "object", "properties": {}}}, r"^function$", "anthropic_messages"),
            ({"input_schema": {"type": "object", "properties": {}}}, r"^function$", "pass_through_endpoint"),
            ({"custom": {"name": "apply_patch"}, "input_schema": {"type": "object"}}, r"^custom$", "acompletion"),
        ],
        ids=[
            "responses_custom_stays_custom",
            "anthropic_custom_reads_as_function",
            "anthropic_custom_reads_as_function_on_a_pass_through_route",
            "chat_custom_with_a_stray_input_schema_stays_custom",
        ],
    )
    async def test_pre_call_tool_type_rules_follow_the_request_format(
        self, tool_fields: dict[str, object], rule_tool_type: str, call_type: CallTypesLiteral
    ) -> None:
        guardrail = ToolPermissionGuardrail(
            guardrail_name=f"tool-type-{call_type}",
            rules=[{"id": "deny_type", "tool_type": rule_tool_type, "decision": "deny"}],
            default_action="allow",
            on_disallowed_action="block",
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
        )
        data = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "custom", "name": "apply_patch", **tool_fields}],
        }

        with pytest.raises(HTTPException) as excinfo:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(default_in_memory_ttl=1),
                data=data,
                call_type=call_type,
            )

        assert excinfo.value.status_code == 400
        assert excinfo.value.detail["detection_message"] == "Tool 'apply_patch' denied by rule 'deny_type'"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_shape", "tool_choice", "call_type", "expected_tool_choice"),
        [
            (
                {"input_schema": {"type": "object", "properties": {}}},
                {"type": "tool", "name": "Read"},
                "anthropic_messages",
                {"type": "none"},
            ),
            (
                {"type": "function", "parameters": {"type": "object", "properties": {}}},
                {"type": "function", "name": "Read"},
                "responses",
                "none",
            ),
        ],
        ids=["anthropic", "responses_api"],
    )
    async def test_pre_call_rewrite_strips_denied_flat_tool_and_forced_choice(
        self,
        tool_shape: dict[str, object],
        tool_choice: dict[str, str],
        call_type: CallTypesLiteral,
        expected_tool_choice: dict[str, str] | str,
    ) -> None:
        data = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "Bash", **tool_shape}, {"name": "Read", **tool_shape}],
            "tool_choice": tool_choice,
        }

        result = await self._always_on_pre_call("rewrite").async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(default_in_memory_ttl=1),
            data=data,
            call_type=call_type,
        )

        assert [tool["name"] for tool in result["tools"]] == ["Bash"]
        assert result["tool_choice"] == expected_tool_choice

    def _pre_call(
        self,
        rules: list[dict[str, str]],
        on_disallowed_action: Literal["block", "rewrite"],
        default_action: Literal["allow", "deny"] = "allow",
    ) -> ToolPermissionGuardrail:
        return ToolPermissionGuardrail(
            guardrail_name=f"request-shapes-{on_disallowed_action}",
            rules=rules,
            default_action=default_action,
            on_disallowed_action=on_disallowed_action,
            event_hook=GuardrailEventHooks.pre_call,
            default_on=True,
        )

    async def _run_pre_call(
        self, guardrail: ToolPermissionGuardrail, data: dict[str, object], call_type: CallTypesLiteral
    ) -> dict[str, object]:
        return await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(default_in_memory_ttl=1),
            data=data,
            call_type=call_type,
        )

    def _namespace(self, *member_names: str) -> dict[str, object]:
        return {
            "type": "namespace",
            "name": "mcp__everything",
            "tools": [{"type": "function", "name": name, "parameters": {"type": "object"}} for name in member_names],
        }

    @pytest.mark.asyncio
    async def test_pre_call_blocks_a_denied_tool_inside_a_responses_namespace(self) -> None:
        guardrail = self._pre_call(
            [{"id": "deny_sum", "tool_name": r"^mcp__everything__get_sum$", "decision": "deny"}], "block"
        )
        data = {"model": "gpt-5", "input": "add 2 and 3", "tools": [self._namespace("echo", "get_sum")]}

        with pytest.raises(HTTPException) as excinfo:
            await self._run_pre_call(guardrail, data, "responses")

        assert excinfo.value.detail["detection_message"] == "Tool 'mcp__everything__get_sum' denied by rule 'deny_sum'"

    @pytest.mark.asyncio
    async def test_pre_call_rewrite_drops_denied_namespace_members_and_empty_namespaces(self) -> None:
        guardrail = self._pre_call([{"id": "deny_sum", "tool_name": r"^.*get_sum$", "decision": "deny"}], "rewrite")
        partly_denied = {**self._namespace("echo", "get_sum"), "name": "mcp__kept"}
        data = {"model": "gpt-5", "input": "add", "tools": [partly_denied, self._namespace("get_sum")]}

        result = await self._run_pre_call(guardrail, data, "responses")

        assert result["tools"] == [{**partly_denied, "tools": partly_denied["tools"][:1]}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rule", "tool", "call_type", "detection_message"),
        [
            (
                {"id": "deny_mcp", "tool_type": r"^mcp$", "decision": "deny"},
                {"type": "mcp", "server_label": "weather", "server_url": "https://mcp.example.com"},
                "responses",
                "Tool 'weather' denied by rule 'deny_mcp'",
            ),
            (
                {"id": "allow_bash", "tool_name": r"^Bash$", "decision": "allow"},
                {"type": "mcp_toolset", "mcp_server_name": "weather"},
                "anthropic_messages",
                "Tool 'weather' denied by default action",
            ),
            (
                {"id": "deny_weather", "tool_name": r"^weather$", "decision": "deny"},
                {"type": "mcp", "server_label": "weather", "server_url": "https://mcp.example.com"},
                "responses",
                "Tool 'weather' denied by rule 'deny_weather'",
            ),
            (
                {"id": "deny_lookup", "tool_name": r"^mcp__weather__lookup$", "decision": "deny"},
                {"type": "namespace", "name": "mcp__weather", "tools": [{"type": "custom", "name": "lookup"}]},
                "responses",
                "Tool 'mcp__weather__lookup' denied by rule 'deny_lookup'",
            ),
            (
                {"id": "deny_lookup", "tool_name": r"^mcp__weather__lookup$", "decision": "deny"},
                {"functionDeclarations": [{"name": "mcp__weather__lookup"}, {"name": "get_weather"}]},
                "agenerate_content",
                "Tool 'mcp__weather__lookup' denied by rule 'deny_lookup'",
            ),
            (
                {"id": "allow_bash", "tool_name": r"^Bash$", "decision": "allow"},
                {"googleSearch": {}},
                "acompletion",
                "Tool 'googleSearch' denied by default action",
            ),
            (
                {"id": "deny_custom", "tool_type": r"^custom$", "decision": "deny"},
                {"type": "custom", "custom": {"name": "grammar_tool"}},
                "acompletion",
                "Tool 'grammar_tool' denied by rule 'deny_custom'",
            ),
            (
                {"id": "deny_read", "tool_name": r"^Read$", "decision": "deny"},
                {"type": "custom", "custom": {"name": "Read"}},
                "acompletion",
                "Tool 'Read' denied by rule 'deny_read'",
            ),
        ],
        ids=[
            "responses_mcp_by_type",
            "anthropic_mcp_toolset_by_server_name",
            "responses_mcp_by_server_label",
            "responses_namespace_custom_member",
            "gemini_function_declarations",
            "gemini_type_only_tool_by_its_key",
            "chat_custom_keeps_its_type",
            "chat_custom_nested_name",
        ],
    )
    async def test_pre_call_checks_tools_that_carry_no_top_level_name(
        self, rule: dict[str, str], tool: dict[str, object], call_type: CallTypesLiteral, detection_message: str
    ) -> None:
        guardrail = self._pre_call([rule], "block", default_action="deny")
        data = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "input": "hi", "tools": [tool]}

        with pytest.raises(HTTPException) as excinfo:
            await self._run_pre_call(guardrail, data, call_type)

        assert excinfo.value.detail["detection_message"] == detection_message

    @pytest.mark.asyncio
    async def test_pre_call_lets_a_type_rule_allow_a_gemini_tool_by_its_key(self) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_search", "tool_type": r"^googleSearch$", "decision": "allow"}], "block", default_action="deny"
        )
        data = {"model": "gemini-3", "messages": [{"role": "user", "content": "hi"}], "tools": [{"googleSearch": {}}]}

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [{"googleSearch": {}}]

    @pytest.mark.asyncio
    async def test_pre_call_blocks_a_denied_tool_loaded_by_responses_tool_search(self) -> None:
        guardrail = self._pre_call([{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}], "block")
        loaded = {"type": "tool_search_output", "tools": [{"type": "function", "name": "mcp__weather__lookup"}]}
        data = {"model": "gpt-5", "input": [loaded], "tools": []}

        with pytest.raises(HTTPException) as excinfo:
            await self._run_pre_call(guardrail, data, "responses")

        assert excinfo.value.detail["detection_message"] == "Tool 'mcp__weather__lookup' denied by rule 'deny_mcp'"

    @pytest.mark.asyncio
    async def test_pre_call_rewrite_keeps_the_allowed_members_of_grouped_and_loaded_tools(self) -> None:
        guardrail = self._pre_call([{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}], "rewrite")
        get_weather = {"type": "function", "name": "get_weather"}
        lookup = {"type": "function", "name": "mcp__weather__lookup"}
        data = {
            "model": "m",
            "input": [{"role": "user", "content": "hi"}, {"type": "tool_search_output", "tools": [get_weather, lookup]}],
            "tools": [{"functionDeclarations": [{"name": "get_weather"}, {"name": "mcp__weather__lookup"}]}],
        }

        result = await self._run_pre_call(guardrail, data, "responses")

        assert result["tools"] == [{"functionDeclarations": [{"name": "get_weather"}]}]
        assert result["input"] == [
            {"role": "user", "content": "hi"},
            {"type": "tool_search_output", "tools": [get_weather]},
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("call_type", "message", "rewritten"),
        [
            (
                "anthropic_messages",
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [{"type": "tool_reference", "tool_name": "mcp__weather__lookup"}],
                        },
                        {"type": "text", "text": "Tool loaded."},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Permission denied: tool mcp__weather__lookup is not available",
                                }
                            ],
                        },
                        {"type": "text", "text": "Tool loaded."},
                    ],
                },
            ),
            (
                "anthropic_messages",
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": {
                                "type": "tool_search_tool_search_result",
                                "tool_references": [
                                    {"type": "tool_reference", "tool_name": "mcp__weather__lookup"},
                                    {"type": "tool_reference", "tool_name": "get_weather"},
                                ],
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_search_tool_result",
                            "tool_use_id": "srvtoolu_1",
                            "content": {
                                "type": "tool_search_tool_search_result",
                                "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
                            },
                        }
                    ],
                },
            ),
            (
                "acompletion",
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "tool_reference", "tool_name": "mcp__weather__lookup"},
                        {"type": "tool_reference", "tool_name": "get_weather"},
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_reference", "tool_name": "get_weather"}],
                },
            ),
        ],
    )
    async def test_pre_call_rewrite_drops_references_to_a_removed_tool(self, call_type, message, rewritten) -> None:
        guardrail = self._pre_call([{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}], "rewrite")
        data = {
            "model": "m",
            "messages": [{"role": "user", "content": "weather in Istanbul"}, message],
            "tools": [
                {"name": "get_weather", "input_schema": {"type": "object"}, "defer_loading": True},
                {"name": "mcp__weather__lookup", "input_schema": {"type": "object"}, "defer_loading": True},
            ],
        }

        result = await self._run_pre_call(guardrail, data, call_type)

        assert [tool["name"] for tool in result["tools"]] == ["get_weather"]
        assert result["messages"] == [{"role": "user", "content": "weather in Istanbul"}, rewritten]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_choice", ["required", {"type": "function", "name": "get_weather"}])
    async def test_pre_call_rewrite_keeps_tool_choice_when_only_loaded_tools_remain(self, tool_choice) -> None:
        guardrail = self._pre_call([{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}], "rewrite")
        get_weather = {"type": "function", "name": "get_weather"}
        lookup = {"type": "function", "name": "mcp__weather__lookup"}
        data = {
            "model": "m",
            "input": [{"role": "user", "content": "hi"}, {"type": "tool_search_output", "tools": [get_weather, lookup]}],
            "tools": [lookup],
            "tool_choice": tool_choice,
        }

        result = await self._run_pre_call(guardrail, data, "responses")

        assert result["tools"] == []
        assert result["input"][1] == {"type": "tool_search_output", "tools": [get_weather]}
        assert result["tool_choice"] == tool_choice

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rule", "tools", "tool_choice"),
        [
            (
                {"id": "deny_search", "tool_type": r"^web_search$", "decision": "deny"},
                [{"type": "web_search"}, {"type": "function", "name": "get_weather"}],
                {"type": "web_search"},
            ),
            (
                {"id": "deny_weather", "tool_name": r"^weather$", "decision": "deny"},
                [
                    {"type": "mcp", "server_label": "weather", "server_url": "https://a.example.com"},
                    {"type": "mcp", "server_label": "files", "server_url": "https://b.example.com"},
                ],
                {"type": "mcp", "server_label": "weather"},
            ),
        ],
        ids=["hosted_tool_type", "hosted_mcp_server"],
    )
    async def test_pre_call_rewrite_relaxes_a_hosted_tool_choice_it_removed(
        self, rule: dict[str, str], tools: list[dict[str, str]], tool_choice: dict[str, str]
    ) -> None:
        guardrail = self._pre_call([rule], "rewrite")
        data = {"model": "gpt-5", "input": "hi", "tools": tools, "tool_choice": tool_choice}

        result = await self._run_pre_call(guardrail, data, "responses")

        assert result["tools"] == tools[1:]
        assert result["tool_choice"] == "none"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tools", "tool_choice", "call_type", "expected_tool_choice"),
        [
            (
                [{"type": "custom", "name": "Bash"}, {"type": "custom", "name": "Read"}],
                {"type": "custom", "name": "Read"},
                "responses",
                "none",
            ),
            (
                [{"type": "function", "name": "Bash"}, {"type": "function", "name": "Read"}],
                {
                    "type": "allowed_tools",
                    "mode": "required",
                    "tools": [{"type": "function", "name": "Read"}, {"type": "function", "name": "Bash"}],
                },
                "responses",
                {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "Bash"}]},
            ),
            (
                [{"type": "function", "name": "Bash"}, {"type": "function", "name": "Read"}],
                {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": "Read"}]},
                "responses",
                "none",
            ),
            (
                [
                    {"type": "function", "function": {"name": "Bash"}},
                    {"type": "function", "function": {"name": "Read"}},
                ],
                {
                    "type": "allowed_tools",
                    "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "Read"}}]},
                },
                "acompletion",
                "none",
            ),
            (
                [{"name": "Read", "input_schema": {"type": "object"}}],
                {"type": "any"},
                "anthropic_messages",
                {"type": "none"},
            ),
            (
                [
                    {"name": "Bash", "input_schema": {"type": "object"}},
                    {"name": "Read", "input_schema": {"type": "object"}},
                ],
                {"type": "any"},
                "anthropic_messages",
                {"type": "any"},
            ),
            (
                [{"type": "custom", "custom": {"name": "Bash"}}, {"type": "custom", "custom": {"name": "Read"}}],
                {"type": "custom", "custom": {"name": "Read"}},
                "acompletion",
                "none",
            ),
        ],
        ids=[
            "responses_custom_choice",
            "responses_allowed_tools_keeps_the_rest",
            "responses_allowed_tools_all_denied",
            "chat_allowed_tools_all_denied",
            "anthropic_any_with_no_tools_left",
            "anthropic_any_with_a_tool_left",
            "chat_custom_choice",
        ],
    )
    async def test_pre_call_rewrite_leaves_no_tool_choice_that_needs_a_removed_tool(
        self,
        tools: list[dict[str, object]],
        tool_choice: dict[str, object],
        call_type: CallTypesLiteral,
        expected_tool_choice: dict[str, object] | str,
    ) -> None:
        guardrail = self._pre_call(self.rules, "rewrite")
        data = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
            "tool_choice": tool_choice,
        }

        result = await self._run_pre_call(guardrail, data, call_type)

        assert result["tool_choice"] == expected_tool_choice

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool", "tool_choice", "call_type", "allowed_tool_type"),
        [
            ({"type": "custom", "name": "apply_patch"}, {"type": "custom", "name": "apply_patch"}, "responses", r"^custom$"),
            (
                {"type": "web_search_20250305", "name": "web_search"},
                {"type": "tool", "name": "web_search"},
                "anthropic_messages",
                r"^web_search_\d+$",
            ),
        ],
        ids=["responses_custom_tool", "anthropic_server_tool"],
    )
    async def test_pre_call_checks_a_forced_choice_with_its_tool_type(
        self,
        tool: dict[str, str],
        tool_choice: dict[str, str],
        call_type: CallTypesLiteral,
        allowed_tool_type: str,
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_type", "tool_type": allowed_tool_type, "decision": "allow"}], "block", default_action="deny"
        )
        data = {"model": "m", "input": "hi", "tools": [tool], "tool_choice": tool_choice}

        result = await self._run_pre_call(guardrail, data, call_type)

        assert result["tools"] == [tool]
        assert result["tool_choice"] == tool_choice

    @pytest.mark.asyncio
    async def test_pre_call_rewrite_relaxes_a_choice_for_a_tool_a_type_rule_removed(self) -> None:
        guardrail = self._pre_call([{"id": "deny_custom", "tool_type": r"^custom$", "decision": "deny"}], "rewrite")
        data = {
            "model": "gpt-5",
            "input": "patch it",
            "tools": [{"type": "function", "name": "Bash"}, {"type": "custom", "name": "apply_patch"}],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
        }

        result = await self._run_pre_call(guardrail, data, "responses")

        assert result["tools"] == [{"type": "function", "name": "Bash"}]
        assert result["tool_choice"] == "none"

    async def _blocked_and_rewritten(
        self, rules: list[dict[str, str]], request: dict[str, object], call_type: CallTypesLiteral
    ) -> tuple[str, dict[str, object]]:
        with pytest.raises(HTTPException) as excinfo:
            await self._run_pre_call(self._pre_call(rules, "block"), copy.deepcopy(request), call_type)
        rewritten = await self._run_pre_call(self._pre_call(rules, "rewrite"), request, call_type)
        return excinfo.value.detail["detection_message"], rewritten

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("denied_tool", "detection_message"),
        [
            (
                {"type": "function", "name": "mcp__weather__lookup"},
                "Tool 'mcp__weather__lookup' denied by rule 'deny_mcp'",
            ),
            (
                {"type": "namespace", "name": "mcp__weather", "tools": [{"type": "function", "name": "lookup"}]},
                "Tool 'mcp__weather__lookup' denied by rule 'deny_mcp'",
            ),
            ({"type": "web_search"}, "Tool 'web_search' denied by rule 'deny_search'"),
        ],
        ids=["flat", "namespace_member", "hosted_type"],
    )
    async def test_pre_call_checks_and_strips_tools_in_responses_additional_tools_items(
        self, denied_tool: dict[str, object], detection_message: str
    ) -> None:
        rules = [
            {"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"},
            {"id": "deny_search", "tool_type": r"^web_search$", "decision": "deny"},
        ]
        get_weather = {"type": "function", "name": "get_weather"}
        request = {
            "model": "gpt-5",
            "input": [
                {"role": "user", "content": "hi"},
                {"type": "additional_tools", "tools": [get_weather, denied_tool]},
                {"type": "additional_tools", "tools": [denied_tool]},
            ],
            "tools": [],
            "tool_choice": "required",
        }

        message, rewritten = await self._blocked_and_rewritten(rules, request, "responses")

        assert message == detection_message
        assert rewritten["input"] == [
            {"role": "user", "content": "hi"},
            {"type": "additional_tools", "tools": [get_weather]},
        ]
        assert rewritten["tool_choice"] == "required"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("call_type", "bash", "read"),
        [
            (
                "acompletion",
                {"type": "function", "function": {"name": "Bash"}},
                {"type": "function", "function": {"name": "Read"}},
            ),
            ("responses", {"type": "function", "name": "Bash"}, {"type": "function", "name": "Read"}),
            (
                "anthropic_messages",
                {"name": "Bash", "input_schema": {"type": "object"}},
                {"name": "Read", "input_schema": {"type": "object"}},
            ),
        ],
        ids=["chat", "responses", "anthropic"],
    )
    async def test_pre_call_checks_and_strips_tools_in_extra_body(
        self, call_type: CallTypesLiteral, bash: dict[str, object], read: dict[str, object]
    ) -> None:
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "input": "hi",
            "extra_body": {"tools": [bash, read], "seed": 7},
        }

        message, rewritten = await self._blocked_and_rewritten(self.rules, request, call_type)

        assert message == "Tool 'Read' denied by rule 'deny_read'"
        assert rewritten["extra_body"] == {"tools": [bash], "seed": 7}

    @pytest.mark.asyncio
    async def test_rewrite_relaxes_an_extra_body_tool_choice_that_names_a_dropped_tool(self) -> None:
        guardrail = self._pre_call([{"id": "deny_delete", "tool_name": r"^delete_db$", "decision": "deny"}], "rewrite")
        weather = {"type": "function", "function": {"name": "get_weather"}}
        delete_db = {"type": "function", "function": {"name": "delete_db"}}
        data = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {
                "tools": [weather, delete_db],
                "tool_choice": {"type": "function", "function": {"name": "delete_db"}},
                "seed": 7,
            },
        }

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["extra_body"] == {"tools": [weather], "tool_choice": "none", "seed": 7}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rule",
        [
            {"id": "deny_deepwiki", "tool_name": r"^deepwiki$", "decision": "deny"},
            {"id": "deny_deepwiki", "tool_name": r"^deepwiki$", "tool_type": r"^mcp_toolset$", "decision": "deny"},
        ],
        ids=["by_server_name", "by_server_name_and_mcp_toolset_type"],
    )
    async def test_pre_call_checks_and_strips_anthropic_mcp_servers_like_an_mcp_toolset(
        self, rule: dict[str, str]
    ) -> None:
        deepwiki = {"type": "url", "url": "https://mcp.deepwiki.com/mcp", "name": "deepwiki"}
        files = {"type": "url", "url": "https://files.example.com/mcp", "name": "files"}
        request = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "mcp_servers": [deepwiki, files],
            "tool_choice": {"type": "any"},
        }

        message, rewritten = await self._blocked_and_rewritten([rule], request, "anthropic_messages")

        assert message == "Tool 'deepwiki' denied by rule 'deny_deepwiki'"
        assert rewritten["mcp_servers"] == [files]
        assert rewritten["tool_choice"] == {"type": "any"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rule", "tool", "detection_message", "kept_tools"),
        [
            (
                {"id": "deny_code", "tool_type": r"^codeExecution$", "decision": "deny"},
                {"googleSearch": {}, "codeExecution": {}},
                "Tool 'codeExecution' denied by rule 'deny_code'",
                [{"googleSearch": {}}],
            ),
            (
                {"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"},
                {"functionDeclarations": [{"name": "mcp__weather__lookup"}], "googleSearch": {}},
                "Tool 'mcp__weather__lookup' denied by rule 'deny_mcp'",
                [{"googleSearch": {}}],
            ),
            (
                {"id": "deny_search", "tool_type": r"^googleSearch$", "decision": "deny"},
                {"functionDeclarations": [{"name": "get_weather"}], "googleSearch": {}},
                "Tool 'googleSearch' denied by rule 'deny_search'",
                [{"functionDeclarations": [{"name": "get_weather"}]}],
            ),
            (
                {"id": "deny_builtin", "tool_type": r"^(googleSearch|codeExecution)$", "decision": "deny"},
                {"googleSearch": {}, "codeExecution": {}},
                "Tool 'googleSearch' denied by rule 'deny_builtin'",
                [],
            ),
        ],
        ids=["denied_key_beside_a_key", "every_declaration_denied", "denied_key_beside_declarations", "all_denied"],
    )
    async def test_pre_call_checks_and_strips_each_part_of_a_multi_key_gemini_tool(
        self,
        rule: dict[str, str],
        tool: dict[str, object],
        detection_message: str,
        kept_tools: list[dict[str, object]],
    ) -> None:
        request = {"model": "gemini-3", "contents": [{"role": "user", "parts": [{"text": "hi"}]}], "tools": [tool]}

        message, rewritten = await self._blocked_and_rewritten([rule], request, "agenerate_content")

        assert message == detection_message
        assert rewritten["tools"] == kept_tools

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    @pytest.mark.parametrize(
        ("tool", "call_type"),
        [({"type": "web_search"}, "responses"), ({"googleSearch": {}}, "acompletion")],
        ids=["responses_hosted_type", "gemini_tool_key"],
    )
    async def test_pre_call_matches_a_name_rule_against_a_nameless_tool_by_its_type(
        self,
        tool: dict[str, object],
        call_type: CallTypesLiteral,
        on_disallowed_action: Literal["block", "rewrite"],
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_all", "tool_name": r".*", "decision": "allow"}], on_disallowed_action, default_action="deny"
        )
        data = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "input": "hi", "tools": [tool]}

        result = await self._run_pre_call(guardrail, data, call_type)

        assert result["tools"] == [tool]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rules", "default_action", "detection_message"),
        [
            (
                [{"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"}],
                "allow",
                "Tool 'exec' denied by rule 'deny_exec'",
            ),
            ([{"id": "allow_exec", "tool_name": r"^exec$", "decision": "allow"}], "deny", None),
            (
                [
                    {"id": "allow_exec", "tool_name": r"^exec$", "decision": "allow"},
                    {"id": "deny_functions", "tool_name": r"^functions__.*$", "decision": "deny"},
                ],
                "allow",
                "Tool 'functions__exec' denied by rule 'deny_functions'",
            ),
            (
                [
                    {"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"},
                    {"id": "allow_functions", "tool_type": r"^function$", "decision": "allow"},
                ],
                "allow",
                "Tool 'exec' denied by rule 'deny_exec'",
            ),
            (
                [
                    {"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"},
                    {"id": "allow_all", "tool_name": r"^.*$", "decision": "allow"},
                ],
                "deny",
                "Tool 'exec' denied by rule 'deny_exec'",
            ),
            (
                [
                    {"id": "allow_functions", "tool_name": r"^functions__.*$", "decision": "allow"},
                    {"id": "deny_all", "tool_name": r"^.*$", "decision": "deny"},
                ],
                "allow",
                None,
            ),
        ],
        ids=[
            "bare_name_deny_rule",
            "bare_name_allow_rule",
            "qualified_rule_wins_over_an_earlier_bare_rule",
            "type_rule_keeps_an_earlier_bare_name_rule_in_charge",
            "bare_name_deny_beats_a_later_catch_all_allow",
            "qualified_allow_beats_a_later_catch_all_deny",
        ],
    )
    async def test_pre_call_judges_a_namespace_member_by_its_qualified_and_bare_names(
        self, rules: list[dict[str, str]], default_action: Literal["allow", "deny"], detection_message: str | None
    ) -> None:
        guardrail = self._pre_call(rules, "block", default_action=default_action)
        namespace = {"type": "namespace", "name": "functions", "tools": [{"type": "function", "name": "exec"}]}
        data = {"model": "gpt-5", "input": "run date", "tools": [namespace]}

        if detection_message is None:
            assert (await self._run_pre_call(guardrail, data, "responses"))["tools"] == [namespace]
            return
        with pytest.raises(HTTPException) as excinfo:
            await self._run_pre_call(guardrail, data, "responses")
        assert excinfo.value.detail["detection_message"] == detection_message

    @pytest.mark.asyncio
    async def test_pre_call_rewrite_drops_a_namespace_member_a_bare_name_rule_denies(self) -> None:
        guardrail = self._pre_call([{"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"}], "rewrite")
        members = [{"type": "function", "name": "exec"}, {"type": "function", "name": "read_file"}]
        namespace = {"type": "namespace", "name": "functions", "tools": members}
        data = {"model": "gpt-5", "input": "run date", "tools": [namespace]}

        result = await self._run_pre_call(guardrail, data, "responses")

        assert result["tools"] == [{"type": "namespace", "name": "functions", "tools": members[1:]}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("placement", ["tools", "functions"])
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    @pytest.mark.parametrize(
        ("member", "detection_message"),
        [
            ({"type": "function", "function": {"name": "delete_db"}}, "Tool 'delete_db' denied by rule 'deny_delete'"),
            (
                {"type": "function", "name": "lookup", "function": {"name": "delete_db"}},
                "Tool 'delete_db' denied by rule 'deny_delete'",
            ),
            ({"googleSearch": {}}, "Tool 'googleSearch' denied by rule 'deny_search'"),
            (
                {
                    "type": "namespace",
                    "name": "inner",
                    "tools": [{"type": "function", "name": "noop"}],
                    "function": {"name": "delete_db"},
                },
                "Tool 'delete_db' denied by rule 'deny_delete'",
            ),
        ],
        ids=["chat_shaped_member", "function_object_behind_another_name", "hosted_key_member", "nested_namespace"],
    )
    async def test_pre_call_judges_a_namespace_member_by_what_a_provider_reads_from_it(
        self,
        member: dict[str, object],
        detection_message: str,
        on_disallowed_action: Literal["block", "rewrite"],
        placement: Literal["tools", "functions"],
    ) -> None:
        guardrail = self._pre_call(
            [
                {"id": "deny_delete", "tool_name": r"^delete_db$", "decision": "deny"},
                {"id": "deny_search", "tool_type": r"^googleSearch$", "decision": "deny"},
            ],
            on_disallowed_action,
        )
        weather = {"type": "function", "name": "get_weather"}
        data = {
            "model": "gemini-3",
            "messages": [{"role": "user", "content": "hi"}],
            placement: [{"type": "namespace", "name": "ops", "tools": [weather, member]}],
        }

        if on_disallowed_action == "block":
            with pytest.raises(HTTPException) as excinfo:
                await self._run_pre_call(guardrail, data, "acompletion")
            assert excinfo.value.detail["detection_message"] == detection_message
            return

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result[placement] == [{"type": "namespace", "name": "ops", "tools": [weather]}]

    @pytest.mark.asyncio
    async def test_pre_call_allows_a_chat_shaped_namespace_member_the_allowlist_names(self) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_create", "tool_name": r"^github__create_issue$", "decision": "allow"}],
            "block",
            default_action="deny",
        )
        namespace = {
            "type": "namespace",
            "name": "github",
            "tools": [{"type": "function", "function": {"name": "create_issue"}}],
        }
        data = {"model": "gemini-3", "messages": [{"role": "user", "content": "hi"}], "tools": [namespace]}

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [namespace]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_config", "rewritten_tool_config"),
        [
            (
                {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": ["mcp__weather__lookup", "get_weather"],
                    }
                },
                {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}},
            ),
            (
                {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["mcp__weather__lookup"]}},
                {"functionCallingConfig": {"mode": "NONE"}},
            ),
            (
                {"function_calling_config": {"mode": "any", "allowed_function_names": ["mcp__weather__lookup"]}},
                {"function_calling_config": {"mode": "NONE"}},
            ),
            ({"functionCallingConfig": {"mode": "ANY"}}, {"functionCallingConfig": {"mode": "ANY"}}),
            (
                {"functionCallingConfig": {"mode": "AUTO", "allowedFunctionNames": []}},
                {"functionCallingConfig": {"mode": "AUTO", "allowedFunctionNames": []}},
            ),
        ],
        ids=[
            "some_allowed_names_left",
            "every_allowed_name_removed",
            "snake_case_fields",
            "no_allowed_names",
            "empty_allowed_names_list",
        ],
    )
    async def test_pre_call_rewrite_drops_removed_functions_from_a_gemini_tool_config(
        self, tool_config: dict[str, object], rewritten_tool_config: dict[str, object]
    ) -> None:
        guardrail = self._pre_call([{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}], "rewrite")
        config_key = "tool_config" if "function_calling_config" in tool_config else "toolConfig"
        data = {
            "model": "gemini-3",
            "contents": [{"role": "user", "parts": [{"text": "weather in Paris?"}]}],
            "tools": [{"functionDeclarations": [{"name": "mcp__weather__lookup"}, {"name": "get_weather"}]}],
            config_key: tool_config,
        }

        result = await self._run_pre_call(guardrail, data, "agenerate_content")

        assert result["tools"] == [{"functionDeclarations": [{"name": "get_weather"}]}]
        assert result[config_key] == rewritten_tool_config

    @pytest.mark.asyncio
    async def test_pre_call_rewrite_turns_off_forced_gemini_calling_when_no_function_is_left(self) -> None:
        guardrail = self._pre_call([{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}], "rewrite")
        data = {
            "model": "gemini-3",
            "contents": [{"role": "user", "parts": [{"text": "weather in Paris?"}]}],
            "tools": [{"functionDeclarations": [{"name": "mcp__weather__lookup"}]}, {"googleSearch": {}}],
            "toolConfig": {"functionCallingConfig": {"mode": "ANY"}},
        }

        result = await self._run_pre_call(guardrail, data, "agenerate_content")

        assert result["tools"] == [{"googleSearch": {}}]
        assert result["toolConfig"] == {"functionCallingConfig": {"mode": "NONE"}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    async def test_pre_call_judges_a_named_hosted_tool_by_its_type_as_well_as_its_name(
        self, on_disallowed_action: Literal["block", "rewrite"]
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"}],
            on_disallowed_action,
            default_action="deny",
        )
        weather = {"type": "function", "function": {"name": "get_weather"}}
        data = {
            "model": "gemini-3",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [weather, {"type": "web_search", "name": "get_weather"}],
        }

        if on_disallowed_action == "block":
            with pytest.raises(HTTPException) as excinfo:
                await self._run_pre_call(guardrail, data, "acompletion")
            assert excinfo.value.detail["detection_message"] == "Tool 'web_search' denied by default action"
            return

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [weather]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "rules, tool",
        [
            (
                [{"id": "allow_bash", "tool_name": r"^bash$", "decision": "allow"}],
                {"type": "bash_20250124", "name": "bash"},
            ),
            (
                [{"id": "allow_edit", "tool_name": r"^str_replace_based_edit_tool$", "decision": "allow"}],
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
            ),
            (
                [{"id": "allow_search", "tool_name": r"^web_search$", "decision": "allow"}],
                {"type": "web_search_20250305", "name": "web_search"},
            ),
            (
                [
                    {
                        "id": "allow_search",
                        "tool_name": r"^web_search$",
                        "tool_type": r"^web_search_20250305$",
                        "decision": "allow",
                    }
                ],
                {"type": "web_search_20250305", "name": "web_search"},
            ),
        ],
        ids=["client_tool_by_name", "client_tool_with_its_own_name", "server_tool_by_name", "name_and_type_rule"],
    )
    async def test_pre_call_allows_an_anthropic_defined_tool_that_an_allowlist_names(
        self, rules: list[dict[str, object]], tool: dict[str, object]
    ) -> None:
        guardrail = self._pre_call(rules, "block", default_action="deny")
        data = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}], "tools": [tool]}

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [tool]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool, call_type",
        [
            ({"type": "web_search_20250305", "name": "get_weather"}, "anthropic_messages"),
            (
                {
                    "type": "computer_20251124",
                    "name": "get_weather",
                    "function": {"parameters": {"display_width_px": 1024, "display_height_px": 768}},
                },
                "acompletion",
            ),
            ({"type": "computer_use", "name": "get_weather", "environment": "browser"}, "acompletion"),
            ({"type": "text_editor_20250728", "name": "get_weather"}, "anthropic_messages"),
        ],
        ids=["server_tool", "anthropic_computer_tool", "gemini_computer_use", "text_editor"],
    )
    async def test_pre_call_blocks_a_provider_defined_tool_whose_name_is_not_its_own(
        self, tool: dict[str, object], call_type: CallTypesLiteral
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"}],
            "block",
            default_action="deny",
        )
        data = {"model": "gemini-3", "messages": [{"role": "user", "content": "hi"}], "tools": [tool]}

        with pytest.raises(HTTPException) as excinfo:
            await self._run_pre_call(guardrail, data, call_type)

        assert excinfo.value.detail["detection_message"] == f"Tool '{tool['type']}' denied by default action"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    @pytest.mark.parametrize(
        "tool",
        [
            {"type": "function", "function": {"name": "delete_db"}, "functionDeclarations": [{"name": "get_weather"}]},
            {"type": "function", "function": {"name": "delete_db"}, "function_declarations": [{"name": "get_weather"}]},
            {"type": "custom", "name": "get_weather", "function": {"name": "delete_db"}},
            {"type": "mcp", "server_label": "get_weather", "function": {"name": "delete_db"}},
            {
                "type": "namespace",
                "name": "github",
                "tools": [{"type": "function", "name": "create_issue"}],
                "function": {"name": "delete_db"},
            },
            {
                "type": "computer_20251124",
                "name": "computer",
                "function": {"name": "delete_db", "parameters": {"display_width_px": 1024, "display_height_px": 768}},
            },
        ],
        ids=[
            "declarations_list",
            "snake_case_declarations_list",
            "custom_top_level_name",
            "mcp_server_label",
            "namespace_with_members",
            "computer_top_level_name",
        ],
    )
    async def test_pre_call_judges_the_function_a_tool_runs_behind_an_allowed_name(
        self, tool: dict[str, object], on_disallowed_action: Literal["block", "rewrite"]
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_known", "tool_name": r"^(get_weather|computer|github__create_issue)$", "decision": "allow"}],
            on_disallowed_action,
            default_action="deny",
        )
        weather = {"type": "function", "function": {"name": "get_weather"}}
        data = {"model": "gemini-3", "messages": [{"role": "user", "content": "hi"}], "tools": [weather, tool]}

        if on_disallowed_action == "block":
            with pytest.raises(HTTPException) as excinfo:
                await self._run_pre_call(guardrail, data, "acompletion")
            assert excinfo.value.detail["detection_message"] == "Tool 'delete_db' denied by default action"
            return

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [weather]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    @pytest.mark.parametrize(
        ("tool", "hosted_key"),
        [
            ({"type": "get_weather", "googleSearch": {}}, "googleSearch"),
            ({"type": "function", "codeExecution": {}}, "codeExecution"),
            ({"type": "get_weather", "url_context": {}}, "url_context"),
        ],
        ids=["named_type_google_search", "function_type_code_execution", "named_type_url_context"],
    )
    async def test_pre_call_judges_a_gemini_hosted_tool_key_whatever_the_tool_type(
        self, tool: dict[str, object], hosted_key: str, on_disallowed_action: Literal["block", "rewrite"]
    ) -> None:
        guardrail = self._pre_call(
            [
                {"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"},
                {"id": "allow_functions", "tool_type": r"^function$", "decision": "allow"},
            ],
            on_disallowed_action,
            default_action="deny",
        )
        weather = {"type": "function", "function": {"name": "get_weather"}}
        data = {"model": "gemini-3", "messages": [{"role": "user", "content": "hi"}], "tools": [weather, tool]}

        if on_disallowed_action == "block":
            with pytest.raises(HTTPException) as excinfo:
                await self._run_pre_call(guardrail, data, "acompletion")
            assert excinfo.value.detail["detection_message"] == f"Tool '{hosted_key}' denied by default action"
            return

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [weather]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    async def test_pre_call_leaves_a_gemini_hosted_key_that_sits_beside_a_function(
        self, on_disallowed_action: Literal["block", "rewrite"]
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"}],
            on_disallowed_action,
            default_action="deny",
        )
        tool = {"type": "function", "function": {"name": "get_weather"}, "googleSearch": {}}
        data = {"model": "gemini-3", "messages": [{"role": "user", "content": "hi"}], "tools": [tool]}

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [tool]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    @pytest.mark.parametrize(
        ("function", "denied"),
        [
            ({"type": "function", "function": {"name": "delete_db"}}, "delete_db"),
            ({"name": "get_weather", "function": {"name": "delete_db"}}, "delete_db"),
            ({"googleSearch": {}}, "googleSearch"),
        ],
        ids=["function_object", "function_object_behind_an_allowed_name", "gemini_hosted_key"],
    )
    async def test_pre_call_judges_a_legacy_functions_entry_like_a_tool(
        self, function: dict[str, object], denied: str, on_disallowed_action: Literal["block", "rewrite"]
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"}],
            on_disallowed_action,
            default_action="deny",
        )
        weather = {"name": "get_weather"}
        data = {
            "model": "gemini-3",
            "messages": [{"role": "user", "content": "hi"}],
            "functions": [weather, function],
        }

        if on_disallowed_action == "block":
            with pytest.raises(HTTPException) as excinfo:
                await self._run_pre_call(guardrail, data, "acompletion")
            assert excinfo.value.detail["detection_message"] == f"Tool '{denied}' denied by default action"
            return

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["functions"] == [weather]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    async def test_pre_call_judges_a_legacy_functions_entry_as_a_function_whatever_its_type(
        self, on_disallowed_action: Literal["block", "rewrite"]
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "deny_delete", "tool_name": r"^delete_db$", "tool_type": r"^function$", "decision": "deny"}],
            on_disallowed_action,
        )
        weather = {"name": "get_weather"}
        data = {
            "model": "gemini-3",
            "messages": [{"role": "user", "content": "hi"}],
            "functions": [weather, {"type": "custom", "name": "delete_db"}],
        }

        if on_disallowed_action == "block":
            with pytest.raises(HTTPException) as excinfo:
                await self._run_pre_call(guardrail, data, "acompletion")
            assert excinfo.value.detail["detection_message"] == "Tool 'delete_db' denied by rule 'deny_delete'"
            return

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["functions"] == [weather]

    @pytest.mark.asyncio
    async def test_rewrite_keeps_an_allowed_function_whose_declaration_list_is_denied(self) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"}],
            "rewrite",
            default_action="deny",
        )
        tool = {
            "type": "function",
            "function": {"name": "get_weather"},
            "functionDeclarations": [{"name": "delete_db"}],
        }
        data = {"model": "gemini-3", "messages": [{"role": "user", "content": "hi"}], "tools": [tool]}

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [{"type": "function", "function": {"name": "get_weather"}}]

    @pytest.mark.asyncio
    async def test_pre_call_allows_a_custom_tool_by_its_function_name(self) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"}],
            "block",
            default_action="deny",
        )
        tool = {"type": "custom", "function": {"name": "get_weather"}}
        data = {"model": "claude-sonnet-4-5", "messages": [{"role": "user", "content": "hi"}], "tools": [tool]}

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [tool]

    @pytest.mark.asyncio
    async def test_pre_call_blocks_a_tool_whose_type_is_not_a_string_by_its_name(self) -> None:
        guardrail = self._pre_call(
            [
                {"id": "allow_search", "tool_type": r"^web_search$", "decision": "allow"},
                {"id": "deny_read", "tool_name": r"^Read$", "decision": "deny"},
            ],
            "block",
        )
        data = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": ["custom"], "name": "Read", "input_schema": {"type": "object"}}],
        }

        with pytest.raises(HTTPException) as excinfo:
            await self._run_pre_call(guardrail, data, "anthropic_messages")

        assert excinfo.value.status_code == 400
        assert excinfo.value.detail["detection_message"] == "Tool 'Read' denied by rule 'deny_read'"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("malformed_tool", "tool_choice", "call_type"),
        [
            ({"type": ["custom"], "name": "Read", "input_schema": {"type": "object"}}, None, "anthropic_messages"),
            ({"type": "function", "name": "Read"}, {"type": {"a": 1}}, "responses"),
        ],
        ids=["tool_type_is_a_list", "tool_choice_type_is_an_object"],
    )
    async def test_pre_call_rewrite_strips_a_denied_tool_and_leaves_malformed_fields_for_the_provider(
        self, malformed_tool: dict[str, object], tool_choice: dict[str, object] | None, call_type: CallTypesLiteral
    ) -> None:
        guardrail = self._pre_call([{"id": "deny_read", "tool_name": r"^Read$", "decision": "deny"}], "rewrite")
        bash = {"type": "function", "name": "Bash"}
        data = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [bash, malformed_tool],
            **({"tool_choice": tool_choice} if tool_choice is not None else {}),
        }

        result = await self._run_pre_call(guardrail, data, call_type)

        assert result["tools"] == [bash]
        assert result.get("tool_choice") == tool_choice

    @pytest.mark.asyncio
    async def test_pre_call_judges_a_tool_with_a_null_type_by_its_nested_function_name(self) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_read", "tool_name": r"^Read$", "decision": "allow"}], "block", default_action="deny"
        )
        tool = {"type": None, "function": {"name": "Read", "parameters": {"type": "object"}}}
        data = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}], "tools": [tool]}

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [tool]

    @pytest.mark.asyncio
    async def test_pre_call_rewrite_drops_a_removed_hosted_tool_from_allowed_tools(self) -> None:
        guardrail = self._pre_call([{"id": "deny_search", "tool_type": r"^web_search$", "decision": "deny"}], "rewrite")
        weather = {"type": "function", "name": "get_weather"}
        tool_choice = {"type": "allowed_tools", "mode": "required", "tools": [weather, {"type": "web_search"}]}
        data = {"model": "gpt-5", "input": "hi", "tools": [weather, {"type": "web_search"}], "tool_choice": tool_choice}

        result = await self._run_pre_call(guardrail, data, "responses")

        assert result["tools"] == [weather]
        assert result["tool_choice"] == {"type": "allowed_tools", "mode": "required", "tools": [weather]}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tools", [{"not": "a list"}, '[{"type": "function"}]'], ids=["object", "json_string"])
    async def test_pre_call_rewrite_leaves_a_tools_field_that_is_not_a_list_for_the_provider_to_reject(
        self, tools: object
    ) -> None:
        guardrail = self._pre_call([{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}], "rewrite")
        data = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "mcp__weather__lookup"}},
        }

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == tools
        assert result["tool_choice"] == "none"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    async def test_pre_call_reads_bedrock_tool_blocks_by_name_and_skips_cache_points(
        self, on_disallowed_action: Literal["block", "rewrite"]
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"}],
            on_disallowed_action,
            default_action="deny",
        )
        weather = {"toolSpec": {"name": "get_weather", "inputSchema": {"json": {}}}}
        tools = [weather, {"cachePoint": {"type": "default"}}]
        data = {"model": "bedrock/claude", "messages": [{"role": "user", "content": "hi"}], "tools": tools}

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == tools

    @pytest.mark.asyncio
    async def test_pre_call_rewrite_drops_a_denied_bedrock_tool_block_by_its_name(self) -> None:
        guardrail = self._pre_call([{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}], "rewrite")
        kept = [{"toolSpec": {"name": "get_weather"}}, {"cachePoint": {"type": "default"}}]
        data = {
            "model": "bedrock/claude",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"toolSpec": {"name": "mcp__weather__lookup"}}, *kept],
        }

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == kept

    @pytest.mark.asyncio
    async def test_denied_anthropic_tool_use_is_blocked(self):
        response = self._response({"type": "text", "text": "reading"}, self._tool_use("Read"))

        with patch.object(self.blocking, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await self.blocking.async_post_call_success_hook(
                    data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
                )

    @pytest.mark.asyncio
    async def test_allowed_anthropic_tool_use_passes_through_untouched(self):
        response = self._response({"type": "text", "text": "listing"}, self._tool_use("Bash"))

        with patch.object(self.blocking, "should_run_guardrail", return_value=True):
            result = await self.blocking.async_post_call_success_hook(
                data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
            )

        assert [b["type"] for b in result["content"]] == ["text", "tool_use"]
        assert result["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_rewrite_mode_strips_the_denied_anthropic_tool_use(self):
        response = self._response({"type": "text", "text": "reading"}, self._tool_use("Read"))

        with patch.object(self.rewriting, "should_run_guardrail", return_value=True):
            result = await self.rewriting.async_post_call_success_hook(
                data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
            )

        assert all(b["type"] != "tool_use" for b in result["content"]), (
            "denied tool_use must not reach the client in rewrite mode"
        )
        assert any("Permission denied" in b.get("text", "") for b in result["content"])
        assert result["stop_reason"] == "end_turn", (
            "leaving stop_reason as tool_use makes the client wait for a tool result that will never come"
        )

    @pytest.mark.asyncio
    async def test_rewrite_mode_keeps_allowed_tool_use_when_only_one_is_denied(self):
        response = self._response(self._tool_use("Bash", "tu_ok"), self._tool_use("Read", "tu_bad"))

        with patch.object(self.rewriting, "should_run_guardrail", return_value=True):
            result = await self.rewriting.async_post_call_success_hook(
                data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
            )

        tool_ids = [b["id"] for b in result["content"] if b["type"] == "tool_use"]
        assert tool_ids == ["tu_ok"]
        assert result["stop_reason"] == "tool_use"

    def _sse_chunks(self, tool_name, tool_id="tu_1"):
        events = [
            {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                                  "model": "claude-sonnet-4-5", "content": [], "stop_reason": None,
                                                  "usage": {"input_tokens": 10, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "working"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}}},
            {"type": "content_block_delta", "index": 1,
             "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events]

    async def _drain(self, guardrail, chunks):
        async def _stream():
            for chunk in chunks:
                yield chunk

        return [
            c
            async for c in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(), response=_stream(), request_data={}
            )
        ]

    @pytest.mark.asyncio
    async def test_denied_tool_use_in_anthropic_sse_stream_is_blocked(self):
        with patch.object(self.blocking, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await self._drain(self.blocking, self._sse_chunks("Read"))

    @pytest.mark.asyncio
    async def test_allowed_tool_use_in_anthropic_sse_stream_is_passed_through_verbatim(self):
        chunks = self._sse_chunks("Bash")

        with patch.object(self.blocking, "should_run_guardrail", return_value=True):
            out = await self._drain(self.blocking, chunks)

        assert out == chunks, "an allowed stream must not be re-serialized"

    @pytest.mark.asyncio
    async def test_rewrite_mode_removes_denied_tool_use_from_anthropic_sse_stream(self):
        with patch.object(self.rewriting, "should_run_guardrail", return_value=True):
            out = await self._drain(self.rewriting, self._sse_chunks("Read"))

        body = b"".join(c if isinstance(c, bytes) else str(c).encode() for c in out).decode()
        assert '"type": "tool_use"' not in body, "denied tool_use must not survive into the rewritten stream"
        assert "Permission denied" in body
        assert '"stop_reason": "end_turn"' in body, (
            "dropping every tool_use must end the turn, or the client waits for a tool result that never comes"
        )
        assert '"stop_reason": "tool_use"' not in body

    @pytest.mark.asyncio
    async def test_rewrite_mode_keeps_the_stream_identity_it_had_before_the_shared_helper(self):
        """Well-formed SSE must round-trip exactly as it did before the helpers were shared.

        The shared module can stamp the upstream message id onto the assembled response for
        callers that ask for it; this path never did, and a client reads those bytes. The model,
        though, is now the upstream's, matching what the untouched passthrough shows clients.
        """
        with patch.object(self.rewriting, "should_run_guardrail", return_value=True):
            out = await self._drain(self.rewriting, self._sse_chunks("Read"))

        body = b"".join(c if isinstance(c, bytes) else str(c).encode() for c in out).decode()
        message_start = next(
            json.loads(line[6:])
            for line in body.splitlines()
            if line.startswith("data: ") and json.loads(line[6:]).get("type") == "message_start"
        )["message"]
        assert message_start["id"].startswith("chatcmpl-"), "the rewritten stream must not adopt the upstream message id"
        assert message_start["model"] == "claude-sonnet-4-5", "the rewritten stream reports the model the upstream served"

    @pytest.mark.asyncio
    async def test_rewrite_mode_keeps_the_allowed_tool_use_verdict_with_its_explanation(self):
        """Claude Code auto mode reads the classifier verdict from message_delta. The rewrite changes
        no text the explanation could repeat, so the allowed call keeps its verdict whole."""
        default_on_rewriting: Final = ToolPermissionGuardrail(
            guardrail_name="anthropic-rewrite-default-on",
            rules=self.rules,
            default_action="deny",
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
            default_on=True,
        )
        bash_verdict: Final = {"type": "evaluated", "outcome": "flagged", "explanation": "lists the home directory"}
        safeguard_results: Final = [
            {
                "type": "dangerous_tool_use",
                "status": {
                    "type": "available",
                    "tool_uses": {"tu_ok": bash_verdict, "tu_bad": {"type": "evaluated", "outcome": "not_flagged"}},
                },
            }
        ]
        bash_chunks: Final = self._sse_chunks("Bash", "tu_ok")
        read_chunks: Final = self._sse_chunks("Read", "tu_bad")
        read_tool_use: Final = [chunk.replace(b'"index": 1', b'"index": 2') for chunk in read_chunks[4:7]]
        message_delta: Final = {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "safeguard_results": safeguard_results},
            "usage": {"output_tokens": 5},
        }
        chunks: Final = [
            *bash_chunks[:7],
            *read_tool_use,
            f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode(),
            bash_chunks[-1],
        ]

        body: Final = b"".join(await self._drain(default_on_rewriting, chunks)).decode()
        delta: Final = next(
            json.loads(line[6:])["delta"]
            for line in body.splitlines()
            if line.startswith("data: ") and json.loads(line[6:]).get("type") == "message_delta"
        )
        assert "Permission denied" in body
        assert delta["safeguard_results"] == [
            {"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": {"tu_ok": bash_verdict}}}
        ]

    @pytest.mark.asyncio
    async def test_message_start_without_a_dict_message_fails_closed(self):
        """Malformed SSE must not be forwarded unscanned.

        The shared assembler requires message_start.message to be a dict; the private helper it
        replaced accepted anything, and assembled a response from it.
        """
        events = [
            {"type": "message_start", "message": "not-a-dict"},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "message_stop"},
        ]
        chunks = [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events]

        with patch.object(self.rewriting, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await self._drain(self.rewriting, chunks)

    UPSTREAM_ERROR: Final = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}

    def _ended_by_upstream_error(self, chunks: list[bytes]) -> list[bytes]:
        return [*chunks, f"event: error\ndata: {json.dumps(self.UPSTREAM_ERROR)}\n\n".encode()]

    def _cut_off_tool_use(self, index: int, name: str, block_type: str = "tool_use") -> list[bytes]:
        events: Final = (
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": block_type, "id": f"tu_{index}", "name": name, "input": {}},
            },
            {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": '{"command": "l'}},
        )
        return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events]

    def _events(self, chunks: list[bytes]) -> list[dict[str, object]]:
        lines: Final = b"".join(chunks).decode().split("\n")
        return [json.loads(line.removeprefix("data: ")) for line in lines if line.startswith("data: ")]

    def _tool_use_blocks(self, events: list[dict[str, object]]) -> list[dict[str, object]]:
        return [
            block
            for event in events
            if isinstance(block := event.get("content_block"), dict) and block.get("type") == "tool_use"
        ]

    def _client_blocks(self, chunks: list[bytes]) -> list[tuple[object, ...]]:
        events: Final = self._events(chunks)
        stopped: Final = {event["index"] for event in events if event["type"] == "content_block_stop"}

        def content_of(index: object) -> str:
            deltas: Final = (e["delta"] for e in events if e["type"] == "content_block_delta" and e["index"] == index)
            return "".join(delta.get("text") or delta.get("partial_json") or "" for delta in deltas)

        return [
            (event["content_block"]["type"], event["content_block"].get("name"), content_of(event["index"]), event["index"] in stopped)
            for event in events
            if event["type"] == "content_block_start"
        ]

    @pytest.mark.asyncio
    async def test_stream_of_only_an_upstream_error_is_passed_through_verbatim(self):
        chunks: Final = self._ended_by_upstream_error([])

        out: Final = await self._drain(self.blocking, chunks)

        assert out == chunks, "the client must see the upstream error, not a guardrail block that hides it"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("delivered_events", [1, 3, 7], ids=["before-content", "mid-text", "after-allowed-tool"])
    async def test_stream_ended_by_an_upstream_error_gives_the_client_what_the_check_read(self, delivered_events: int):
        delivered: Final = self._sse_chunks("Bash")[:delivered_events]

        out: Final = await self._drain(self.blocking, self._ended_by_upstream_error(delivered))

        events: Final = self._events(out)
        assert self._client_blocks(out) == self._client_blocks(delivered)
        assert events[0]["message"]["id"] == "msg_1", "only the frames are rebuilt, the message stays the upstream's"
        assert events[-1] == self.UPSTREAM_ERROR, "the client must see the upstream error, not a guardrail block"
        assert {"message_delta", "message_stop"}.isdisjoint(event["type"] for event in events)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_input",
        ['{"command" "ls"', "3", "[1]", "[" * 200_000 + "]" * 200_000],
        ids=["not-json", "number", "list", "nested-too-deep"],
    )
    async def test_finished_tool_call_the_rewrite_cannot_rebuild_gets_the_upstream_error_alone(self, tool_input: str):
        sse_chunks: Final = self._sse_chunks("Bash")
        input_delta: Final = {"type": "input_json_delta", "partial_json": tool_input}
        bad_input: Final = json.dumps({"type": "content_block_delta", "index": 1, "delta": input_delta})
        chunks: Final = self._ended_by_upstream_error(
            [*sse_chunks[:5], f"event: content_block_delta\ndata: {bad_input}\n\n".encode(), sse_chunks[6]]
        )

        out: Final = await self._drain(self.blocking, chunks)

        assert self._events(out) == [self.UPSTREAM_ERROR]

    @pytest.mark.asyncio
    async def test_param_rule_denies_a_tool_input_nested_too_deep_to_parse(self):
        guardrail: Final = ToolPermissionGuardrail(
            guardrail_name="anthropic-param-block",
            rules=[
                {
                    "id": "bash_ls_only",
                    "tool_name": r"^Bash$",
                    "decision": "allow",
                    "allowed_param_patterns": {"command": r"^ls$"},
                }
            ],
            default_action="deny",
            on_disallowed_action="block",
        )
        sse_chunks: Final = self._sse_chunks("Bash")
        input_delta: Final = {"type": "input_json_delta", "partial_json": "[" * 200_000 + "]" * 200_000}
        deep_input: Final = json.dumps({"type": "content_block_delta", "index": 1, "delta": input_delta})

        with pytest.raises(GuardrailRaisedException, match="could not be parsed"):
            await self._drain(
                guardrail, [*sse_chunks[:5], f"event: content_block_delta\ndata: {deep_input}\n\n".encode(), *sse_chunks[6:]]
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "frame",
        ["event: error\u2028\n\n", "event: error\x0b\n\n", "event:  error\n\n", "event: error\nevent: ping\n\n", "event: error"],
        ids=["u2028", "vertical-tab", "second-space", "later-name", "no-blank-line"],
    )
    async def test_healthy_stream_with_a_frame_no_client_reads_as_an_error_is_passed_through_verbatim(self, frame: str):
        chunks: Final = [*self._sse_chunks("Bash"), frame.encode()]

        out: Final = await self._drain(self.blocking, chunks)

        assert out == chunks

    @pytest.mark.asyncio
    async def test_non_object_data_frame_in_a_stream_ended_by_an_upstream_error_is_left_out(self):
        sse_chunks: Final = self._sse_chunks("Bash")
        chunks: Final = self._ended_by_upstream_error([sse_chunks[0], b"data: 5\n\n", *sse_chunks[1:3]])

        out: Final = await self._drain(self.blocking, chunks)

        assert self._client_blocks(out) == self._client_blocks(sse_chunks[:3])
        assert b"data: 5" not in b"".join(out)

    @pytest.mark.asyncio
    async def test_allowed_stream_led_by_a_non_object_data_line_is_passed_through_verbatim(self):
        chunks: Final = [b"data: 5\n\n", *self._sse_chunks("Bash")]

        out: Final = await self._drain(self.blocking, chunks)

        assert out == chunks

    ERROR_FRAMES_WITH_NO_ERROR_TYPE: Final = {
        "non-object-data": b"event: error\ndata: 5\n\n",
        "non-json-data": b"event: error\ndata: Overloaded\n\n",
        "object-with-no-type": b'event: error\ndata: {"message": "Overloaded"}\n\n',
    }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_frame", ERROR_FRAMES_WITH_NO_ERROR_TYPE.values(), ids=ERROR_FRAMES_WITH_NO_ERROR_TYPE)
    async def test_rewrite_ends_in_an_error_event_known_only_by_its_name(self, error_frame: bytes):
        chunks: Final = [*self._sse_chunks("Read")[:7], *self._cut_off_tool_use(2, "Bash"), error_frame]

        out: Final = await self._drain(self.rewriting, chunks)

        events: Final = self._events(out[:-1])
        assert out[-1] == error_frame
        assert {"message_delta", "message_stop"}.isdisjoint(event["type"] for event in events)
        assert self._tool_use_blocks(events) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_frame", ERROR_FRAMES_WITH_NO_ERROR_TYPE.values(), ids=ERROR_FRAMES_WITH_NO_ERROR_TYPE)
    async def test_allowed_stream_drops_a_call_cut_off_before_an_error_event_known_only_by_its_name(
        self, error_frame: bytes
    ):
        delivered: Final = self._sse_chunks("Bash")[:3]
        chunks: Final = [*delivered, *self._cut_off_tool_use(2, "Bash"), error_frame]

        out: Final = await self._drain(self.blocking, chunks)

        assert self._client_blocks(out[:-1]) == self._client_blocks(delivered)
        assert out[-1] == error_frame

    @pytest.mark.asyncio
    @pytest.mark.parametrize("block_type", ["server_tool_use", "mcp_tool_use"])
    async def test_stream_drops_a_server_side_call_cut_off_by_an_upstream_error(self, block_type: str):
        delivered: Final = self._sse_chunks("Bash")[:3]
        chunks: Final = [*delivered, *self._cut_off_tool_use(2, "web_search", block_type), *self._ended_by_upstream_error([])]

        out: Final = await self._drain(self.blocking, chunks)

        assert self._client_blocks(out) == self._client_blocks(delivered)
        assert self._events(out)[-1] == self.UPSTREAM_ERROR

    @pytest.mark.asyncio
    async def test_error_name_on_a_message_delta_frame_keeps_the_stream_as_sent(self):
        source: Final = self._sse_chunks("Bash")
        message_delta_frame: Final = source[-2]
        chunks: Final = [*source[:-2], message_delta_frame.replace(b"event: message_delta", b"event: error"), source[-1]]

        out: Final = await self._drain(self.blocking, chunks)

        assert b"".join(out) == b"".join(chunks)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_type", [[], {"a": 1}], ids=["list", "dict"])
    async def test_rewrite_keeps_an_error_frame_whose_type_is_not_a_string(self, error_type: object):
        error_frame: Final = f"event: error\ndata: {json.dumps({'type': error_type})}\n\n".encode()

        out: Final = await self._drain(self.rewriting, [*self._sse_chunks("Read")[:7], error_frame])

        assert b"".join(out).endswith(error_frame)
        assert self._tool_use_blocks(self._events(out)) == []

    @pytest.mark.asyncio
    async def test_crlf_stream_that_opens_with_an_error_is_still_blocked(self):
        frames: Final = self._ended_by_upstream_error([])
        chunks: Final = [frame.replace(b"\n", b"\r\n") for frame in (*frames, *self._sse_chunks("Read"))]

        with pytest.raises(GuardrailRaisedException, match="could not be verified"):
            await self._drain(self.blocking, chunks)

    @pytest.mark.asyncio
    async def test_denied_tool_use_split_over_data_lines_beside_an_upstream_error_is_blocked(self):
        def over_two_data_lines(frame: bytes) -> bytes:
            return frame.replace(b", ", b",\ndata: ", 1)

        chunks: Final = self._ended_by_upstream_error([over_two_data_lines(frame) for frame in self._sse_chunks("Read")])

        with pytest.raises(GuardrailRaisedException, match="could not be verified"):
            await self._drain(self.blocking, chunks)

    DENIED_CALL_EVENTS: Final = (
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "tu_1", "name": "Read", "input": {}}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'}},
        {"type": "content_block_stop", "index": 0},
    )
    DENIED_CALL: Final = tuple(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in DENIED_CALL_EVENTS)
    THINKING_AT_THE_SAME_INDEX: Final = (
        b'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, '
        b'"content_block": {"type": "thinking", "thinking": "", "signature": ""}}\n\n'
    )
    ERROR_WITH_CR_LINE_ENDS: Final = f"event: error\rdata: {json.dumps(UPSTREAM_ERROR)}\r\r".encode()
    DENIED_CALLS_ASSEMBLY_CANNOT_READ: Final = {
        "reused-block-index": (*DENIED_CALL[:2], THINKING_AT_THE_SAME_INDEX),
        "cr-line-ends": tuple(frame.replace(b"\n", b"\r") for frame in DENIED_CALL),
        "behind-a-glued-crlf-ping": (b'event: ping\r\ndata: {"type": "ping"}\r\n\r\n' + b"".join(DENIED_CALL),),
        "finished-mcp-call": tuple(frame.replace(b'"tool_use"', b'"mcp_tool_use"') for frame in DENIED_CALL),
        "behind-a-form-feed": tuple(
            b'data: {"type": "ping"}\x0c' + frame.split(b"\n", 1)[1] for frame in DENIED_CALL
        ),
        "over-two-data-lines": tuple(frame.replace(b", ", b",\ndata: ", 1) for frame in DENIED_CALL),
        "glued-to-the-error-frame": (ERROR_WITH_CR_LINE_ENDS + b"".join(DENIED_CALL).replace(b"\n", b"\r"),),
    }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("frames", DENIED_CALLS_ASSEMBLY_CANNOT_READ.values(), ids=DENIED_CALLS_ASSEMBLY_CANNOT_READ)
    async def test_denied_call_assembly_cannot_read_never_reaches_the_client_past_an_upstream_error(
        self, frames: tuple[bytes, ...]
    ):
        chunks: Final = self._ended_by_upstream_error([self._sse_chunks("Read")[0], *frames])

        out: Final = await self._drain(self.blocking, chunks)

        assert b"Read" not in b"".join(out)
        assert self._events(out)[-1] == self.UPSTREAM_ERROR

    @pytest.mark.asyncio
    async def test_denied_call_inside_message_start_never_reaches_the_client_past_an_upstream_error(self):
        message_start, *_ = self._sse_chunks("Read")
        call: Final = json.dumps(self._tool_use("Read")).encode()
        chunks: Final = self._ended_by_upstream_error([message_start.replace(b'"content": []', b'"content": [' + call + b"]")])

        out: Final = await self._drain(self.blocking, chunks)

        assert b"Read" not in b"".join(out)
        assert self._events(out)[-1] == self.UPSTREAM_ERROR

    @pytest.mark.asyncio
    async def test_client_gets_the_input_the_check_read_when_deltas_interleave_before_an_upstream_error(self):
        guardrail: Final = ToolPermissionGuardrail(
            guardrail_name="anthropic-param-block",
            rules=[
                {"id": "read_any", "tool_name": r"^Read$", "decision": "allow"},
                {
                    "id": "bash_ls_only",
                    "tool_name": r"^Bash$",
                    "decision": "allow",
                    "allowed_param_patterns": {"command": r"^ls$"},
                },
            ],
            default_action="deny",
            on_disallowed_action="block",
        )

        def tool_start(index: int, tool_id: str, name: str) -> dict[str, object]:
            block: Final = {"type": "tool_use", "id": tool_id, "name": name, "input": {}}
            return {"type": "content_block_start", "index": index, "content_block": block}

        def tool_input(index: int, command: str) -> dict[str, object]:
            partial_json: Final = json.dumps({"command": command})
            return {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": partial_json}}

        events: Final = (
            tool_start(0, "tu_1", "Read"),
            tool_start(1, "tu_2", "Bash"),
            tool_input(0, "ls"),
            {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
            tool_input(1, "rm -rf /"),
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_stop", "index": 2},
        )
        frames: Final = [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events]

        out: Final = await self._drain(guardrail, self._ended_by_upstream_error([self._sse_chunks("Bash")[0], *frames]))

        assert b"rm -rf" not in b"".join(out)
        assert self._events(out)[-1] == self.UPSTREAM_ERROR

    @pytest.mark.asyncio
    async def test_denied_tool_use_named_only_by_its_event_lines_beside_an_upstream_error_is_blocked(self):
        def without_type(frame: bytes) -> bytes:
            return re.sub(rb'"type": "(?:content_block_\w+|message_\w+)"', b'"error": {}', frame, count=1)

        chunks: Final = self._ended_by_upstream_error([without_type(frame) for frame in self._sse_chunks("Bash")])

        with pytest.raises(GuardrailRaisedException, match="could not be verified"):
            await self._drain(self.blocking, chunks)

    @pytest.mark.asyncio
    async def test_denied_tool_use_between_space_led_data_lines_is_blocked(self):
        def wrapped(frame: bytes) -> bytes:
            return b' data: {"type": "error", "error": {}, "w": [\n' + frame.rstrip(b"\n") + b"\n data: ]}\n\n"

        chunks: Final = [wrapped(frame) for frame in self._sse_chunks("Bash")]

        with pytest.raises(GuardrailRaisedException, match="could not be verified"):
            await self._drain(self.blocking, chunks)

    @pytest.mark.asyncio
    async def test_denied_tool_use_delivered_before_an_upstream_error_is_still_blocked(self):
        chunks: Final = self._ended_by_upstream_error(self._sse_chunks("Read")[:7])

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await self._drain(self.blocking, chunks)

        assert "deny_read" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("delivered_events", [5, 6], ids=["before-input", "full-input-before-stop"])
    async def test_denied_tool_use_cut_off_by_an_upstream_error_never_reaches_the_client(self, delivered_events: int):
        chunks: Final = self._ended_by_upstream_error(self._sse_chunks("Read")[:delivered_events])

        events = self._events(await self._drain(self.blocking, chunks))

        assert events[-1] == self.UPSTREAM_ERROR
        assert self._tool_use_blocks(events) == [], "a client can still read a cut-off call, so a denied one must not reach it"
        assert any("working" in json.dumps(event) for event in events)

    @pytest.mark.asyncio
    async def test_rewrite_of_a_stream_cut_off_mid_tool_input_ends_in_the_upstream_error(self):
        chunks: Final = self._ended_by_upstream_error([*self._sse_chunks("Read")[:7], *self._cut_off_tool_use(2, "Bash")])

        events = self._events(await self._drain(self.rewriting, chunks))

        assert events[-1] == self.UPSTREAM_ERROR
        assert {"message_delta", "message_stop"}.isdisjoint(event["type"] for event in events)
        assert self._tool_use_blocks(events) == [], (
            "neither the denied call nor the call the upstream cut off may reach the client as a tool call"
        )
        assert any("Permission denied" in json.dumps(event) for event in events)

    @pytest.mark.asyncio
    async def test_tool_use_cut_off_by_an_upstream_error_is_not_judged_on_its_partial_input(self):
        guardrail: Final = ToolPermissionGuardrail(
            guardrail_name="anthropic-param-block",
            rules=[
                {
                    "id": "bash_ls_only",
                    "tool_name": r"^Bash$",
                    "decision": "allow",
                    "allowed_param_patterns": {"command": r"^ls$"},
                }
            ],
            default_action="deny",
            on_disallowed_action="block",
        )
        chunks: Final = self._ended_by_upstream_error([*self._sse_chunks("Bash")[:4], *self._cut_off_tool_use(1, "Bash")])

        events: Final = self._events(await self._drain(guardrail, chunks))

        assert events[-1] == self.UPSTREAM_ERROR, (
            "the client must see the upstream error, not a block over input the upstream never finished"
        )
        assert self._tool_use_blocks(events) == []
        assert any("working" in json.dumps(event) for event in events)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("sent_events", "cut_off_index"), [(1, 0), (4, 1)], ids=["tool-call-first", "after-finished-text"]
    )
    async def test_stream_cut_off_mid_tool_call_with_nothing_denied_shows_the_client_no_block_the_upstream_did_not_finish(
        self, sent_events: int, cut_off_index: int
    ):
        """Claude Code retries a stream that failed before any block finished, but not one that shows a
        finished block, so the rebuilt stream must not finish a block for the upstream."""
        sent_before_the_cut: Final = self._sse_chunks("Bash")[:sent_events]
        chunks: Final = self._ended_by_upstream_error([*sent_before_the_cut, *self._cut_off_tool_use(cut_off_index, "Bash")])

        out: Final = await self._drain(self.blocking, chunks)

        assert self._client_blocks(out) == self._client_blocks(sent_before_the_cut)
        assert self._events(out)[-1] == self.UPSTREAM_ERROR

    @pytest.mark.asyncio
    async def test_unfinished_allowed_tool_call_with_no_upstream_error_reaches_the_client_as_sent(self):
        unfinished_tool_call: Final = self._sse_chunks("Bash")[:6]

        out: Final = await self._drain(self.blocking, unfinished_tool_call)

        assert out == unfinished_tool_call

    def _resplit(self, chunks, size=7):
        joined = b"".join(chunks)
        return [joined[i : i + size] for i in range(0, len(joined), size)]

    @pytest.mark.asyncio
    async def test_denied_tool_use_is_caught_when_sse_events_are_split_across_chunk_boundaries(self):
        with patch.object(self.blocking, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException) as exc_info:
                await self._drain(self.blocking, self._resplit(self._sse_chunks("Read")))

        assert "deny_read" in str(exc_info.value), (
            "a stream split mid-event must still assemble and hit the rule, not fail as unparseable"
        )

    @pytest.mark.asyncio
    async def test_allowed_stream_split_across_chunk_boundaries_is_passed_through_verbatim(self):
        chunks = self._resplit(self._sse_chunks("Bash"))

        with patch.object(self.blocking, "should_run_guardrail", return_value=True):
            out = await self._drain(self.blocking, chunks)

        assert out == chunks

    @pytest.mark.asyncio
    async def test_non_anthropic_sse_stream_fails_closed(self):
        gemini_chunks = [
            b'data: {"candidates": [{"content": {"parts": [{"functionCall": '
            b'{"name": "run_shell", "args": {"command": "ls"}}}], "role": "model"}}]}\n\n',
            b'data: {"candidates": [{"content": {"parts": [{"text": "done"}]}, "finishReason": "STOP"}]}\n\n',
        ]

        with patch.object(self.blocking, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await self._drain(self.blocking, gemini_chunks)

    @pytest.mark.asyncio
    async def test_unparseable_sse_stream_fails_closed(self):
        with patch.object(self.blocking, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await self._drain(self.blocking, [b"data: not-json\n\n", b"event: weird\n\n"])
