"""
Unit tests for Tool Permission Guardrail (OpenAI tool_calls semantics)
"""

import json
import os
import re
import sys
from unittest.mock import patch

import pytest

from litellm.caching.dual_cache import DualCache

sys.path.insert(0, os.path.abspath("../../../../../.."))

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
    ChatCompletionMessageToolCall,
    Choices,
    ModelResponse,
    ModelResponseStream,
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
        with pytest.raises(ValueError):
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

        with pytest.raises(ValueError):
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
