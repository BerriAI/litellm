"""
Unit tests for Tool Permission Guardrail (OpenAI tool_calls semantics)
"""

import copy
import json
import logging
import re
from collections.abc import AsyncIterator, Sequence
from itertools import chain
from typing import Final, Literal
from unittest.mock import patch

import anthropic
import httpx
import pytest
from anthropic.types import Message, ToolUseBlock

from litellm.caching.dual_cache import DualCache


from fastapi import HTTPException

from litellm.exceptions import GuardrailRaisedException
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.tool_permission import (
    ToolPermissionGuardrail,
)
from litellm.types.guardrails import GuardrailEventHooks, LitellmParams
from litellm.types.responses.main import build_web_search_call
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


_SERVER_WEB_SEARCH_CALL: Final = {
    "id": "srvtoolu_01Search",
    "type": "function",
    "function": {"name": "web_search", "arguments": '{"query": "weather in Paris"}'},
}
_ALLOWED_BASH_CALL: Final = {"id": "toolu_01Bash", "type": "function", "function": {"name": "Bash", "arguments": "{}"}}


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
    @pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
    async def test_post_call_judges_a_returned_namespace_member_by_its_qualified_name(self, stream: bool) -> None:
        guardrail = ToolPermissionGuardrail(
            guardrail_name="post-mcp",
            rules=[{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}],
            default_action="allow",
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
        )
        namespace = {"type": "namespace", "name": "mcp__weather", "tools": [{"type": "function", "name": "lookup"}]}
        data = {"guardrails": ["post-mcp"], "tools": [namespace]}
        function = Function(name="lookup", arguments='{"city": "Paris"}')

        if stream:

            async def _stream():
                yield ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            index=0,
                            delta=Delta(
                                tool_calls=[
                                    ChatCompletionDeltaToolCall(
                                        id="call_lookup", index=0, type="function", function=function
                                    )
                                ]
                            ),
                        )
                    ]
                )

            chunks = [
                chunk
                async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(), response=_stream(), request_data=data
                )
            ]
            deltas = [choice.delta for choice in chain.from_iterable(chunk.choices for chunk in chunks)]
            called = [call.function.name for call in chain.from_iterable(delta.tool_calls or () for delta in deltas)]
            content = "".join(delta.content or "" for delta in deltas)
        else:
            response = ModelResponse(
                choices=[
                    Choices(
                        finish_reason="tool_calls",
                        message={"tool_calls": [{"id": "call_lookup", "type": "function", "function": function}]},
                    )
                ]
            )
            await guardrail.async_post_call_success_hook(
                data=data, user_api_key_dict=UserAPIKeyAuth(), response=response
            )
            message = response.choices[0].message
            called = [call.function.name for call in message.tool_calls or ()]
            content = message.content or ""

        assert called == []
        assert "Tool 'mcp__weather__lookup' denied by rule 'deny_mcp'" in content

    @pytest.mark.asyncio
    async def test_post_call_judges_a_returned_namespace_member_by_the_rule_its_arguments_satisfy(self) -> None:
        guardrail = ToolPermissionGuardrail(
            guardrail_name="post-exec",
            rules=[
                {
                    "id": "allow_ls",
                    "tool_name": r"^exec$",
                    "decision": "allow",
                    "allowed_param_patterns": {"command": r"^ls$"},
                },
                {"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"},
                {"id": "allow_ops", "tool_name": r"^ops__.*$", "decision": "allow"},
            ],
            default_action="allow",
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
        )
        namespace = {"type": "namespace", "name": "ops", "tools": [{"type": "function", "name": "exec"}]}
        data = {"guardrails": ["post-exec"], "tools": [namespace]}
        function = Function(name="exec", arguments='{"command": "rm -rf /data"}')
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={"tool_calls": [{"id": "call_exec", "type": "function", "function": function}]},
                )
            ]
        )

        await guardrail.async_post_call_success_hook(data=data, user_api_key_dict=UserAPIKeyAuth(), response=response)

        message = response.choices[0].message
        assert [call.function.name for call in message.tool_calls or ()] == []
        assert "Tool 'exec'" in (message.content or "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("rules", "command", "kept"),
        [
            (
                [
                    {
                        "id": "allow_ls",
                        "tool_name": r"^exec$",
                        "decision": "allow",
                        "allowed_param_patterns": {"command": r"^ls$"},
                    },
                    {"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"},
                ],
                "ls",
                ["run"],
            ),
            (
                [
                    {
                        "id": "allow_ls",
                        "tool_name": r"^exec$",
                        "decision": "allow",
                        "allowed_param_patterns": {"command": r"^ls$"},
                    },
                    {"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"},
                ],
                "pwd",
                [],
            ),
            (
                [
                    {"id": "allow_run", "tool_name": r"^run$", "decision": "allow"},
                    {"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"},
                ],
                "ls",
                [],
            ),
        ],
        ids=["namespace_rule_allows", "namespace_rule_denies", "namespace_deny_beats_member_allow"],
    )
    async def test_post_call_judges_a_returned_namespace_member_by_a_rule_on_its_namespace_name(
        self, rules: list[dict[str, object]], command: str, kept: list[str]
    ) -> None:
        guardrail = ToolPermissionGuardrail(
            guardrail_name="post-exec",
            rules=rules,
            default_action="allow",
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
        )
        namespace = {"type": "namespace", "name": "exec", "tools": [{"type": "function", "name": "run"}]}
        function = Function(name="run", arguments=json.dumps({"command": command}))
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={"tool_calls": [{"id": "call_run", "type": "function", "function": function}]},
                )
            ]
        )

        await guardrail.async_post_call_success_hook(
            data={"guardrails": ["post-exec"], "tools": [namespace]},
            user_api_key_dict=UserAPIKeyAuth(),
            response=response,
        )

        message = response.choices[0].message
        assert [call.function.name for call in message.tool_calls or ()] == kept
        assert ("Tool 'exec'" in (message.content or "")) == (not kept)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("namespace_first", [True, False], ids=["namespace_first", "top_level_first"])
    async def test_post_call_judges_a_returned_name_a_top_level_function_carries_as_that_function(
        self, namespace_first: bool
    ) -> None:
        guardrail = ToolPermissionGuardrail(
            guardrail_name="post-mcp",
            rules=[{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}],
            default_action="allow",
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
        )
        namespace = {"type": "namespace", "name": "mcp__weather", "tools": [{"type": "function", "name": "lookup"}]}
        top_level = {"type": "function", "function": {"name": "lookup"}}
        tools = [namespace, top_level] if namespace_first else [top_level, namespace]
        function = Function(name="lookup", arguments='{"country": "France"}')
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={"tool_calls": [{"id": "call_lookup", "type": "function", "function": function}]},
                )
            ]
        )

        await guardrail.async_post_call_success_hook(
            data={"guardrails": ["post-mcp"], "tools": tools}, user_api_key_dict=UserAPIKeyAuth(), response=response
        )

        assert [call.function.name for call in response.choices[0].message.tool_calls or ()] == ["lookup"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "same_named_tool",
        [
            {"type": "mcp", "server_label": "lookup", "server_url": "https://mcp.example.com/sse"},
            {"type": "mcp_toolset", "mcp_server_name": "lookup"},
            {"ignored": {"name": "lookup"}},
            {"type": "custom", "custom": {"name": "lookup"}},
            {"functionDeclarations": [{"name": "lookup"}]},
        ],
        ids=["mcp", "mcp_toolset", "unknown_keyed_block", "chat_custom", "native_declaration"],
    )
    async def test_post_call_judges_a_returned_namespace_member_as_that_member_beside_a_same_named_non_function(
        self, same_named_tool: dict[str, object]
    ) -> None:
        guardrail = ToolPermissionGuardrail(
            guardrail_name="post-mcp",
            rules=[{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}],
            default_action="allow",
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
        )
        namespace = {"type": "namespace", "name": "mcp__weather", "tools": [{"type": "function", "name": "lookup"}]}
        function = Function(name="lookup", arguments='{"country": "France"}')
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={"tool_calls": [{"id": "call_lookup", "type": "function", "function": function}]},
                )
            ]
        )

        await guardrail.async_post_call_success_hook(
            data={"guardrails": ["post-mcp"], "tools": [namespace, same_named_tool]},
            user_api_key_dict=UserAPIKeyAuth(),
            response=response,
        )

        assert [call.function.name for call in response.choices[0].message.tool_calls or ()] == []

    @pytest.mark.asyncio
    async def test_post_call_judges_a_call_tagged_with_a_namespace_as_that_member_beside_a_same_named_function(
        self,
    ) -> None:
        guardrail = ToolPermissionGuardrail(
            guardrail_name="post-mcp",
            rules=[{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}],
            default_action="allow",
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
        )
        namespace = {"type": "namespace", "name": "mcp__weather", "tools": [{"type": "function", "name": "lookup"}]}
        top_level = {"type": "function", "function": {"name": "lookup"}}
        tagged_call = {
            "id": "call_lookup",
            "type": "function",
            "function": Function(name="lookup", arguments='{"country": "France"}'),
            "provider_specific_fields": {"namespace": "mcp__weather"},
        }
        response = ModelResponse(
            choices=[Choices(finish_reason="tool_calls", message={"tool_calls": [tagged_call]})]
        )

        await guardrail.async_post_call_success_hook(
            data={"guardrails": ["post-mcp"], "tools": [namespace, top_level]},
            user_api_key_dict=UserAPIKeyAuth(),
            response=response,
        )

        message = response.choices[0].message
        assert [call.function.name for call in message.tool_calls or ()] == []
        assert "Tool 'mcp__weather__lookup' denied by rule 'deny_mcp'" in (message.content or "")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("decision", "default_action", "kept"),
        [("allow", "deny", ["weather__get_weather"]), ("deny", "allow", [])],
        ids=["bare_name_allowed", "bare_name_denied"],
    )
    async def test_post_call_judges_a_returned_qualified_member_name_by_the_member_names_pre_call_judged(
        self, decision: Literal["allow", "deny"], default_action: Literal["allow", "deny"], kept: list[str]
    ) -> None:
        guardrail = ToolPermissionGuardrail(
            guardrail_name="post-weather",
            rules=[{"id": "get_weather", "tool_name": r"^get_weather$", "decision": decision}],
            default_action=default_action,
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
        )
        namespace = {"type": "namespace", "name": "weather", "tools": [{"type": "function", "name": "get_weather"}]}
        top_level = {"type": "function", "function": {"name": "get_weather"}}
        function = Function(name="weather__get_weather", arguments='{"city": "Istanbul"}')
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={"tool_calls": [{"id": "call_weather", "type": "function", "function": function}]},
                )
            ]
        )

        await guardrail.async_post_call_success_hook(
            data={"guardrails": ["post-weather"], "tools": [namespace, top_level]},
            user_api_key_dict=UserAPIKeyAuth(),
            response=response,
        )

        assert [call.function.name for call in response.choices[0].message.tool_calls or ()] == kept

    @pytest.mark.asyncio
    async def test_post_call_judges_a_top_level_function_by_its_own_name_beside_a_member_qualified_the_same(
        self,
    ) -> None:
        guardrail = ToolPermissionGuardrail(
            guardrail_name="post-exec",
            rules=[{"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"}],
            default_action="allow",
            on_disallowed_action="rewrite",
            event_hook=GuardrailEventHooks.post_call,
        )
        namespace = {"type": "namespace", "name": "ops", "tools": [{"type": "function", "name": "exec"}]}
        top_level = {"type": "function", "function": {"name": "ops__exec"}}
        function = Function(name="ops__exec", arguments="{}")
        response = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={"tool_calls": [{"id": "call_ops", "type": "function", "function": function}]},
                )
            ]
        )

        await guardrail.async_post_call_success_hook(
            data={"guardrails": ["post-exec"], "tools": [namespace, top_level]},
            user_api_key_dict=UserAPIKeyAuth(),
            response=response,
        )

        assert [call.function.name for call in response.choices[0].message.tool_calls or ()] == ["ops__exec"]

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


    def _server_search_response(self) -> ModelResponse:
        return ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={
                        "content": "Searching.",
                        "tool_calls": [_ALLOWED_BASH_CALL],
                        "provider_specific_fields": {"server_tool_calls": [_SERVER_WEB_SEARCH_CALL]},
                    },
                )
            ]
        )

    @pytest.mark.asyncio
    async def test_denied_server_tool_call_is_blocked(self):
        with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                await self.guardrail.async_post_call_success_hook(
                    data={}, user_api_key_dict=UserAPIKeyAuth(), response=self._server_search_response()
                )

    @pytest.mark.asyncio
    async def test_denied_server_tool_call_in_a_chat_stream_is_blocked(self):
        async def _stream():
            yield ModelResponseStream(
                choices=[
                    StreamingChoices(
                        index=0,
                        delta=Delta(
                            content="Searching.",
                            provider_specific_fields={"server_tool_calls": [{**_SERVER_WEB_SEARCH_CALL, "index": 0}]},
                        ),
                        finish_reason="stop",
                    )
                ]
            )

        with patch.object(self.guardrail, "should_run_guardrail", return_value=True):
            with pytest.raises(GuardrailRaisedException):
                async for _ in self.guardrail.async_post_call_streaming_iterator_hook(
                    user_api_key_dict=UserAPIKeyAuth(), response=_stream(), request_data={}
                ):
                    pass

    @pytest.mark.asyncio
    async def test_rewrite_mode_removes_the_denied_server_tool_call(self):
        rewriting: Final = ToolPermissionGuardrail(
            guardrail_name="test-tool-permission",
            rules=self.test_rules,
            default_action="deny",
            on_disallowed_action="rewrite",
        )

        with patch.object(rewriting, "should_run_guardrail", return_value=True):
            result: Final = await rewriting.async_post_call_success_hook(
                data={}, user_api_key_dict=UserAPIKeyAuth(), response=self._server_search_response()
            )

        message: Final = result.choices[0].message
        assert message.provider_specific_fields["server_tool_calls"] == []
        assert [tool_call.id for tool_call in message.tool_calls] == [_ALLOWED_BASH_CALL["id"]]
        assert "Permission denied" in message.content

    @pytest.mark.asyncio
    async def test_rewrite_mode_removes_the_denied_server_tool_calls_hosted_copy(self):
        rewriting: Final = ToolPermissionGuardrail(
            guardrail_name="test-tool-permission",
            rules=self.test_rules,
            default_action="deny",
            on_disallowed_action="rewrite",
        )
        response: Final = self._server_search_response()
        response.choices[0].message.provider_specific_fields = {
            **response.choices[0].message.provider_specific_fields,
            "web_search_calls": [
                build_web_search_call("srvtoolu_01Search", {"query": "weather in Paris"}, {"content": []})
            ],
        }

        with patch.object(rewriting, "should_run_guardrail", return_value=True):
            result: Final = await rewriting.async_post_call_success_hook(
                data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
            )

        assert result.choices[0].message.provider_specific_fields["web_search_calls"] == []

    @pytest.mark.asyncio
    async def test_rewrite_mode_removes_the_results_of_denied_server_tool_calls(self):
        rewriting: Final = ToolPermissionGuardrail(
            guardrail_name="test-tool-permission",
            rules=self.test_rules,
            default_action="deny",
            on_disallowed_action="rewrite",
        )
        mcp_call: Final = {
            "id": "mcptoolu_01Wiki",
            "type": "function",
            "function": {
                "name": "read_wiki_structure",
                "arguments": "{}",
                "provider_specific_fields": {"server_name": "deepwiki"},
            },
        }
        unrelated_result: Final = {"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_01Kept", "content": {}}
        response: Final = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={
                        "content": "Reading.",
                        "tool_calls": [_ALLOWED_BASH_CALL],
                        "provider_specific_fields": {
                            "server_tool_calls": [_SERVER_WEB_SEARCH_CALL, mcp_call],
                            "web_search_results": [
                                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_01Search", "content": []}
                            ],
                            "tool_results": [
                                {
                                    "type": "mcp_tool_result",
                                    "tool_use_id": "mcptoolu_01Wiki",
                                    "content": [{"type": "text", "text": "wiki page"}],
                                },
                                unrelated_result,
                            ],
                        },
                    },
                )
            ]
        )

        with patch.object(rewriting, "should_run_guardrail", return_value=True):
            result: Final = await rewriting.async_post_call_success_hook(
                data={}, user_api_key_dict=UserAPIKeyAuth(), response=response
            )

        fields: Final = result.choices[0].message.provider_specific_fields
        assert (fields["server_tool_calls"], fields["web_search_results"]) == ([], [])
        assert fields["tool_results"] == [unrelated_result]

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
        ("call_type", "malformed"),
        [
            (
                "agenerate_content",
                {"toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [{"a": 1}]}}},
            ),
            ("responses", {"input": [{"type": {"x": 1}, "content": "hi"}]}),
            ("acompletion", {"extra_body": {"input": [{"type": [1]}]}}),
            ("acompletion", {"extra_body": {"function_call": {"name": 5}}}),
        ],
        ids=["unhashable_allowed_function_name", "unhashable_input_type", "extra_body_input_type", "int_function_call"],
    )
    async def test_pre_call_rewrite_leaves_a_malformed_value_for_the_provider_to_reject(
        self, call_type: CallTypesLiteral, malformed: dict[str, object]
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"}], "rewrite", "deny"
        )
        tools = [{"type": "function", "function": {"name": name}} for name in ("get_weather", "shell")]

        result = await self._run_pre_call(guardrail, {"model": "m", "tools": tools, **malformed}, call_type)

        assert result["tools"] == tools[:1]
        assert {key: result[key] for key in malformed} == malformed

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
        ("rules", "default_action", "kept"),
        [
            ([{"id": "allow_exec", "tool_name": r"^exec$", "decision": "allow"}], "deny", ["run"]),
            ([{"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"}], "allow", []),
            (
                [
                    {"id": "allow_run", "tool_name": r"^run$", "decision": "allow"},
                    {"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"},
                ],
                "allow",
                [],
            ),
            (
                [
                    {"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"},
                    {"id": "allow_all", "tool_name": r"^.*$", "decision": "allow"},
                ],
                "allow",
                [],
            ),
            (
                [
                    {"id": "allow_exec", "tool_name": r"^exec$", "decision": "allow"},
                    {"id": "deny_all", "tool_name": r"^.*$", "decision": "deny"},
                ],
                "deny",
                ["run"],
            ),
        ],
        ids=[
            "namespace_allowed",
            "namespace_denied",
            "namespace_deny_beats_member_allow",
            "namespace_deny_beats_a_catch_all_allow",
            "namespace_allow_beats_a_catch_all_deny",
        ],
    )
    async def test_pre_call_judges_a_namespace_member_by_a_rule_on_its_namespace_name(
        self, rules: list[dict[str, str]], default_action: Literal["allow", "deny"], kept: list[str]
    ) -> None:
        namespace = {"type": "namespace", "name": "exec", "tools": [{"type": "function", "name": "run"}]}
        request = {"model": "gpt-5", "input": "Run pwd", "tools": [namespace]}

        rewritten = await self._run_pre_call(self._pre_call(rules, "rewrite", default_action), request, "responses")

        tools = rewritten.get("tools") or []
        assert [member["name"] for member in chain.from_iterable(tool["tools"] for tool in tools)] == kept

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
    async def test_pre_call_checks_and_strips_tools_loaded_in_extra_body_input(self) -> None:
        rules = [{"id": "deny_mcp", "tool_name": r"^mcp__.*$", "decision": "deny"}]
        get_weather = {"type": "function", "name": "get_weather"}
        lookup = {"type": "function", "name": "mcp__weather__lookup"}
        user = {"role": "user", "content": "hi"}
        request = {
            "model": "gpt-5",
            "input": "hi",
            "tools": [],
            "extra_body": {"input": [user, {"type": "tool_search_output", "tools": [get_weather, lookup]}], "seed": 7},
        }

        message, rewritten = await self._blocked_and_rewritten(rules, request, "responses")

        assert message == "Tool 'mcp__weather__lookup' denied by rule 'deny_mcp'"
        assert rewritten["extra_body"] == {
            "input": [user, {"type": "tool_search_output", "tools": [get_weather]}],
            "seed": 7,
        }

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
    async def test_rewrite_relaxes_a_required_tool_choice_when_the_extra_body_tools_sent_lose_every_tool(
        self,
    ) -> None:
        guardrail = self._pre_call([{"id": "deny_delete", "tool_name": r"^delete_db$", "decision": "deny"}], "rewrite")
        weather = {"type": "function", "function": {"name": "get_weather"}}
        delete_db = {"type": "function", "function": {"name": "delete_db"}}
        data = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [weather],
            "extra_body": {"tools": [delete_db], "tool_choice": "required", "seed": 7},
        }

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [weather]
        assert result["extra_body"] == {"tools": [], "tool_choice": "none", "seed": 7}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("call_type", "tool_choice"),
        [
            ("acompletion", {"type": "function", "function": {"name": "Read"}}),
            ("responses", {"type": "function", "name": "Read"}),
        ],
        ids=["chat", "responses"],
    )
    async def test_pre_call_checks_an_extra_body_tool_choice_that_names_an_unlisted_tool(
        self, call_type: CallTypesLiteral, tool_choice: dict[str, object]
    ) -> None:
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "input": "hi",
            "tools": [{"type": "function", "name": "Bash"}],
            "extra_body": {"tool_choice": tool_choice, "seed": 7},
        }

        message, rewritten = await self._blocked_and_rewritten(self.rules, request, call_type)

        assert message == "Tool 'Read' denied by rule 'deny_read'"
        assert rewritten["extra_body"] == {"tool_choice": "none", "seed": 7}

    @pytest.mark.asyncio
    async def test_pre_call_checks_and_strips_legacy_functions_in_extra_body(self) -> None:
        rules = [{"id": "deny_delete", "tool_name": r"^delete_db$", "decision": "deny"}]
        weather = {"name": "get_weather", "parameters": {"type": "object"}}
        delete_db = {"name": "delete_db", "parameters": {"type": "object"}}
        request = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"functions": [weather, delete_db], "function_call": {"name": "delete_db"}, "seed": 7},
        }

        message, rewritten = await self._blocked_and_rewritten(rules, request, "acompletion")

        assert message == "Tool 'delete_db' denied by rule 'deny_delete'"
        assert rewritten["extra_body"] == {"functions": [weather], "function_call": "none", "seed": 7}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "fields",
        [
            {"functions": [{"name": "get_weather"}, {"name": "delete_db"}], "function_call": "delete_db"},
            {
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "mcp_servers": [{"type": "url", "url": "https://mcp.example.com/mcp", "name": "delete_db"}],
                "tool_choice": "required",
            },
            {
                "tools": [{"functionDeclarations": [{"name": "get_weather"}, {"name": "delete_db"}]}],
                "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["delete_db"]}},
            },
        ],
        ids=["legacy_functions", "mcp_servers", "gemini_tool_config"],
    )
    async def test_extra_body_fields_are_judged_and_rewritten_like_the_same_top_level_fields(
        self, fields: dict[str, object]
    ) -> None:
        rules = [{"id": "deny_delete", "tool_name": r"^delete_db$", "decision": "deny"}]
        base = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        top_request = {**base, **copy.deepcopy(fields)}
        extra_request = {**base, "extra_body": {**copy.deepcopy(fields), "seed": 7}}

        top_message, top = await self._blocked_and_rewritten(rules, top_request, "acompletion")
        extra_message, extra = await self._blocked_and_rewritten(rules, extra_request, "acompletion")

        top_fields = {key: top[key] for key in fields}
        assert top_fields != fields
        assert extra_message == top_message
        assert extra["extra_body"] == {**top_fields, "seed": 7}

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
    @pytest.mark.parametrize(
        ("choice", "relaxed"),
        [
            ({"tool_choice": {"type": "function", "function": {"name": "exec"}}}, {"tool_choice": "none"}),
            ({"tool_choice": {"type": "function", "function": {"name": "ops"}}}, {"tool_choice": "none"}),
            (
                {
                    "tool_choice": {
                        "type": "allowed_tools",
                        "allowed_tools": {
                            "mode": "required",
                            "tools": [
                                {"type": "function", "function": {"name": "exec"}},
                                {"type": "function", "function": {"name": "get_weather"}},
                            ],
                        },
                    }
                },
                {
                    "tool_choice": {
                        "type": "allowed_tools",
                        "allowed_tools": {
                            "mode": "required",
                            "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                        },
                    }
                },
            ),
            (
                {"toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["exec"]}}},
                {"toolConfig": {"functionCallingConfig": {"mode": "NONE"}}},
            ),
        ],
        ids=["bare_member_choice", "namespace_choice", "allowed_tools_entry", "allowed_function_names"],
    )
    async def test_pre_call_rewrite_relaxes_a_bare_name_a_dropped_namespace_member_was_sent_under(
        self, choice: dict[str, object], relaxed: dict[str, object]
    ) -> None:
        guardrail = self._pre_call([{"id": "deny_exec", "tool_name": r"^ops__exec$", "decision": "deny"}], "rewrite")
        weather = {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
        namespace = {"type": "namespace", "name": "ops", "tools": [{"type": "function", "name": "exec"}]}
        data = {
            "model": "gemini/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [weather, namespace],
            **choice,
        }

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [weather]
        assert {key: result[key] for key in relaxed} == relaxed

    @pytest.mark.asyncio
    async def test_pre_call_rewrite_keeps_a_bare_name_another_kept_namespace_still_serves(self) -> None:
        guardrail = self._pre_call([{"id": "deny_exec", "tool_name": r"^ops__exec$", "decision": "deny"}], "rewrite")
        dropped_namespace = {"type": "namespace", "name": "ops", "tools": [{"type": "function", "name": "exec"}]}
        kept_namespace = {"type": "namespace", "name": "dev", "tools": [{"type": "function", "name": "exec"}]}
        tool_choice = {"type": "function", "function": {"name": "exec"}}
        data = {
            "model": "gemini/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [dropped_namespace, kept_namespace],
            "tool_choice": tool_choice,
        }

        result = await self._run_pre_call(guardrail, data, "acompletion")

        assert result["tools"] == [kept_namespace]
        assert result["tool_choice"] == tool_choice

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exec_first", [True, False], ids=["exec_namespace_first", "exec_namespace_last"])
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    async def test_pre_call_denies_a_bare_name_another_namespace_shares_a_qualified_name_with(
        self, on_disallowed_action: Literal["block", "rewrite"], exec_first: bool
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "deny_exec", "tool_name": r"^exec$", "decision": "deny"}], on_disallowed_action
        )
        exec_namespace = {"type": "namespace", "name": "mcp__ops", "tools": [{"type": "function", "name": "exec"}]}
        twin_namespace = {"type": "namespace", "name": "mcp", "tools": [{"type": "function", "name": "ops__exec"}]}
        tools = [exec_namespace, twin_namespace] if exec_first else [twin_namespace, exec_namespace]
        data = {"model": "gpt-5", "input": "run date", "tools": tools}

        if on_disallowed_action == "rewrite":
            assert (await self._run_pre_call(guardrail, data, "responses"))["tools"] == []
            return
        with pytest.raises(HTTPException) as excinfo:
            await self._run_pre_call(guardrail, data, "responses")
        assert excinfo.value.detail["detection_message"] == "Tool 'exec' denied by rule 'deny_exec'"

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
    @pytest.mark.parametrize("on_disallowed_action", ["block", "rewrite"])
    async def test_pre_call_judges_a_gemini_hosted_key_by_the_key_whatever_name_it_carries(
        self, on_disallowed_action: Literal["block", "rewrite"]
    ) -> None:
        guardrail = self._pre_call(
            [{"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"}],
            on_disallowed_action,
            default_action="deny",
        )
        weather = {"type": "function", "function": {"name": "get_weather"}}
        computer_use = {"computerUse": {"name": "get_weather", "environment": "browser"}}
        data = {"model": "gemini-3", "messages": [{"role": "user", "content": "hi"}], "tools": [weather, computer_use]}

        if on_disallowed_action == "block":
            with pytest.raises(HTTPException) as excinfo:
                await self._run_pre_call(guardrail, data, "acompletion")
            assert excinfo.value.detail["detection_message"] == "Tool 'computerUse' denied by default action"
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

        with pytest.raises(GuardrailRaisedException, match="deny_read"):
            await self._drain(self.blocking, chunks)

    @pytest.mark.asyncio
    async def test_denied_tool_use_split_over_data_lines_beside_an_upstream_error_is_blocked(self):
        def over_two_data_lines(frame: bytes) -> bytes:
            return frame.replace(b", ", b",\ndata: ", 1)

        chunks: Final = self._ended_by_upstream_error([over_two_data_lines(frame) for frame in self._sse_chunks("Read")])

        with pytest.raises(GuardrailRaisedException, match="deny_read"):
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
        "behind-a-form-feed": tuple(
            b'data: {"type": "ping"}\x0c' + frame.split(b"\n", 1)[1] for frame in DENIED_CALL
        ),
    }
    DENIED_CALLS_IN_FRAMES_A_CLIENT_READS: Final = {
        "finished-mcp-call": tuple(frame.replace(b'"tool_use"', b'"mcp_tool_use"') for frame in DENIED_CALL),
        "cr-line-ends": tuple(frame.replace(b"\n", b"\r") for frame in DENIED_CALL),
        "behind-a-glued-crlf-ping": (b'event: ping\r\ndata: {"type": "ping"}\r\n\r\n' + b"".join(DENIED_CALL),),
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
    @pytest.mark.parametrize(
        "frames", DENIED_CALLS_IN_FRAMES_A_CLIENT_READS.values(), ids=DENIED_CALLS_IN_FRAMES_A_CLIENT_READS
    )
    async def test_denied_call_in_frames_a_client_reads_is_blocked_past_an_upstream_error(
        self, frames: tuple[bytes, ...]
    ):
        chunks: Final = self._ended_by_upstream_error([self._sse_chunks("Read")[0], *frames])

        with pytest.raises(GuardrailRaisedException, match="deny_read"):
            await self._drain(self.blocking, chunks)

    @pytest.mark.asyncio
    async def test_denied_call_inside_message_start_never_reaches_the_client_past_an_upstream_error(self):
        message_start, *_ = self._sse_chunks("Read")
        call: Final = json.dumps(self._tool_use("Read")).encode()
        chunks: Final = self._ended_by_upstream_error([message_start.replace(b'"content": []', b'"content": [' + call + b"]")])

        out: Final = await self._drain(self.blocking, chunks)

        assert b"Read" not in b"".join(out)
        assert self._events(out)[-1] == self.UPSTREAM_ERROR

    @pytest.mark.asyncio
    async def test_denied_tool_input_whose_deltas_interleave_before_an_upstream_error_is_blocked(self):
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
            on_disallowed_action="rewrite",
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

        assert {"command": "rm -rf /"} not in _tool_inputs_read_line_by_line(out)
        assert self._events(out)[-1] == self.UPSTREAM_ERROR

    @pytest.mark.asyncio
    async def test_denied_tool_use_named_only_by_its_event_lines_beside_an_upstream_error_is_blocked(self):
        def without_type(frame: bytes) -> bytes:
            return re.sub(rb'"type": "(?:content_block_\w+|message_\w+)"', b'"error": {}', frame, count=1)

        chunks: Final = self._ended_by_upstream_error([without_type(frame) for frame in self._sse_chunks("Read")])

        with pytest.raises(GuardrailRaisedException, match="deny_read"):
            await self._drain(self.blocking, chunks)

    @pytest.mark.asyncio
    async def test_denied_tool_use_between_space_led_data_lines_is_blocked(self):
        def wrapped(frame: bytes) -> bytes:
            return b' data: {"type": "error", "error": {}, "w": [\n' + frame.rstrip(b"\n") + b"\n data: ]}\n\n"

        chunks: Final = [wrapped(frame) for frame in self._sse_chunks("Read")]

        with pytest.raises(GuardrailRaisedException, match="deny_read"):
            await self._drain(self.blocking, chunks)

    @pytest.mark.asyncio
    async def test_denied_tool_use_delivered_before_an_upstream_error_is_still_blocked(self):
        chunks: Final = self._ended_by_upstream_error(self._sse_chunks("Read")[:7])

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await self._drain(self.blocking, chunks)

        assert "deny_read" in str(exc_info.value)

    MCP_CALL: Final = {"type": "mcp_tool_use", "id": "mcptoolu_01", "server_name": "deepwiki", "input": {"command": "ls"}}
    MCP_RESULT: Final = {
        "type": "mcp_tool_result",
        "tool_use_id": "mcptoolu_01",
        "is_error": False,
        "content": [{"type": "text", "text": "wiki page"}],
    }

    def _finished_mcp_call(self, index: int, name: str) -> list[bytes]:
        events: Final = (
            {"type": "content_block_start", "index": index, "content_block": {**self.MCP_CALL, "name": name, "input": {}}},
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(self.MCP_CALL["input"])},
            },
            {"type": "content_block_stop", "index": index},
            {"type": "content_block_start", "index": index + 1, "content_block": self.MCP_RESULT},
            {"type": "content_block_stop", "index": index + 1},
        )
        return [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events]

    def _ended_by_end_turn(self, chunks: list[bytes]) -> list[bytes]:
        message_delta, message_stop = self._sse_chunks("Bash")[-2:]
        return [*chunks, message_delta.replace(b'"tool_use"', b'"end_turn"'), message_stop]

    def _mcp_blocks(self, chunks: list[bytes]) -> list[dict[str, object]]:
        return [
            block
            for event in self._events(chunks)
            if isinstance(block := event.get("content_block"), dict) and str(block.get("type")).startswith("mcp_tool_")
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cut_short", [False, True], ids=["end-turn", "upstream-error"])
    async def test_finished_mcp_call_named_by_a_deny_rule_is_blocked(self, cut_short: bool):
        delivered: Final = [*self._sse_chunks("Bash")[:4], *self._finished_mcp_call(1, "Read")]
        chunks: Final = self._ended_by_upstream_error(delivered) if cut_short else self._ended_by_end_turn(delivered)

        with pytest.raises(GuardrailRaisedException) as exc_info:
            await self._drain(self.blocking, chunks)

        assert "deny_read" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rewrite_drops_a_denied_mcp_call_and_its_result(self):
        chunks: Final = self._ended_by_end_turn([*self._sse_chunks("Bash")[:4], *self._finished_mcp_call(1, "Read")])

        out: Final = await self._drain(self.rewriting, chunks)

        assert self._mcp_blocks(out) == []
        assert b"Permission denied" in b"".join(out)

    @pytest.mark.asyncio
    async def test_rewrite_of_a_denied_client_call_keeps_the_allowed_mcp_call_and_its_result(self):
        source: Final = self._sse_chunks("Read")
        chunks: Final = [*source[:7], *self._finished_mcp_call(2, "Bash"), *source[7:]]

        out: Final = await self._drain(self.rewriting, chunks)

        assert self._mcp_blocks(out) == [{**self.MCP_CALL, "name": "Bash"}, self.MCP_RESULT]
        assert self._tool_use_blocks(self._events(out)) == []

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


def _sse_frame(event: dict[str, object], name: str | None = None) -> bytes:
    return f"event: {name or event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _tool_start(index: int, name: str, tool_input: dict[str, object] | None = None) -> dict[str, object]:
    block: Final = {
        "type": "tool_use",
        "id": f"tu_{index}",
        "name": name,
        "input": tool_input or {},
        "caller": {"type": "direct"},
    }
    return {"type": "content_block_start", "index": index, "content_block": block}


def _input_delta(index: int, partial_json: str) -> dict[str, object]:
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }


def _block_stop(index: int) -> dict[str, object]:
    return {"type": "content_block_stop", "index": index}


_LS: Final = {"command": "ls"}
_RM_RF: Final = {"command": "rm -rf /"}
_OPENING: Final = tuple(
    _sse_frame(event)
    for event in (
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 10, "output_tokens": 0, "service_tier": "standard", "inference_geo": "us"},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "working"}},
        _block_stop(0),
    )
)
_CLOSING: Final = (
    _sse_frame({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
    _sse_frame({"type": "message_stop"}),
)


def _stream(*call_frames: bytes) -> tuple[bytes, ...]:
    return (*_OPENING, *call_frames, *_CLOSING)


def _bash_call(command: dict[str, str]) -> tuple[bytes, ...]:
    return tuple(
        _sse_frame(event) for event in (_tool_start(1, "Bash"), _input_delta(1, json.dumps(command)), _block_stop(1))
    )


def _bash_start_behind_a_typescript_input_buffer(tool_input: dict[str, object] | None = None) -> dict[str, object]:
    start: Final = _tool_start(1, "Bash", tool_input)
    block: Final = start["content_block"]
    assert isinstance(block, dict)
    return {**start, "content_block": {**block, "__json_buf": '{"command": "rm -rf /", "ignored": '}}


def _halves(value: dict[str, object]) -> tuple[str, str]:
    text: Final = json.dumps(value)
    return text[: len(text) // 2], text[len(text) // 2 :]


_RM_RF_CALL: Final = _bash_call(_RM_RF)
_DENIED_CALL_SHAPES: Final = {
    "commented-frames": _stream(*(b": data: keepalive\n" + frame for frame in _RM_RF_CALL)),
    "crlf-frames": _stream(*(frame.replace(b"\n", b"\r\n") for frame in _RM_RF_CALL)),
    "cr-frames": _stream(*(frame.replace(b"\n", b"\r") for frame in _RM_RF_CALL)),
    "over-many-data-lines": _stream(*(frame.replace(b", ", b",\ndata: ") for frame in _RM_RF_CALL)),
    "byte-order-mark": (b"\xef\xbb\xbf" + _OPENING[0].split(b"\n", 1)[1], *_stream(*_RM_RF_CALL)[1:]),
    "typed-only-by-event-names": _stream(
        *(re.sub(rb'"type": "content_block_\w+", ', b"", frame, count=1) for frame in _RM_RF_CALL)
    ),
    "inside-message-start": (
        _OPENING[0].replace(
            b'"content": []',
            b'"content": ['
            + json.dumps({"type": "tool_use", "id": "tu_9", "name": "Bash", "input": _RM_RF}).encode()
            + b"]",
        ),
        *_OPENING[1:],
        *_CLOSING,
    ),
    "interleaved-input-deltas": _stream(
        *(
            _sse_frame(event)
            for event in (
                _tool_start(1, "Read"),
                _tool_start(2, "Bash"),
                _input_delta(1, json.dumps(_LS)),
                {"type": "content_block_start", "index": 3, "content_block": {"type": "text", "text": ""}},
                _input_delta(2, json.dumps(_RM_RF)),
                _block_stop(1),
                _block_stop(2),
                _block_stop(3),
            )
        )
    ),
    "input-finished-by-a-ping-named-frame": _stream(
        _sse_frame(_tool_start(1, "Bash")),
        _sse_frame(_input_delta(1, '{"command": "rm -rf /"')),
        _sse_frame(_input_delta(1, ', "command": "ls"}'), name="ping"),
        _sse_frame(_block_stop(1)),
    ),
    "input-finished-by-a-frame-the-stream-never-ended": (
        *_OPENING,
        _sse_frame(_tool_start(1, "Bash")),
        _sse_frame(_input_delta(1, '{"command": "rm -rf /"')),
        _sse_frame(_input_delta(1, ', "command": "ls"}')).removesuffix(b"\n\n"),
    ),
    "input-finished-by-a-thinking-delta": _stream(
        _sse_frame(_tool_start(1, "Bash")),
        _sse_frame(_input_delta(1, '{"command": "rm -rf /"')),
        _sse_frame(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "thinking_delta", "partial_json": ', "command": "ls"}'},
            }
        ),
        _sse_frame(_block_stop(1)),
    ),
    "text-block-stopped-only-at-the-end": (
        *_OPENING[:3],
        *(_sse_frame(event) for event in (_tool_start(1, "Bash"), _input_delta(1, json.dumps(_RM_RF)))),
        *(_sse_frame(_block_stop(index)) for index in (0, 1)),
        *_CLOSING,
    ),
    "no-block-stops": (
        *_OPENING[:3],
        *(_sse_frame(event) for event in (_tool_start(1, "Bash"), _input_delta(1, json.dumps(_RM_RF)))),
        *_CLOSING,
    ),
    "input-hidden-behind-a-text-field": _stream(
        _sse_frame(_tool_start(1, "Bash")),
        _sse_frame(_input_delta(1, '{"command": "ls"')),
        _sse_frame(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "text": "", "partial_json": ', "command": "rm -rf /"'},
            }
        ),
        _sse_frame(_input_delta(1, "}")),
        _sse_frame(_block_stop(1)),
    ),
}


_SHAPES_THE_CHECK_READS_AS_THE_ALLOWED_CALL: Final = frozenset(
    {"input-finished-by-a-ping-named-frame", "input-finished-by-a-frame-the-stream-never-ended"}
)
_SHAPES_WHOSE_CALL_THE_CHECK_DENIES: Final = {
    shape: chunks
    for shape, chunks in _DENIED_CALL_SHAPES.items()
    if shape not in _SHAPES_THE_CHECK_READS_AS_THE_ALLOWED_CALL
}


def _call_in_a_comment_broken_by(line_break: bytes) -> tuple[bytes, ...]:
    return _stream(b": keepalive" + line_break * 2 + b"".join(_RM_RF_CALL).replace(b"\n", line_break) + b"\n")


def _call_as_data_lines_of_one_frame(name_line: bytes) -> tuple[bytes, ...]:
    events: Final = (_tool_start(1, "Bash"), _input_delta(1, json.dumps(_RM_RF)), _block_stop(1))
    return _stream(name_line + b"".join(b"data: " + json.dumps(event).encode() + b"\n" for event in events) + b"\n")


def _overlapped_calls(first_command: dict[str, str], second_command: dict[str, str]) -> tuple[bytes, ...]:
    return _stream(
        *(
            _sse_frame(event)
            for event in (
                _tool_start(1, "Bash"),
                _tool_start(2, "Bash"),
                _input_delta(1, json.dumps(first_command)),
                _block_stop(1),
                _input_delta(2, json.dumps(second_command)),
                _block_stop(2),
            )
        )
    )


_DENIED_CALLS_ONLY_A_LINE_SPLITTING_CLIENT_READS: Final = {
    "form-feed": _call_in_a_comment_broken_by(b"\x0c"),
    "line-separator": _call_in_a_comment_broken_by("\u2028".encode()),
    "data-lines-of-a-keepalive-frame": _call_as_data_lines_of_one_frame(b"event: keepalive\n"),
    "data-lines-of-a-ping-frame": _call_as_data_lines_of_one_frame(b"event: ping\n"),
    "data-lines-of-an-unnamed-frame": _call_as_data_lines_of_one_frame(b""),
}
_LS_STREAM: Final = _stream(*_bash_call(_LS))
_LS_ONLY_STREAM: Final = (
    _OPENING[0],
    *(_sse_frame(event) for event in (_tool_start(0, "Bash"), _input_delta(0, json.dumps(_LS)), _block_stop(0))),
    *_CLOSING,
)
_ALLOWED_CALL_SHAPES: Final = {
    "crlf": tuple(frame.replace(b"\n", b"\r\n") for frame in _LS_STREAM),
    "commented": tuple(b": data: keepalive\n" + frame for frame in _LS_STREAM),
    "input-in-the-start-event": _stream(_sse_frame(_tool_start(1, "Bash", _LS)), _sse_frame(_block_stop(1))),
    "input-in-the-start-event-and-its-deltas": _stream(
        *(
            _sse_frame(event)
            for event in (_tool_start(1, "Bash", _LS), _input_delta(1, json.dumps(_LS)), _block_stop(1))
        )
    ),
    "byte-order-mark": (b"\xef\xbb\xbf" + _LS_STREAM[0], *_LS_STREAM[1:]),
    "input-in-the-start-event-and-an-empty-delta": _stream(
        *(_sse_frame(event) for event in (_tool_start(1, "Bash", _LS), _input_delta(1, ""), _block_stop(1)))
    ),
    "typescript-input-buffer-before-the-deltas": _stream(
        *(
            _sse_frame(event)
            for event in (
                _bash_start_behind_a_typescript_input_buffer(),
                _input_delta(1, json.dumps(_LS)),
                _block_stop(1),
            )
        )
    ),
    "typescript-input-buffer-beside-input-in-the-start-event": _stream(
        *(
            _sse_frame(event)
            for event in (_bash_start_behind_a_typescript_input_buffer(_LS), _input_delta(1, ""), _block_stop(1))
        )
    ),
}
_READ_A: Final = {"file_path": "/tmp/a.txt"}
_READ_AND_LS_IN_BLOCK_ORDER: Final = _stream(
    *(
        _sse_frame(event)
        for event in (
            _tool_start(1, "Read"),
            _input_delta(1, json.dumps(_READ_A)),
            _block_stop(1),
            _tool_start(2, "Bash"),
            _input_delta(2, json.dumps(_LS)),
            _block_stop(2),
        )
    )
)
_READ_AND_LS_OUT_OF_BLOCK_ORDER: Final = {
    "interleaved-input-deltas": _stream(
        *(
            _sse_frame(event)
            for event in (
                _tool_start(1, "Read"),
                _tool_start(2, "Bash"),
                _input_delta(1, _halves(_READ_A)[0]),
                _input_delta(2, _halves(_LS)[0]),
                _input_delta(1, _halves(_READ_A)[1]),
                _input_delta(2, _halves(_LS)[1]),
                _block_stop(1),
                _block_stop(2),
            )
        )
    ),
    "block-stops-at-the-end": _stream(
        *(
            _sse_frame(event)
            for event in (
                _tool_start(1, "Read"),
                _input_delta(1, json.dumps(_READ_A)),
                _tool_start(2, "Bash"),
                _input_delta(2, json.dumps(_LS)),
                _block_stop(1),
                _block_stop(2),
            )
        )
    ),
    "no-block-stops": _stream(
        *(
            _sse_frame(event)
            for event in (
                _tool_start(1, "Read"),
                _input_delta(1, json.dumps(_READ_A)),
                _tool_start(2, "Bash"),
                _input_delta(2, json.dumps(_LS)),
            )
        )
    ),
}
_UNSTOPPED_WEB_SEARCH_BEFORE_A_TEXT_BLOCK: Final = (
    _OPENING[0],
    *(
        _sse_frame(event)
        for event in (
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "server_tool_use", "id": "srvtoolu_01Search", "name": "web_search", "input": {}},
            },
            _input_delta(0, json.dumps({"query": "weather in Paris"})),
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Searching."}},
            _block_stop(1),
        )
    ),
    *_CLOSING,
)
_RM_RF_CALLS_ONLY_A_LENIENT_CLIENT_READS: Final = {
    "space-before-named-data-lines": _stream(*(frame.replace(b"\ndata:", b"\n data:") for frame in _RM_RF_CALL)),
    "space-before-unnamed-data-lines": _stream(*(b" " + frame.split(b"\n", 1)[1] for frame in _RM_RF_CALL)),
    "space-before-the-event-names-typing-the-data": _stream(
        *(b" " + re.sub(rb'"type": "content_block_\w+", ', b"", frame, count=1) for frame in _RM_RF_CALL)
    ),
    "data-lines-in-comments": _stream(*(b": " + frame.split(b"\n", 1)[1] for frame in _RM_RF_CALL)),
    "carriage-return-inside-unnamed-json": _stream(
        *(frame.split(b"\n", 1)[1].replace(b", ", b",\r", 1) for frame in _RM_RF_CALL)
    ),
}
_WRITE_SPELLED_ACROSS_THE_DATA_LINES_OF_ONE_FRAME: Final = _stream(
    *(
        b"data: " + json.dumps(event).encode() + f"\ndata: event: {event['type']}\n\n".encode()
        for event in (_tool_start(1, "Write", {"file_path": "/etc/passwd"}), _block_stop(1))
    )
)
_BASH_INPUT_FINISHED_AFTER_MESSAGE_STOP: Final = (
    *_OPENING,
    _sse_frame(_tool_start(1, "Bash")),
    _sse_frame(_input_delta(1, '{"command": "rm -rf /"')),
    *_CLOSING,
    _sse_frame(_input_delta(1, ', "command": "ls"}')),
    _sse_frame(_block_stop(1)),
)
_UNKNOWN_EVENT_FRAME: Final = b'event: content_block_metadata\ndata: {"type": "content_block_metadata"}\n\n'
_DONE_FRAME: Final = b"data: [DONE]\n\n"
_TOOL_ONLY_SHAPES_THE_CHECK_CANNOT_FORWARD: Final = {
    "byte-order-mark": (b"\xef\xbb\xbf" + _LS_ONLY_STREAM[0], *_LS_ONLY_STREAM[1:]),
    "typed-only-by-event-names": tuple(
        re.sub(rb'"type": "\w+", ', b"", frame, count=1) for frame in _LS_ONLY_STREAM
    ),
}


def _body(chunks: Sequence[object]) -> bytes:
    return b"".join(chunk for chunk in chunks if isinstance(chunk, bytes))


def _sdk_client(chunks: Sequence[object]) -> anthropic.Anthropic:
    body: Final = _body(chunks)
    transport: Final = httpx.MockTransport(
        lambda _request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    )
    return anthropic.Anthropic(api_key="test", base_url="http://test", http_client=httpx.Client(transport=transport))


def _sdk_message(chunks: Sequence[object]) -> Message:
    with _sdk_client(chunks).messages.stream(
        model="claude-sonnet-4-5", max_tokens=1, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        return stream.get_final_message()


def _tool_input_as_the_typescript_sdk_reads(start_input: object, json_buf: str, partial_jsons: Sequence[str]) -> object:
    # @anthropic-ai/sdk 1.12.0 internal/message-stream-utils.js: `jsonBuf ? partialParse(jsonBuf) : {}`, and
    # MessageStream.js copies the start's content_block whole, then reads `(snapshotContent.__json_buf || '')
    # + event.delta.partial_json` (2026-11-23)
    if not partial_jsons:
        return start_input
    streamed: Final = json_buf + "".join(partial_jsons)
    return json.loads(streamed) if streamed else {}


def _typescript_sdk_input_buffer(block: ToolUseBlock) -> str:
    json_buf: Final = (block.model_extra or {}).get("__json_buf")
    return json_buf if isinstance(json_buf, str) else ""


def _typescript_sdk_tool_inputs(chunks: Sequence[object]) -> list[object]:
    events: Final = list(
        _sdk_client(chunks).messages.create(
            model="claude-sonnet-4-5", max_tokens=1, messages=[{"role": "user", "content": "hi"}], stream=True
        )
    )
    tool_starts: Final = [
        (event.index, event.content_block)
        for event in events
        if event.type == "content_block_start" and event.content_block.type == "tool_use"
    ]
    return [
        _tool_input_as_the_typescript_sdk_reads(
            block.input,
            _typescript_sdk_input_buffer(block),
            [
                event.delta.partial_json
                for event in events
                if event.type == "content_block_delta" and event.index == index and event.delta.type == "input_json_delta"
            ],
        )
        for index, block in tool_starts
    ]


def _data_line_event(line: str) -> dict[str, object] | None:
    if not line.startswith("data:"):
        return None
    try:
        event: Final = json.loads(line.removeprefix("data:"))
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _is_tool_use_start(event: dict[str, object]) -> bool:
    block: Final = event.get("content_block")
    return event.get("type") == "content_block_start" and isinstance(block, dict) and block.get("type") == "tool_use"


def _partial_jsons(events: Sequence[dict[str, object]], index: object) -> list[str]:
    deltas: Final = [
        event.get("delta")
        for event in events
        if event.get("type") == "content_block_delta" and event.get("index") == index
    ]
    return [
        partial
        for delta in deltas
        if isinstance(delta, dict) and isinstance(partial := delta.get("partial_json"), str)
    ]


def _tool_inputs_of(events: Sequence[dict[str, object]]) -> list[object]:
    tool_indexes: Final = [event.get("index") for event in events if _is_tool_use_start(event)]
    return [json.loads("".join(_partial_jsons(events, index)) or "{}") for index in tool_indexes]


def _tool_inputs_read_line_by_line(chunks: Sequence[object]) -> list[object]:
    lines: Final = httpx.Response(200, content=_body(chunks)).iter_lines()
    return _tool_inputs_of([event for line in lines if (event := _data_line_event(line)) is not None])


def _event_read_leniently(name: str, line: str) -> dict[str, object] | None:
    _, data_field, payload = line.partition("data:")
    event: Final = _data_line_event(data_field + payload)
    return {"type": name, **event} if event is not None and name else event


def _events_of_a_frame_read_leniently(frame: str) -> list[dict[str, object]]:
    lines: Final = [line.strip() for line in frame.split("\n")]
    names: Final = [line.removeprefix("event:").strip() for line in lines if line.startswith("event:")]
    name: Final = names[-1] if names else ""
    return [event for line in lines if (event := _event_read_leniently(name, line)) is not None]


def _tool_inputs_read_leniently(chunks: Sequence[object]) -> list[object]:
    frames: Final = _body(chunks).decode().split("\n\n")
    return _tool_inputs_of(list(chain.from_iterable(map(_events_of_a_frame_read_leniently, frames))))


def _tool_inputs_at_message_stop(chunks: Sequence[object]) -> list[object]:
    with _sdk_client(chunks).messages.stream(
        model="claude-sonnet-4-5", max_tokens=1, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        snapshots: Final = [event.message.model_copy(deep=True) for event in stream if event.type == "message_stop"]
    return [block.input for block in snapshots[0].content if block.type == "tool_use"]


class TestToolPermissionGuardrailAnthropicStreamShapes:
    """What an Anthropic SDK client reads from the stream the guardrail delivers, whatever SSE shape it has."""

    def _bash_ls_only(self, on_disallowed_action: Literal["block", "rewrite"]) -> ToolPermissionGuardrail:
        return ToolPermissionGuardrail(
            guardrail_name=f"anthropic-bash-ls-only-{on_disallowed_action}",
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
            on_disallowed_action=on_disallowed_action,
        )

    async def _drain(self, guardrail: ToolPermissionGuardrail, chunks: tuple[bytes, ...]) -> list[object]:
        async def _stream() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        return [
            chunk
            async for chunk in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(), response=_stream(), request_data={}
            )
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("chunks", _DENIED_CALL_SHAPES.values(), ids=_DENIED_CALL_SHAPES)
    async def test_client_never_reads_a_denied_call_whatever_the_stream_shape(self, chunks: tuple[bytes, ...]):
        out: Final = await self._drain(self._bash_ls_only("rewrite"), chunks)

        tool_inputs: Final = [block.input for block in _sdk_message(out).content if block.type == "tool_use"]
        assert _RM_RF not in tool_inputs

    @pytest.mark.asyncio
    async def test_allowed_calls_that_overlap_are_forwarded_as_the_upstream_sent_them(self):
        chunks: Final = _overlapped_calls(_LS, _LS)

        assert await self._drain(self._bash_ls_only("block"), chunks) == list(chunks)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("first", "second", "kept_id"), [(_LS, _RM_RF, "tu_1"), (_RM_RF, _LS, "tu_2")])
    async def test_of_two_calls_that_overlap_only_the_allowed_one_reaches_the_client_with_its_own_input(
        self, first: dict[str, str], second: dict[str, str], kept_id: str
    ):
        out: Final = await self._drain(self._bash_ls_only("rewrite"), _overlapped_calls(first, second))

        calls: Final = [block for block in _sdk_message(out).content if block.type == "tool_use"]
        assert [(call.id, call.input) for call in calls] == [(kept_id, _LS)]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("chunks", _ALLOWED_CALL_SHAPES.values(), ids=_ALLOWED_CALL_SHAPES)
    async def test_allowed_call_reaches_the_client_as_the_plain_stream_would(self, chunks: tuple[bytes, ...]):
        out: Final = await self._drain(self._bash_ls_only("block"), chunks)

        assert _sdk_message(out) == _sdk_message(list(_LS_STREAM))
        assert _typescript_sdk_tool_inputs(out) == [_LS]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks", _TOOL_ONLY_SHAPES_THE_CHECK_CANNOT_FORWARD.values(), ids=_TOOL_ONLY_SHAPES_THE_CHECK_CANNOT_FORWARD
    )
    async def test_allowed_call_the_check_cannot_forward_keeps_every_field_the_upstream_sent(
        self, chunks: tuple[bytes, ...]
    ):
        out: Final = await self._drain(self._bash_ls_only("block"), chunks)

        assert _sdk_message(out) == _sdk_message(list(_LS_ONLY_STREAM))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks", _READ_AND_LS_OUT_OF_BLOCK_ORDER.values(), ids=_READ_AND_LS_OUT_OF_BLOCK_ORDER
    )
    async def test_allowed_calls_out_of_block_order_reach_the_client_as_in_block_order(
        self, chunks: tuple[bytes, ...]
    ):
        out: Final = await self._drain(self._bash_ls_only("block"), chunks)

        assert _sdk_message(out) == _sdk_message(list(_READ_AND_LS_IN_BLOCK_ORDER))
        assert _typescript_sdk_tool_inputs(out) == [_READ_A, _LS]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks",
        (*_DENIED_CALL_SHAPES.values(), *_DENIED_CALLS_ONLY_A_LINE_SPLITTING_CLIENT_READS.values()),
        ids=(*_DENIED_CALL_SHAPES, *_DENIED_CALLS_ONLY_A_LINE_SPLITTING_CLIENT_READS),
    )
    async def test_client_splitting_lines_at_every_unicode_break_never_reads_a_denied_call(
        self, chunks: tuple[bytes, ...]
    ):
        out: Final = await self._drain(self._bash_ls_only("rewrite"), chunks)

        assert _RM_RF not in _tool_inputs_read_line_by_line(out)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks", _SHAPES_WHOSE_CALL_THE_CHECK_DENIES.values(), ids=_SHAPES_WHOSE_CALL_THE_CHECK_DENIES
    )
    async def test_denied_reply_keeps_the_upstream_message_id_only_when_the_plain_stream_would(
        self, chunks: tuple[bytes, ...]
    ):
        guardrail: Final = self._bash_ls_only("rewrite")
        upstream_id: Final = _sdk_message(list(_LS_STREAM)).id

        plain_id: Final = _sdk_message(await self._drain(guardrail, _stream(*_RM_RF_CALL))).id
        shaped_id: Final = _sdk_message(await self._drain(guardrail, chunks)).id

        assert (shaped_id == upstream_id) == (plain_id == upstream_id)

    @pytest.mark.asyncio
    async def test_check_denies_a_hosted_call_whose_block_the_next_block_starts_without_a_stop(self):
        with pytest.raises(GuardrailRaisedException):
            await self._drain(self._bash_ls_only("block"), _UNSTOPPED_WEB_SEARCH_BEFORE_A_TEXT_BLOCK)

    @pytest.mark.asyncio
    async def test_check_judges_the_input_field_the_sdk_reads_when_a_delta_carries_another(self):
        with pytest.raises(GuardrailRaisedException):
            await self._drain(
                self._bash_ls_only("block"), _DENIED_CALL_SHAPES["input-hidden-behind-a-text-field"]
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks", _RM_RF_CALLS_ONLY_A_LENIENT_CLIENT_READS.values(), ids=_RM_RF_CALLS_ONLY_A_LENIENT_CLIENT_READS
    )
    async def test_client_reading_any_line_that_holds_data_never_reads_a_call_the_check_could_not_see(
        self, chunks: tuple[bytes, ...]
    ):
        out: Final = await self._drain(self._bash_ls_only("rewrite"), chunks)

        assert _RM_RF not in _tool_inputs_read_leniently(out)

    @pytest.mark.asyncio
    async def test_client_never_reads_a_call_the_upstream_spelled_only_across_one_frames_data_lines(self):
        out: Final = await self._drain(self._bash_ls_only("block"), _WRITE_SPELLED_ACROSS_THE_DATA_LINES_OF_ONE_FRAME)

        assert _sdk_message(out) == _sdk_message(list(_stream()))

    @pytest.mark.asyncio
    async def test_client_reading_the_message_at_its_stop_reads_the_input_the_check_judged(self):
        out: Final = await self._drain(self._bash_ls_only("rewrite"), _BASH_INPUT_FINISHED_AFTER_MESSAGE_STOP)

        assert _RM_RF not in _tool_inputs_at_message_stop(out)

    @pytest.mark.asyncio
    async def test_allowed_stream_the_check_reframes_keeps_the_frames_no_client_applies_where_they_were(self):
        reframed_calls: Final = _READ_AND_LS_OUT_OF_BLOCK_ORDER["block-stops-at-the-end"]
        chunks: Final = (*reframed_calls[:1], _UNKNOWN_EVENT_FRAME, *reframed_calls[1:], _DONE_FRAME)

        out: Final = await self._drain(self._bash_ls_only("block"), chunks)

        assert _body(out).startswith(_body(reframed_calls[:1]) + _UNKNOWN_EVENT_FRAME)
        assert _body(out).endswith(_CLOSING[-1] + _DONE_FRAME)
        assert _sdk_message(out) == _sdk_message(list(_READ_AND_LS_IN_BLOCK_ORDER))


_GENERATE_CONTENT_STREAM_ROUTE: Final = "/v1beta/models/gemini-3.8-flash:streamGenerateContent"


def _generate_content_frame(*candidate_parts: list[dict[str, object]], first_index: int = 0) -> bytes:
    candidates: Final = [
        {"content": {"role": "model", "parts": parts}, "index": index}
        for index, parts in enumerate(candidate_parts, start=first_index)
    ]
    return f"data: {json.dumps({'candidates': candidates})}\r\n\r\n".encode()


_THOUGHT_SIGNATURE: Final = {"thoughtSignature": "c2lnbmF0dXJl"}


def _gemini_function_call(name: str) -> dict[str, object]:
    return {"functionCall": {"name": name, "args": {"path": "/tmp/report.txt"}}}


def _signed(part: dict[str, object]) -> dict[str, object]:
    return {**part, **_THOUGHT_SIGNATURE}


def _generate_content_request() -> dict[str, object]:
    return {
        "model": "gemini-3.8-flash",
        "contents": [{"role": "user", "parts": [{"text": "Check the weather, then delete the report"}]}],
        "tools": [{"functionDeclarations": [{"name": "get_weather"}, {"name": "delete_file"}]}],
    }


def _joined_text(chunks: Sequence[object]) -> str:
    return "".join(chunk.decode() if isinstance(chunk, bytes) else str(chunk) for chunk in chunks)


def _event_parts(event: str) -> tuple[dict[str, object], ...]:
    candidates: Final = json.loads(event.removeprefix("data: "))["candidates"]
    return tuple(chain.from_iterable(candidate["content"]["parts"] for candidate in candidates))


def _stream_parts(chunks: Sequence[object]) -> tuple[dict[str, object], ...]:
    events: Final = (event for event in _joined_text(chunks).split("\r\n\r\n") if event)
    return tuple(chain.from_iterable(map(_event_parts, events)))


_DENIED_IN_A_LATER_FRAME: Final = [
    _generate_content_frame([{"text": "Deleting it"}]),
    _generate_content_frame([_signed(_gemini_function_call("delete_file"))]),
]
_DENIED_FRAME: Final = _generate_content_frame(
    [_signed(_gemini_function_call("get_weather")), _gemini_function_call("delete_file")]
)
_DENIED_FIRST_CALL: Final = _signed(_gemini_function_call("delete_file"))
_COMPACT_SIGNED_ALLOWED_FRAME: Final = (
    b'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather"},'
    b'"thoughtSignature":"c2lnbmF0dXJl"}]}}]}\r\n\r\n'
)
_COMPACT_ALLOWED_FRAME: Final = b'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather"}}]}}]}\r\n\r\n'


class TestToolPermissionGuardrailGenerateContentStream:
    """The Google ``:streamGenerateContent`` route hands the hook raw Gemini SSE frames, which the guardrail
    refused as an unparseable Anthropic stream, so every stream on that route answered 400."""

    def setup_method(self) -> None:
        rules: Final = [
            {"id": "allow_weather", "tool_name": r"^get_weather$", "decision": "allow"},
            {"id": "deny_delete", "tool_name": r"^delete_file$", "decision": "deny"},
        ]
        self.blocking = ToolPermissionGuardrail(
            guardrail_name="gemini-block", rules=rules, default_action="deny", on_disallowed_action="block"
        )
        self.rewriting = ToolPermissionGuardrail(
            guardrail_name="gemini-rewrite", rules=rules, default_action="deny", on_disallowed_action="rewrite"
        )

    async def _drain(self, guardrail: ToolPermissionGuardrail, chunks: Sequence[object]) -> list[object]:
        async def _stream() -> AsyncIterator[object]:
            for chunk in chunks:
                yield chunk

        return [
            c
            async for c in guardrail.async_post_call_streaming_iterator_hook(
                user_api_key_dict=UserAPIKeyAuth(request_route=_GENERATE_CONTENT_STREAM_ROUTE),
                response=_stream(),
                request_data=_generate_content_request(),
            )
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["block", "rewrite"])
    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param(
                [_generate_content_frame([{"text": "Sunny"}]), _generate_content_frame([{"text": " and warm"}])],
                id="text",
            ),
            pytest.param(
                [
                    _generate_content_frame([{"text": "Checking"}]),
                    _generate_content_frame([_signed(_gemini_function_call("get_weather"))]),
                ],
                id="allowed-call",
            ),
            pytest.param(
                [_generate_content_frame([_gemini_function_call("get_weather")])[:37],
                 _generate_content_frame([_gemini_function_call("get_weather")])[37:]],
                id="allowed-call-split-mid-line",
            ),
            pytest.param(
                [
                    b": keepalive\r\n\r\n",
                    b'data: {"candidates": [{"finishReason": "STOP", "index": 0}]}\r\n\r\n',
                    b'data: {"usageMetadata": {"totalTokenCount": 3}}\r\n\r\n',
                    b"data: [DONE]\r\n\r\n",
                ],
                id="frames-without-parts",
            ),
            pytest.param(
                [b'data: {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}\r\n\r\n'],
                id="upstream-error",
            ),
        ],
    )
    async def test_stream_without_a_denied_call_is_delivered_verbatim(self, chunks: list[bytes], mode: str) -> None:
        guardrail: Final = self.blocking if mode == "block" else self.rewriting

        delivered: Final = await self._drain(guardrail, chunks)

        assert len(delivered) == len(chunks)
        assert all(out is sent for out, sent in zip(delivered, chunks))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param(_DENIED_IN_A_LATER_FRAME, id="later-frame"),
            pytest.param([_DENIED_FRAME], id="second-part"),
            pytest.param(
                [_generate_content_frame([{"text": "Done"}], [_gemini_function_call("delete_file")])],
                id="second-candidate",
            ),
            pytest.param(
                [b'data: {"candidates": [{"content": {"parts": [{"function_call": {"name": "delete_file"}}]}}]}\r\r'],
                id="snake-case-cr-line-breaks",
            ),
            pytest.param([_DENIED_FRAME[:50], _DENIED_FRAME[50:]], id="split-mid-line"),
        ],
    )
    async def test_denied_call_blocks_the_stream(self, chunks: list[bytes]) -> None:
        with pytest.raises(GuardrailRaisedException, match="deny_delete"):
            await self._drain(self.blocking, chunks)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("as_text", [False, True], ids=["bytes", "str"])
    async def test_rewrite_replaces_only_the_denied_call(self, as_text: bool) -> None:
        sent: Final = [
            _generate_content_frame([{"text": "Checking"}]),
            _DENIED_FRAME,
            _generate_content_frame([{"text": "Done"}]),
        ]
        chunks: Final = [chunk.decode() for chunk in sent] if as_text else sent

        delivered: Final = await self._drain(self.rewriting, chunks)

        assert all(isinstance(chunk, str if as_text else bytes) for chunk in delivered)
        first, rewritten, last, after_last = _joined_text(delivered).split("\r\n\r\n")
        assert (first, last, after_last) == (sent[0].decode().removesuffix("\r\n\r\n"), sent[2].decode().removesuffix("\r\n\r\n"), "")
        allowed_call, denial = _event_parts(rewritten)
        assert allowed_call == _signed(_gemini_function_call("get_weather"))
        assert list(denial) == ["text"]
        assert "deny_delete" in denial["text"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param([_generate_content_frame([_DENIED_FIRST_CALL, _gemini_function_call("get_weather")])], id="same-frame"),
            pytest.param(
                [
                    _generate_content_frame([_DENIED_FIRST_CALL]),
                    _generate_content_frame([_gemini_function_call("get_weather")]),
                ],
                id="later-frame",
            ),
        ],
    )
    async def test_rewrite_moves_a_denied_first_call_signature_to_the_first_kept_call(
        self, chunks: list[bytes]
    ) -> None:
        delivered: Final = await self._drain(self.rewriting, chunks)

        denial, allowed_call = _stream_parts(delivered)
        assert list(denial) == ["text"]
        assert "deny_delete" in denial["text"]
        assert allowed_call == _signed(_gemini_function_call("get_weather"))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param(
                [_generate_content_frame([_DENIED_FIRST_CALL, _gemini_function_call("delete_file")])],
                id="every-call-denied",
            ),
            pytest.param(
                [_generate_content_frame([_DENIED_FIRST_CALL], [_gemini_function_call("get_weather")])],
                id="kept-call-in-another-candidate",
            ),
            pytest.param(
                [
                    _generate_content_frame([_DENIED_FIRST_CALL]),
                    _generate_content_frame([_gemini_function_call("get_weather")], first_index=1),
                ],
                id="kept-call-of-another-candidate-in-a-later-frame",
            ),
        ],
    )
    async def test_rewrite_keeps_the_signature_on_the_denial_when_its_turn_has_no_kept_call(
        self, chunks: list[bytes]
    ) -> None:
        delivered: Final = await self._drain(self.rewriting, chunks)

        signed_denial, other_part = _stream_parts(delivered)
        assert signed_denial == {"text": signed_denial["text"], **_THOUGHT_SIGNATURE}
        assert "deny_delete" in signed_denial["text"]
        assert _THOUGHT_SIGNATURE.keys().isdisjoint(other_part)

    @pytest.mark.asyncio
    async def test_rewrite_leaves_a_signed_kept_call_its_own_signature(self) -> None:
        kept_call: Final = {**_gemini_function_call("get_weather"), "thoughtSignature": "a2VwdA=="}

        delivered: Final = await self._drain(self.rewriting, [_generate_content_frame([_DENIED_FIRST_CALL, kept_call])])

        signed_denial, allowed_call = _stream_parts(delivered)
        assert signed_denial == {"text": signed_denial["text"], **_THOUGHT_SIGNATURE}
        assert "deny_delete" in signed_denial["text"]
        assert allowed_call == kept_call

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunks, untouched",
        [
            pytest.param(
                [_COMPACT_SIGNED_ALLOWED_FRAME, _generate_content_frame([_gemini_function_call("delete_file")])],
                0,
                id="signed-kept-call-first",
            ),
            pytest.param(
                [_generate_content_frame([_gemini_function_call("delete_file")]), _COMPACT_ALLOWED_FRAME],
                1,
                id="unsigned-denied-call-first",
            ),
        ],
    )
    async def test_rewrite_leaves_a_frame_without_a_changed_part_byte_for_byte(
        self, chunks: list[bytes], untouched: int
    ) -> None:
        delivered: Final = await self._drain(self.rewriting, chunks)

        frames: Final = _joined_text(delivered).split("\r\n\r\n")
        assert frames[untouched] == chunks[untouched].decode().removesuffix("\r\n\r\n")
        assert "deny_delete" in frames[1 - untouched]

    @pytest.mark.asyncio
    async def test_rewrite_keeps_the_other_candidates_of_the_denied_frame(self) -> None:
        finished: Final = {"finishReason": "STOP", "index": 0}
        denied: Final = {"content": {"role": "model", "parts": [_gemini_function_call("delete_file")]}, "index": 1}
        frame: Final = f"data: {json.dumps({'candidates': [finished, denied]})}\r\n\r\n".encode()

        delivered: Final = await self._drain(self.rewriting, [frame])

        kept, rewritten = json.loads(_joined_text(delivered).removeprefix("data: "))["candidates"]
        assert kept == finished
        (denial,) = rewritten["content"]["parts"]
        assert list(denial) == ["text"]
        assert "deny_delete" in denial["text"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["block", "rewrite"])
    @pytest.mark.parametrize(
        "chunks",
        [
            pytest.param([b"data: {not json\r\n\r\n"], id="non-json-data"),
            pytest.param(
                [b'{"candidates": [{"content": {"parts": [{"functionCall": {"name": "delete_file"}}]}}]}\r\n'],
                id="raw-json-line",
            ),
            pytest.param(
                [_generate_content_frame([{"functionCall": {"name": "delete_file", "partialArgs": [], "willContinue": True}}])],
                id="unknown-call-field",
            ),
            pytest.param([_generate_content_frame([{"functionCall": {"args": {}}}])], id="nameless-call"),
            pytest.param([_generate_content_frame([{"functionCall": {"name": "get_weather", "args": "rm"}}])], id="non-object-args"),
            pytest.param(
                [_generate_content_frame([{"functionCall": {"name": "get_weather"}, "function_call": {"name": "delete_file"}}])],
                id="both-call-keys",
            ),
            pytest.param([b'data: {"candidates": {"content": {}}}\r\n\r\n'], id="candidates-not-a-list"),
            pytest.param([b'data: {"candidates": [{"content": "delete_file"}]}\r\n\r\n'], id="content-not-an-object"),
            pytest.param([b'data: {"candidates": [{"content": {"parts": ["delete_file"]}}]}\r\n\r\n'], id="part-not-an-object"),
            pytest.param(
                [b'data: {"candidates": [{"content": {"parts": {"functionCall": {"name": "delete_file"}}}}]}\r\n\r\n'],
                id="parts-not-a-list",
            ),
            pytest.param(
                [b'data: {"candidates": [{"index": "1", "content": {"parts": [{"functionCall": {"name": "get_weather"}}]}}]}\r\n\r\n'],
                id="non-integer-candidate-index",
            ),
            pytest.param(
                [f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': 'a data: {}'}]}}]}, ensure_ascii=False)}\r\n\r\n".encode()],
                id="line-separator-starts-a-data-line",
            ),
            pytest.param(
                [b'data: {"candidates": [{"content": {"parts": [{"text": "caf\xe9"}]}}]}\r\n\r\n'], id="invalid-utf8"
            ),
            pytest.param(
                [
                    _generate_content_frame([{"text": "Deleting the report"}]),
                    {"candidates": [{"content": {"parts": [_gemini_function_call("delete_file")]}}]},
                ],
                id="non-text-chunk",
            ),
        ],
    )
    async def test_stream_no_client_reads_safely_fails_closed(self, chunks: list[object], mode: str) -> None:
        guardrail: Final = self.blocking if mode == "block" else self.rewriting

        with pytest.raises(GuardrailRaisedException, match="could not be verified"):
            await self._drain(guardrail, chunks)


_VAULT_NAMESPACE: Final = "mcp__vault"
_VAULT_TOOL: Final = "read_the_current_secret_value_from_the_vault_that_this_codex_session_was_given"
_VAULT_FULL_NAME: Final = f"{_VAULT_NAMESPACE}__{_VAULT_TOOL}"
_VAULT_SENT_NAME: Final = "mcp__vault__read_the_current_secret_value_from_the_vaul_f96e7908"
_DENY_VAULT_ALLOW_MCP: Final = [
    {"id": "deny_vault_read", "tool_name": f"^{_VAULT_FULL_NAME}$", "decision": "deny"},
    {"id": "allow_mcp", "tool_name": r"^mcp__.*$", "decision": "allow"},
]


def _bridged_vault_request(event_hook: GuardrailEventHooks) -> dict[str, object]:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.litellm_core_utils.tool_fields import NamespaceMember

    logging_obj: Final = Logging(
        model="anthropic/claude-sonnet-5-5",
        messages=[],
        stream=False,
        call_type="acompletion",
        start_time=1.0,
        litellm_call_id="bridged-vault",
        function_id="test",
    )
    logging_obj.record_bridged_tool_names({_VAULT_SENT_NAME: NamespaceMember(_VAULT_NAMESPACE, _VAULT_TOOL)})
    return {
        "guardrails": [f"vault-{event_hook.value}"],
        "tools": [{"type": "function", "function": {"name": _VAULT_SENT_NAME, "parameters": {"type": "object"}}}],
        "litellm_logging_obj": logging_obj,
    }


class TestToolPermissionOnBridgedNamespaceTools:
    """The Responses bridge sends a long namespace tool under a shortened name. Rules are written against the
    name the caller sees, so the guardrail has to judge the shortened name as the caller's tool."""

    @staticmethod
    def _guardrail(event_hook: GuardrailEventHooks, action: Literal["block", "rewrite"]) -> ToolPermissionGuardrail:
        return ToolPermissionGuardrail(
            guardrail_name=f"vault-{event_hook.value}",
            rules=_DENY_VAULT_ALLOW_MCP,
            default_action="deny",
            on_disallowed_action=action,
            event_hook=event_hook,
        )

    @pytest.mark.asyncio
    async def test_a_returned_call_to_a_shortened_name_is_denied_by_the_rule_on_its_full_name(self) -> None:
        guardrail: Final = self._guardrail(GuardrailEventHooks.post_call, "rewrite")
        response: Final = ModelResponse(
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    message={
                        "tool_calls": [
                            {
                                "id": "call_vault",
                                "type": "function",
                                "function": Function(name=_VAULT_SENT_NAME, arguments="{}"),
                            }
                        ]
                    },
                )
            ]
        )

        await guardrail.async_post_call_success_hook(
            data=_bridged_vault_request(GuardrailEventHooks.post_call),
            user_api_key_dict=UserAPIKeyAuth(),
            response=response,
        )

        message: Final = response.choices[0].message
        assert [call.function.name for call in message.tool_calls or ()] == []
        assert "deny_vault_read" in (message.content or "")

    @pytest.mark.asyncio
    async def test_a_sent_tool_under_a_shortened_name_is_denied_by_the_rule_on_its_full_name(self) -> None:
        guardrail: Final = self._guardrail(GuardrailEventHooks.pre_call, "block")

        with pytest.raises(HTTPException) as raised:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=_bridged_vault_request(GuardrailEventHooks.pre_call),
                call_type="acompletion",
            )

        assert raised.value.status_code == 400
        assert "deny_vault_read" in json.dumps(raised.value.detail)

    @pytest.mark.asyncio
    async def test_a_name_map_sent_by_the_client_does_not_rename_a_denied_tool(self) -> None:
        guardrail: Final = self._guardrail(GuardrailEventHooks.pre_call, "block")
        forged: Final = {
            "guardrails": ["vault-pre_call"],
            "tools": [{"type": "function", "function": {"name": "drop_tables", "parameters": {"type": "object"}}}],
            "litellm_logging_obj": {
                "model_call_details": {"litellm_bridged_tool_names": {"drop_tables": "mcp__safe__read"}}
            },
            "litellm_bridged_tool_names": {"drop_tables": "mcp__safe__read"},
        }

        with pytest.raises(HTTPException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(), cache=DualCache(), data=forged, call_type="acompletion"
            )


class _CapturedBridgedCall(Exception):
    def __init__(self, kwargs: dict[str, object]) -> None:
        super().__init__("bridged call captured")
        self.kwargs = kwargs


def _bridged_completion_kwargs(
    request_input: list[dict[str, object]], tools: list[dict[str, object]]
) -> dict[str, object]:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.responses.litellm_completion_transformation.handler import LiteLLMCompletionTransformationHandler

    def capture(**kwargs: object) -> None:
        raise _CapturedBridgedCall(dict(kwargs))

    logging_obj: Final = Logging(
        model="anthropic/claude-sonnet-5-5",
        messages=[],
        stream=False,
        call_type="responses",
        start_time=1.0,
        litellm_call_id="bridged-identity",
        function_id="test",
    )
    with (
        patch("litellm.completion", capture),  # test-quality-ok: no DI seam; the bridge calls litellm.completion
        pytest.raises(_CapturedBridgedCall) as captured,
    ):
        LiteLLMCompletionTransformationHandler().response_api_handler(
            model="anthropic/claude-sonnet-5-5",
            input=request_input,
            responses_api_request={"tools": tools},
            custom_llm_provider="anthropic",
            _is_async=False,
            litellm_logging_obj=logging_obj,
        )
    return captured.value.kwargs


def _history_call(namespace: str, name: str) -> list[dict[str, object]]:
    return [
        {"type": "message", "role": "user", "content": "go on"},
        {"type": "function_call", "call_id": "call_earlier", "namespace": namespace, "name": name, "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_earlier", "output": "done"},
    ]


def _returns_a_call_to(name: str) -> ModelResponse:
    return ModelResponse(
        choices=[
            Choices(
                finish_reason="tool_calls",
                message={
                    "tool_calls": [
                        {"id": "call_now", "type": "function", "function": Function(name=name, arguments="{}")}
                    ]
                },
            )
        ]
    )


_JUDGED_AS_RETURNED: Final = {
    "a namespace tool known only from history": (
        _history_call("exec", "run"),
        [],
        "exec__run",
        "^exec$",
        {"type": "function_call", "namespace": "exec", "name": "run"},
    ),
    "a custom tool named like a shortened tool in history": (
        _history_call(_VAULT_NAMESPACE, _VAULT_TOOL),
        [{"type": "custom", "name": _VAULT_SENT_NAME}],
        _VAULT_SENT_NAME,
        f"^{_VAULT_SENT_NAME}$",
        {"type": "custom_tool_call", "name": _VAULT_SENT_NAME},
    ),
}


@pytest.mark.parametrize(
    ("request_input", "tools", "returned_name", "denied_pattern", "caller_sees"),
    _JUDGED_AS_RETURNED.values(),
    ids=_JUDGED_AS_RETURNED.keys(),
)
@pytest.mark.asyncio
async def test_a_bridged_call_is_judged_as_the_tool_the_caller_gets_back(
    request_input: list[dict[str, object]],
    tools: list[dict[str, object]],
    returned_name: str,
    denied_pattern: str,
    caller_sees: dict[str, str],
) -> None:
    from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig

    completion_kwargs: Final = _bridged_completion_kwargs(request_input, tools)
    returned_items: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
        _returns_a_call_to(returned_name), {"tools": tools}, request_input
    )
    assert [{key: getattr(item, key, None) for key in caller_sees} for item in returned_items] == [caller_sees]

    guardrail: Final = ToolPermissionGuardrail(
        guardrail_name="bridged-identity",
        rules=[
            {"id": "deny_what_the_caller_gets", "tool_name": denied_pattern, "decision": "deny"},
            {"id": "allow_everything_else", "tool_name": "^.*$", "decision": "allow"},
        ],
        default_action="deny",
        on_disallowed_action="rewrite",
        event_hook=GuardrailEventHooks.post_call,
    )
    response: Final = _returns_a_call_to(returned_name)

    await guardrail.async_post_call_success_hook(
        data={**completion_kwargs, "guardrails": ["bridged-identity"]},
        user_api_key_dict=UserAPIKeyAuth(),
        response=response,
    )

    message: Final = response.choices[0].message
    assert message.tool_calls in (None, [])
    assert "deny_what_the_caller_gets" in (message.content or "")


@pytest.mark.asyncio
async def test_a_deferred_top_level_tool_is_judged_by_its_own_name_not_as_a_namespace_member() -> None:
    from litellm.responses.litellm_completion_transformation.transformation import LiteLLMCompletionResponsesConfig

    tools: Final = [
        {"type": "function", "name": "read", "defer_loading": True, "parameters": {"type": "object"}},
    ]
    request_input: Final = [{"type": "message", "role": "user", "content": "read it"}]
    completion_kwargs: Final = _bridged_completion_kwargs(request_input, tools)
    returned_items: Final = LiteLLMCompletionResponsesConfig.transform_chat_completion_tools_to_responses_tools(
        _returns_a_call_to("read"), {"tools": tools}, request_input
    )
    assert [(item.name, getattr(item, "namespace", None)) for item in returned_items] == [("read", "read")]

    guardrail: Final = ToolPermissionGuardrail(
        guardrail_name="bridged-identity",
        rules=[{"id": "allow_read_namespace", "tool_name": "^read__.*$", "decision": "allow"}],
        default_action="deny",
        on_disallowed_action="rewrite",
        event_hook=GuardrailEventHooks.post_call,
    )
    response: Final = _returns_a_call_to("read")

    await guardrail.async_post_call_success_hook(
        data={**completion_kwargs, "guardrails": ["bridged-identity"]},
        user_api_key_dict=UserAPIKeyAuth(),
        response=response,
    )

    message: Final = response.choices[0].message
    assert message.tool_calls in (None, [])
    assert "Tool 'read'" in (message.content or "")
