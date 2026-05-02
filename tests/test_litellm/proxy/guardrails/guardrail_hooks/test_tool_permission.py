"""
Unit tests for Tool Permission Guardrail (OpenAI tool_calls semantics)
"""

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
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.tool_permission import (
    PermissionError,
)
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    ModelResponse,
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
    async def test_async_post_call_success_hook_missing_arguments_default_allows(self):
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


class TestToolPermissionGuardrailMCPPreCall:
    """Tests for the pre_mcp_call fast-path in async_pre_call_hook."""

    def _make_guardrail(
        self,
        *,
        decision: str,
        pattern: str,
        default_action: str = "allow",
        default_on: bool = True,
    ) -> ToolPermissionGuardrail:
        return ToolPermissionGuardrail(
            guardrail_name="test-mcp-guardrail",
            event_hook=["pre_mcp_call", "during_mcp_call"],
            rules=[{"id": "rule_1", "tool_name": pattern, "decision": decision}],
            default_action=default_action,
            on_disallowed_action="block",
            default_on=default_on,
        )

    @pytest.mark.asyncio
    async def test_mcp_tool_name_denied_raises_http_exception(self):
        """pre_mcp_call: denied tool raises HTTPException(400)."""
        guardrail = self._make_guardrail(decision="deny", pattern=r"exa-.*")
        data = {"mcp_tool_name": "exa-web_search_exa", "mcp_arguments": {"query": "test"}}

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="call_mcp_tool",
            )
        assert exc_info.value.status_code == 400
        assert "Violated guardrail policy" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_mcp_tool_name_allowed_passes(self):
        """pre_mcp_call: allowed tool returns data without raising."""
        guardrail = self._make_guardrail(decision="allow", pattern=r"exa-.*")
        data = {"mcp_tool_name": "exa-web_search_exa", "mcp_arguments": {"query": "test"}}

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="call_mcp_tool",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_mcp_namespaced_tool_name_matches_prefix_rule(self):
        """Rule written as 'exa-.*' must match the namespaced form produced by
        _convert_mcp_to_llm_format, i.e. '{server}-{tool}' = 'exa-web_search_exa'."""
        guardrail = self._make_guardrail(decision="deny", pattern=r"exa-.*")
        data = {"mcp_tool_name": "exa-web_search_exa"}

        with pytest.raises(HTTPException):
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="call_mcp_tool",
            )

    @pytest.mark.asyncio
    async def test_mcp_tool_name_not_matching_rule_uses_default_allow(self):
        """When no rule matches, default_action='allow' should pass."""
        guardrail = self._make_guardrail(
            decision="deny", pattern=r"exa-.*", default_action="allow"
        )
        data = {"mcp_tool_name": "github-search_repos"}

        result = await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="call_mcp_tool",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_mcp_tool_name_not_matching_rule_uses_default_deny(self):
        """When no rule matches, default_action='deny' should block."""
        guardrail = self._make_guardrail(
            decision="allow", pattern=r"exa-.*", default_action="deny"
        )
        data = {"mcp_tool_name": "github-search_repos"}

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="call_mcp_tool",
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_no_mcp_tool_name_falls_through_to_tools_list(self):
        """When mcp_tool_name is absent, hook falls through to the tools-list path."""
        guardrail = self._make_guardrail(decision="deny", pattern=r"exa-.*", default_on=False)
        data = {
            "tools": [{"type": "function", "function": {"name": "Bash"}}],
            "guardrails": ["test-mcp-guardrail"],
        }

        with patch.object(guardrail, "should_run_guardrail", return_value=True):
            result = await guardrail.async_pre_call_hook(
                user_api_key_dict=UserAPIKeyAuth(),
                cache=DualCache(),
                data=data,
                call_type="completion",
            )
        assert isinstance(result, dict)
        assert "tools" in result

    @pytest.mark.asyncio
    async def test_mcp_allowed_tool_sets_applied_guardrails_header(self):
        """Allowed MCP tool must call add_guardrail_to_applied_guardrails_header
        so the guardrail appears in x-litellm-applied-guardrails."""
        guardrail = self._make_guardrail(decision="allow", pattern=r"exa-.*")
        data = {"mcp_tool_name": "exa-web_search_exa"}

        await guardrail.async_pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data=data,
            call_type="call_mcp_tool",
        )

        applied = (data.get("metadata") or {}).get("applied_guardrails", [])
        assert "test-mcp-guardrail" in applied

    def test_supported_event_hooks_includes_mcp(self):
        """ToolPermissionGuardrail must declare pre_mcp_call and during_mcp_call
        so that mode=['pre_mcp_call','during_mcp_call'] passes _validate_event_hook."""
        guardrail = self._make_guardrail(decision="deny", pattern=r".*")
        assert GuardrailEventHooks.pre_mcp_call in (guardrail.supported_event_hooks or [])
        assert GuardrailEventHooks.during_mcp_call in (guardrail.supported_event_hooks or [])

    @pytest.mark.asyncio
    async def test_during_mcp_call_denied_raises_http_exception(self):
        """during_mcp_call: denied tool raises HTTPException(400) via async_moderation_hook."""
        guardrail = self._make_guardrail(decision="deny", pattern=r"exa-.*")
        data = {"mcp_tool_name": "exa-web_search_exa", "mcp_arguments": {"query": "test"}}

        with pytest.raises(HTTPException) as exc_info:
            await guardrail.async_moderation_hook(
                data=data,
                user_api_key_dict=UserAPIKeyAuth(),
                call_type="call_mcp_tool",
            )
        assert exc_info.value.status_code == 400
        assert "Violated guardrail policy" in exc_info.value.detail["error"]

    @pytest.mark.asyncio
    async def test_during_mcp_call_allowed_passes(self):
        """during_mcp_call: allowed tool does not raise."""
        guardrail = self._make_guardrail(decision="allow", pattern=r"exa-.*")
        data = {"mcp_tool_name": "exa-web_search_exa", "mcp_arguments": {"query": "test"}}

        await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="call_mcp_tool",
        )

    @pytest.mark.asyncio
    async def test_during_mcp_call_allowed_sets_applied_guardrails_header(self):
        """during_mcp_call: allowed tool must appear in applied-guardrails header."""
        guardrail = self._make_guardrail(decision="allow", pattern=r"exa-.*")
        data = {"mcp_tool_name": "exa-web_search_exa"}

        await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="call_mcp_tool",
        )

        applied = (data.get("metadata") or {}).get("applied_guardrails", [])
        assert "test-mcp-guardrail" in applied

    @pytest.mark.asyncio
    async def test_during_mcp_call_no_mcp_tool_name_is_noop(self):
        """during_mcp_call: missing mcp_tool_name is a no-op (does not raise)."""
        guardrail = self._make_guardrail(decision="deny", pattern=r".*")
        data = {"mcp_arguments": {"query": "test"}}

        await guardrail.async_moderation_hook(
            data=data,
            user_api_key_dict=UserAPIKeyAuth(),
            call_type="call_mcp_tool",
        )


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
