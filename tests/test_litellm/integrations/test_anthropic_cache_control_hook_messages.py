"""Tests for ``cache_control_injection_points`` on the Anthropic /v1/messages
path.

The OpenAI-shaped /v1/chat/completions path picks the injection points up via
``litellm_logging_obj.async_get_chat_completion_prompt`` inside
``litellm.completion``. The native Anthropic Messages path (used for
``anthropic/*`` and ``bedrock/*claude*`` deployments) historically did not, so
model-config caching silently no-op'd for Claude Code customers.

These tests pin three things:
    1. The hook supports the three /v1/messages-shaped locations:
       ``message``, ``system``, ``tools``.
    2. The hook writes ``cache_control`` at the *block* level (Anthropic
       rejects message-level cache_control on /v1/messages).
    3. The wiring inside ``anthropic_messages_handler`` actually forwards
       the mutations to ``base_llm_http_handler.anthropic_messages_handler``.
"""

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath("../.."))

from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook


# ---------------------------------------------------------------------------
# Unit tests for AnthropicCacheControlHook.apply_to_anthropic_messages_request
# ---------------------------------------------------------------------------


class TestApplyToAnthropicMessagesRequest:
    """Direct unit tests for the new Anthropic-Messages-format entrypoint."""

    def test_should_noop_when_no_injection_points(self):
        messages = [{"role": "user", "content": "hi"}]
        system = "you are helpful"
        tools = [{"name": "t", "input_schema": {"type": "object"}}]

        (
            new_messages,
            new_system,
            new_tools,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            tools=tools,
            non_default_params={"foo": "bar"},
        )

        assert new_messages == messages
        assert new_system == system
        assert new_tools == tools

    def test_should_pop_injection_points_from_non_default_params(self):
        non_default_params = {
            "cache_control_injection_points": [{"location": "system"}],
            "other": "value",
        }

        AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=[{"role": "user", "content": "hi"}],
            system="sys",
            tools=None,
            non_default_params=non_default_params,
        )

        assert "cache_control_injection_points" not in non_default_params
        assert non_default_params == {"other": "value"}

    def test_should_apply_cache_control_to_string_system_prompt(self):
        """``location: "system"`` upgrades a string system prompt into a
        single-block list so ``cache_control`` has a valid place to live."""
        (
            _,
            new_system,
            _,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=[{"role": "user", "content": "hi"}],
            system="you are a helpful assistant",
            tools=None,
            non_default_params={
                "cache_control_injection_points": [{"location": "system"}]
            },
        )

        assert new_system == [
            {
                "type": "text",
                "text": "you are a helpful assistant",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def test_should_apply_cache_control_to_list_system_prompt_last_block(self):
        """Multi-block system prompts get ``cache_control`` on the final
        block only, per Anthropic spec — preceding blocks are covered."""
        system = [
            {"type": "text", "text": "first block"},
            {"type": "text", "text": "second block"},
        ]

        (
            _,
            new_system,
            _,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=[{"role": "user", "content": "hi"}],
            system=system,
            tools=None,
            non_default_params={
                "cache_control_injection_points": [{"location": "system"}]
            },
        )

        assert "cache_control" not in new_system[0]
        assert new_system[1]["cache_control"] == {"type": "ephemeral"}

    def test_should_apply_cache_control_to_last_tool(self):
        tools = [
            {"name": "tool_a", "input_schema": {"type": "object"}},
            {"name": "tool_b", "input_schema": {"type": "object"}},
            {"name": "tool_c", "input_schema": {"type": "object"}},
        ]

        (
            _,
            _,
            new_tools,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=[{"role": "user", "content": "hi"}],
            system=None,
            tools=tools,
            non_default_params={
                "cache_control_injection_points": [{"location": "tools"}]
            },
        )

        assert "cache_control" not in new_tools[0]
        assert "cache_control" not in new_tools[1]
        assert new_tools[2]["cache_control"] == {"type": "ephemeral"}

    def test_should_promote_string_content_to_block_list_for_message_injection(self):
        """Anthropic /v1/messages rejects message-level ``cache_control`` —
        when content is a string the hook must wrap it into a single-block
        list so the marker is valid."""
        messages = [
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "first response"},
            {"role": "user", "content": "second turn"},
        ]

        (
            new_messages,
            _,
            _,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=None,
            tools=None,
            non_default_params={
                "cache_control_injection_points": [{"location": "message", "index": -1}]
            },
        )

        # Last message converted from string → list-of-blocks with cache_control
        assert new_messages[-1] == {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "second turn",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
        # Earlier messages untouched
        assert new_messages[0]["content"] == "first turn"
        # Caller's messages not mutated
        assert messages[-1]["content"] == "second turn"

    def test_should_apply_cache_control_to_last_block_of_list_content_message(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "page 1"},
                    {"type": "text", "text": "page 2"},
                    {"type": "text", "text": "page 3"},
                ],
            }
        ]

        (
            new_messages,
            _,
            _,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=None,
            tools=None,
            non_default_params={
                "cache_control_injection_points": [
                    {"location": "message", "role": "user"}
                ]
            },
        )

        content = new_messages[0]["content"]
        assert "cache_control" not in content[0]
        assert "cache_control" not in content[1]
        assert content[2]["cache_control"] == {"type": "ephemeral"}

    def test_should_auto_translate_role_system_to_location_system_when_no_system_message(
        self,
    ):
        """Customer-reported config shape: ``location: message, role: system``.

        On /v1/chat/completions this targets the system message inside
        ``messages`` (where OpenAI puts it). On /v1/messages the system
        prompt is a top-level parameter, NOT a message — so the config
        previously silently no-op'd, which is exactly the customer's repro:
        ``cache_creation_input_tokens=0`` despite a valid-looking config.

        The hook auto-translates ``role: system`` → ``location: system``
        when (a) no message in the array has ``role: system`` and (b) a
        top-level system prompt IS present. This makes the customer's exact
        existing config produce cache hits on /v1/messages with no config
        edit required.
        """
        messages = [{"role": "user", "content": "hi"}]

        (
            new_messages,
            new_system,
            _,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system="instructions",
            tools=None,
            non_default_params={
                "cache_control_injection_points": [
                    {"location": "message", "role": "system"}
                ]
            },
        )

        assert new_messages == messages
        assert new_system == [
            {
                "type": "text",
                "text": "instructions",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def test_should_not_auto_translate_when_system_role_message_exists(self):
        """If the caller actually has a system-role message in the array,
        respect that and target it — do not steal the marker for the
        top-level ``system`` param. This matches /v1/chat/completions
        semantics so users with mixed-shape requests aren't surprised."""
        messages = [
            {"role": "system", "content": "in-array system message"},
            {"role": "user", "content": "hi"},
        ]

        (
            new_messages,
            new_system,
            _,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system="top-level system",
            tools=None,
            non_default_params={
                "cache_control_injection_points": [
                    {"location": "message", "role": "system"}
                ]
            },
        )

        # System-role message in array was targeted → upgraded to block list
        assert new_messages[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        # Top-level system left alone
        assert new_system == "top-level system"

    def test_should_not_auto_translate_role_system_when_no_system_prompt_either(self):
        """If neither a system-role message nor a top-level system prompt
        exists, the injection point has nowhere to apply. The hook must
        not raise and must not mutate anything."""
        messages = [{"role": "user", "content": "hi"}]

        (
            new_messages,
            new_system,
            _,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=None,
            tools=None,
            non_default_params={
                "cache_control_injection_points": [
                    {"location": "message", "role": "system"}
                ]
            },
        )

        assert new_messages == messages
        assert new_system is None

    def test_should_handle_multiple_injection_points_in_one_call(self):
        """Realistic Claude Code config: cache the system prompt, the tool
        list, and the last user message all at once."""
        (
            new_messages,
            new_system,
            new_tools,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "second"},
            ],
            system="long system prompt",
            tools=[{"name": "search", "input_schema": {"type": "object"}}],
            non_default_params={
                "cache_control_injection_points": [
                    {"location": "system"},
                    {"location": "tools"},
                    {"location": "message", "index": -1},
                ]
            },
        )

        assert new_system[0]["cache_control"] == {"type": "ephemeral"}
        assert new_tools[0]["cache_control"] == {"type": "ephemeral"}
        assert new_messages[-1]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_should_respect_explicit_control_value(self):
        (
            _,
            new_system,
            _,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=[{"role": "user", "content": "hi"}],
            system="instructions",
            tools=None,
            non_default_params={
                "cache_control_injection_points": [
                    {
                        "location": "system",
                        "control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ]
            },
        )

        assert new_system[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}

    def test_should_noop_location_system_when_no_system_prompt(self):
        """No system prompt → ``location: "system"`` logs a warning and
        passes ``system=None`` through unchanged. It must not raise."""
        (
            _,
            new_system,
            _,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=[{"role": "user", "content": "hi"}],
            system=None,
            tools=None,
            non_default_params={
                "cache_control_injection_points": [{"location": "system"}]
            },
        )
        assert new_system is None

    def test_should_noop_location_tools_when_no_tools(self):
        (
            _,
            _,
            new_tools,
        ) = AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=[{"role": "user", "content": "hi"}],
            system=None,
            tools=None,
            non_default_params={
                "cache_control_injection_points": [{"location": "tools"}]
            },
        )
        assert new_tools is None

    def test_should_warn_on_out_of_bounds_message_index(self):
        with patch(
            "litellm.integrations.anthropic_cache_control_hook.verbose_logger"
        ) as mock_logger:
            AnthropicCacheControlHook.apply_to_anthropic_messages_request(
                messages=[{"role": "user", "content": "hi"}],
                system=None,
                tools=None,
                non_default_params={
                    "cache_control_injection_points": [
                        {"location": "message", "index": 99}
                    ]
                },
            )
            mock_logger.warning.assert_called_once()
            assert "out of bounds" in mock_logger.warning.call_args[0][0]

    def test_should_not_mutate_caller_messages_system_or_tools(self):
        messages = [{"role": "user", "content": "hi"}]
        system = [{"type": "text", "text": "instr"}]
        tools = [{"name": "t", "input_schema": {"type": "object"}}]

        AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=messages,
            system=system,
            tools=tools,
            non_default_params={
                "cache_control_injection_points": [
                    {"location": "message", "index": 0},
                    {"location": "system"},
                    {"location": "tools"},
                ]
            },
        )

        assert messages == [{"role": "user", "content": "hi"}]
        assert system == [{"type": "text", "text": "instr"}]
        assert tools == [{"name": "t", "input_schema": {"type": "object"}}]


# ---------------------------------------------------------------------------
# End-to-end wiring through anthropic_messages_handler
# ---------------------------------------------------------------------------


def _async_return(value):
    async def _coro():
        return value

    return _coro()


class TestWiringIntoAnthropicMessagesHandler:
    """Verify that the dispatch layer actually invokes the hook and forwards
    the mutated values to ``base_llm_http_handler.anthropic_messages_handler``.
    Without this wiring the hook is dead code on /v1/messages."""

    def _run_handler_and_capture(self, **call_kwargs):
        """Invoke ``anthropic_messages_handler`` with the native Anthropic
        provider config branch and capture the outbound payload."""
        from litellm.llms.anthropic.experimental_pass_through.messages.handler import (
            anthropic_messages_handler,
        )

        captured = {}

        def fake_base_handler(**kwargs):
            captured["messages"] = kwargs.get("messages")
            captured["anthropic_messages_optional_request_params"] = kwargs.get(
                "anthropic_messages_optional_request_params"
            )
            captured["kwargs"] = kwargs.get("kwargs")
            return MagicMock()

        with patch(
            "litellm.llms.anthropic.experimental_pass_through.messages.handler.base_llm_http_handler"
        ) as mock_handler:
            mock_handler.anthropic_messages_handler = fake_base_handler
            try:
                anthropic_messages_handler(**call_kwargs)
            except (ValueError, TypeError, AttributeError):
                pass

        return captured

    def test_should_inject_cache_control_into_system_on_bedrock_messages_path(self):
        """Repro of customer report: bedrock/us.anthropic.claude-sonnet-4-5
        + ``cache_control_injection_points`` + Claude Code /v1/messages.
        Before this fix the outbound system block had no cache_control,
        producing cache_creation_input_tokens=0."""
        captured = self._run_handler_and_capture(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            system="A very long Claude Code system prompt over 1024 tokens...",
            custom_llm_provider="bedrock",
            api_key="fake",
            cache_control_injection_points=[{"location": "system"}],
        )

        optional = captured["anthropic_messages_optional_request_params"]
        assert isinstance(optional["system"], list), (
            "system should have been upgraded from str to a block list so "
            "cache_control has a valid place to live"
        )
        assert optional["system"][0]["cache_control"] == {"type": "ephemeral"}
        # Param must not leak to upstream
        assert "cache_control_injection_points" not in optional
        assert "cache_control_injection_points" not in (captured["kwargs"] or {})

    def test_should_inject_cache_control_into_tools_on_messages_path(self):
        captured = self._run_handler_and_capture(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            tools=[
                {"name": "a", "input_schema": {"type": "object"}},
                {"name": "b", "input_schema": {"type": "object"}},
            ],
            custom_llm_provider="bedrock",
            api_key="fake",
            cache_control_injection_points=[{"location": "tools"}],
        )

        optional = captured["anthropic_messages_optional_request_params"]
        assert "cache_control" not in optional["tools"][0]
        assert optional["tools"][1]["cache_control"] == {"type": "ephemeral"}

    def test_should_inject_cache_control_into_last_message_block_on_messages_path(
        self,
    ):
        captured = self._run_handler_and_capture(
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "second"},
            ],
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            custom_llm_provider="bedrock",
            api_key="fake",
            cache_control_injection_points=[{"location": "message", "index": -1}],
        )

        last_message = captured["messages"][-1]
        assert isinstance(last_message["content"], list)
        assert last_message["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_should_handle_customer_exact_repro_config(self):
        """End-to-end repro of the customer report:

            model: bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0
            cache_control_injection_points:
              - location: message
                role: system

        Before this fix the outbound system payload had no cache marker on
        /v1/messages (Claude Code path), producing
        ``cache_creation_input_tokens=0``. After this fix the customer's
        unmodified config produces a system block with cache_control.
        """
        captured = self._run_handler_and_capture(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            system="A long Claude Code system prompt (>1024 tokens).",
            custom_llm_provider="bedrock",
            api_key="fake",
            cache_control_injection_points=[{"location": "message", "role": "system"}],
        )

        optional = captured["anthropic_messages_optional_request_params"]
        assert isinstance(optional["system"], list)
        assert optional["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_should_be_noop_when_injection_points_absent(self):
        """Regression guard: without the param, request must flow through
        unchanged. No silent upgrades to block-list shape."""
        captured = self._run_handler_and_capture(
            max_tokens=1024,
            messages=[{"role": "user", "content": "hello"}],
            model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            system="instructions",
            custom_llm_provider="bedrock",
            api_key="fake",
        )

        assert captured["messages"] == [{"role": "user", "content": "hello"}]
        optional = captured["anthropic_messages_optional_request_params"]
        assert optional.get("system") == "instructions"
