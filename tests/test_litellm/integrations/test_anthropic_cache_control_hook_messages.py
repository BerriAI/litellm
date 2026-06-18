"""Unit + wiring tests for cache_control_injection_points on the native
Anthropic ``/v1/messages`` path. Regression coverage for
BerriAI/litellm#30293, where deployment-level cache injection was silently
dropped on ``/v1/messages`` (worked only on ``/chat/completions``).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook

EPHEMERAL = {"type": "ephemeral"}


def _apply(messages, system, tools, injection_points):
    """Helper: run the hook entrypoint with the given injection points."""
    non_default_params = {"cache_control_injection_points": injection_points}
    return AnthropicCacheControlHook.apply_to_anthropic_messages_request(
        messages=messages,
        system=system,
        tools=tools,
        non_default_params=non_default_params,
    )


class TestApplyToAnthropicMessagesRequest:
    def test_noop_when_no_injection_points(self):
        messages = [{"role": "user", "content": "hi"}]
        out_messages, out_system, out_tools = _apply(messages, "sys", None, [])
        assert out_messages == messages
        assert out_system == "sys"
        assert out_tools is None

    def test_pops_injection_points_from_params(self):
        non_default_params = {
            "cache_control_injection_points": [{"location": "system"}],
            "other": 1,
        }
        AnthropicCacheControlHook.apply_to_anthropic_messages_request(
            messages=[{"role": "user", "content": "hi"}],
            system="sys",
            tools=None,
            non_default_params=non_default_params,
        )
        assert "cache_control_injection_points" not in non_default_params
        assert non_default_params["other"] == 1

    def test_string_system_promoted_to_block_with_cache_control(self):
        _, out_system, _ = _apply(
            [{"role": "user", "content": "hi"}],
            "long system prompt",
            None,
            [{"location": "system"}],
        )
        assert out_system == [
            {
                "type": "text",
                "text": "long system prompt",
                "cache_control": EPHEMERAL,
            }
        ]

    def test_list_system_marks_last_block(self):
        system = [
            {"type": "text", "text": "block 1"},
            {"type": "text", "text": "block 2"},
        ]
        _, out_system, _ = _apply(
            [{"role": "user", "content": "hi"}], system, None, [{"location": "system"}]
        )
        assert "cache_control" not in out_system[0]
        assert out_system[1]["cache_control"] == EPHEMERAL

    def test_tools_marks_last_tool(self):
        tools = [{"name": "a"}, {"name": "b"}]
        _, _, out_tools = _apply(
            [{"role": "user", "content": "hi"}], None, tools, [{"location": "tools"}]
        )
        assert "cache_control" not in out_tools[0]
        assert out_tools[1]["cache_control"] == EPHEMERAL

    def test_tools_skips_non_cacheable_tool_search_tool(self):
        # Anthropic rejects cache_control on tool-search tools; the marker must
        # land on the last *cacheable* tool, not the trailing search tool.
        tools = [
            {"name": "real_tool"},
            {"type": "tool_search_tool_regex_20251119", "name": "search"},
        ]
        _, _, out_tools = _apply(
            [{"role": "user", "content": "hi"}], None, tools, [{"location": "tools"}]
        )
        assert out_tools[0]["cache_control"] == EPHEMERAL
        assert "cache_control" not in out_tools[1]

    def test_tools_all_non_cacheable_is_noop(self):
        tools = [{"type": "tool_search_tool_bm25_20251119", "name": "s"}]
        _, _, out_tools = _apply(
            [{"role": "user", "content": "hi"}], None, tools, [{"location": "tools"}]
        )
        assert "cache_control" not in out_tools[0]

    def test_message_string_content_promoted_to_block(self):
        messages = [{"role": "user", "content": "hello"}]
        out_messages, _, _ = _apply(
            messages, None, None, [{"location": "message", "index": -1}]
        )
        assert out_messages[0]["content"] == [
            {"type": "text", "text": "hello", "cache_control": EPHEMERAL}
        ]

    def test_message_list_content_marks_last_block(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
            }
        ]
        out_messages, _, _ = _apply(
            messages, None, None, [{"location": "message", "index": -1}]
        )
        content = out_messages[0]["content"]
        assert "cache_control" not in content[0]
        assert content[1]["cache_control"] == EPHEMERAL

    def test_message_targets_by_role(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        out_messages, _, _ = _apply(
            messages, None, None, [{"location": "message", "role": "user"}]
        )
        assert out_messages[0]["content"][0]["cache_control"] == EPHEMERAL
        assert out_messages[2]["content"][0]["cache_control"] == EPHEMERAL
        # assistant message untouched (still a string)
        assert out_messages[1]["content"] == "reply"

    def test_role_system_auto_translates_to_system_prompt(self):
        # The customer's exact /chat-completions config: {message, role: system}.
        # On /v1/messages there is no role:system message, so it must target the
        # top-level system prompt instead of silently matching nothing.
        messages = [{"role": "user", "content": "hi"}]
        _, out_system, _ = _apply(
            messages, "system prompt", None, [{"location": "message", "role": "system"}]
        )
        assert out_system == [
            {"type": "text", "text": "system prompt", "cache_control": EPHEMERAL}
        ]

    def test_role_system_not_translated_when_system_message_present(self):
        # If an explicit role:system message exists, target it (don't redirect).
        messages = [
            {"role": "system", "content": "in-band system"},
            {"role": "user", "content": "hi"},
        ]
        out_messages, out_system, _ = _apply(
            messages,
            "top-level system",
            None,
            [{"location": "message", "role": "system"}],
        )
        assert out_messages[0]["content"][0]["cache_control"] == EPHEMERAL
        # top-level system prompt left alone
        assert out_system == "top-level system"

    def test_role_system_noop_when_no_system_at_all(self):
        messages = [{"role": "user", "content": "hi"}]
        out_messages, out_system, _ = _apply(
            messages, None, None, [{"location": "message", "role": "system"}]
        )
        assert out_messages[0]["content"] == "hi"
        assert out_system is None

    def test_explicit_control_value_respected(self):
        control = {"type": "ephemeral", "ttl": "1h"}
        _, out_system, _ = _apply(
            [{"role": "user", "content": "hi"}],
            "sys",
            None,
            [{"location": "system", "control": control}],
        )
        assert out_system[0]["cache_control"] == control

    def test_multiple_injection_points(self):
        messages = [{"role": "user", "content": "hi"}]
        out_messages, out_system, out_tools = _apply(
            messages,
            "sys",
            [{"name": "t"}],
            [
                {"location": "system"},
                {"location": "tools"},
                {"location": "message", "index": -1},
            ],
        )
        assert out_system[0]["cache_control"] == EPHEMERAL
        assert out_tools[0]["cache_control"] == EPHEMERAL
        assert out_messages[0]["content"][0]["cache_control"] == EPHEMERAL

    def test_out_of_bounds_index_is_noop(self):
        messages = [{"role": "user", "content": "hi"}]
        out_messages, _, _ = _apply(
            messages, None, None, [{"location": "message", "index": 5}]
        )
        assert out_messages[0]["content"] == "hi"

    def test_inputs_not_mutated(self):
        messages = [{"role": "user", "content": "hi"}]
        system = "sys"
        tools = [{"name": "t"}]
        _apply(
            messages,
            system,
            tools,
            [
                {"location": "system"},
                {"location": "tools"},
                {"location": "message", "index": -1},
            ],
        )
        # originals untouched (deep copy)
        assert messages == [{"role": "user", "content": "hi"}]
        assert system == "sys"
        assert tools == [{"name": "t"}]

    def test_tool_config_injection_point_forwarded_downstream(self):
        # `tool_config` (Bedrock) is not representable in the /v1/messages
        # payload here; it must be forwarded downstream for the provider
        # transform (e.g. Bedrock Converse) to consume, NOT silently dropped.
        # Regression guard for the original /v1/messages bypass: before, an
        # unrecognised location was popped and lost.
        non_default_params = {
            "cache_control_injection_points": [{"location": "tool_config"}],
        }
        out_messages, out_system, out_tools = (
            AnthropicCacheControlHook.apply_to_anthropic_messages_request(
                messages=[{"role": "user", "content": "hi"}],
                system="sys",
                tools=[{"name": "t"}],
                non_default_params=non_default_params,
            )
        )
        # forwarded for downstream handling, not consumed
        assert non_default_params["cache_control_injection_points"] == [
            {"location": "tool_config"}
        ]
        # payload left untouched for the location we don't handle here
        assert out_system == "sys"
        assert out_tools == [{"name": "t"}]
        assert out_messages == [{"role": "user", "content": "hi"}]

    def _count_all_blocks(self, messages, system, tools):
        """Count cache_control markers across system + tools + messages."""
        total = 0
        if isinstance(system, list):
            total += sum(1 for b in system if b.get("cache_control"))
        if isinstance(tools, list):
            total += sum(1 for t in tools if t.get("cache_control"))
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                total += sum(
                    1 for b in content if isinstance(b, dict) and b.get("cache_control")
                )
        return total

    def test_respects_max_cache_control_blocks_limit(self):
        # role:user matches 6 messages, but Anthropic allows at most 4 markers.
        messages = [{"role": "user", "content": f"m{i}"} for i in range(6)]
        out_messages, _, _ = _apply(
            messages, None, None, [{"location": "message", "role": "user"}]
        )
        marked = sum(
            1
            for m in out_messages
            if isinstance(m["content"], list) and m["content"][-1].get("cache_control")
        )
        assert marked == 4  # capped at MAX_CACHE_CONTROL_BLOCKS

    def test_existing_cache_control_counts_toward_limit(self):
        # Client already marked 3 message blocks; only 1 slot remains.
        marked_block = [{"type": "text", "text": "x", "cache_control": EPHEMERAL}]
        messages = [
            {"role": "user", "content": list(marked_block)},
            {"role": "user", "content": list(marked_block)},
            {"role": "user", "content": list(marked_block)},
            {"role": "user", "content": "d"},
            {"role": "user", "content": "e"},
        ]
        out_messages, out_system, out_tools = _apply(
            messages, None, None, [{"location": "message", "role": "user"}]
        )
        # 3 client markers preserved + exactly 1 injected = 4 total, not 5.
        assert self._count_all_blocks(out_messages, out_system, out_tools) == 4

    def test_system_tools_messages_share_block_budget(self):
        # system + tools + 5 role-matched messages would be 7 markers; cap at 4.
        messages = [{"role": "user", "content": f"m{i}"} for i in range(5)]
        out_messages, out_system, out_tools = _apply(
            messages,
            "sys",
            [{"name": "t"}],
            [
                {"location": "system"},
                {"location": "tools"},
                {"location": "message", "role": "user"},
            ],
        )
        assert self._count_all_blocks(out_messages, out_system, out_tools) == 4

    def test_tool_config_does_not_reserve_budget_on_native_path(self):
        # tool_config is inert on the native /v1/messages path (only the
        # chat/completions Converse transform consumes it), so it must NOT
        # reserve a block slot here: all 4 markers go to the message points,
        # and tool_config is still forwarded for completeness.
        messages = [{"role": "user", "content": f"m{i}"} for i in range(6)]
        non_default_params = {
            "cache_control_injection_points": [
                {"location": "message", "role": "user"},
                {"location": "tool_config"},
            ],
        }
        out_messages, out_system, out_tools = (
            AnthropicCacheControlHook.apply_to_anthropic_messages_request(
                messages=messages,
                system=None,
                tools=None,
                non_default_params=non_default_params,
            )
        )
        assert self._count_all_blocks(out_messages, out_system, out_tools) == 4
        assert non_default_params["cache_control_injection_points"] == [
            {"location": "tool_config"}
        ]

    def test_nested_tool_result_markers_count_toward_limit(self):
        # 4 markers already present inside tool_result.content nested blocks =
        # cap reached. A further system injection would be the 5th and must be
        # skipped (Anthropic rejects >4 cache_control blocks).
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": f"t{i}",
                        "content": [
                            {
                                "type": "text",
                                "text": f"r{i}",
                                "cache_control": EPHEMERAL,
                            }
                        ],
                    }
                    for i in range(4)
                ],
            }
        ]
        _, out_system, _ = _apply(messages, "sys", None, [{"location": "system"}])
        # budget already full from the 4 nested markers -> system left untouched
        assert out_system == "sys"

    def test_does_not_overwrite_existing_message_marker(self):
        # A message the client already marked is left untouched.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "keep",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ],
            }
        ]
        out_messages, _, _ = _apply(
            messages, None, None, [{"location": "message", "role": "user"}]
        )
        # original 1h TTL preserved, not overwritten with default ephemeral
        assert out_messages[0]["content"][0]["cache_control"] == {
            "type": "ephemeral",
            "ttl": "1h",
        }

    def test_mixed_known_and_unknown_locations(self):
        # Known locations are applied to the payload; unknown ones (tool_config)
        # are forwarded. The two must not interfere.
        non_default_params = {
            "cache_control_injection_points": [
                {"location": "system"},
                {"location": "tool_config"},
            ],
        }
        _, out_system, _ = (
            AnthropicCacheControlHook.apply_to_anthropic_messages_request(
                messages=[{"role": "user", "content": "hi"}],
                system="sys",
                tools=None,
                non_default_params=non_default_params,
            )
        )
        # system applied inline
        assert out_system[0]["cache_control"] == EPHEMERAL
        # only the unhandled tool_config point survives for downstream
        assert non_default_params["cache_control_injection_points"] == [
            {"location": "tool_config"}
        ]


class TestWiringIntoAnthropicMessagesHandler:
    """End-to-end: deployment-level cache_control_injection_points must reach
    the outbound /v1/messages payload via the native handler.
    """

    @pytest.mark.asyncio
    async def test_injection_applied_on_native_messages_path(self, monkeypatch):
        from litellm.llms.anthropic.experimental_pass_through.messages import handler

        captured = {}

        def fake_handler(*args, **kwargs):
            # The native path forwards system/tools inside
            # `anthropic_messages_optional_request_params` and the remaining
            # caller kwargs inside `kwargs`.
            captured["optional_params"] = kwargs.get(
                "anthropic_messages_optional_request_params"
            )
            captured["inner_kwargs"] = kwargs.get("kwargs") or {}
            return {"id": "msg_test", "type": "message", "content": []}

        monkeypatch.setattr(
            handler.base_llm_http_handler,
            "anthropic_messages_handler",
            fake_handler,
        )

        await handler.anthropic_messages(
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            model="anthropic/claude-sonnet-4-5",
            system="a long cached system prompt",
            custom_llm_provider="anthropic",
            cache_control_injection_points=[
                {"location": "message", "role": "system"},
            ],
            api_key="fake-key",
        )

        # system prompt promoted to a cached block on the outbound payload
        assert captured["optional_params"]["system"] == [
            {
                "type": "text",
                "text": "a long cached system prompt",
                "cache_control": EPHEMERAL,
            }
        ]
        # injection param popped, not forwarded upstream as an unknown field
        assert "cache_control_injection_points" not in captured["inner_kwargs"]

    @pytest.mark.asyncio
    async def test_noop_when_no_injection_points(self, monkeypatch):
        from litellm.llms.anthropic.experimental_pass_through.messages import handler

        captured = {}

        def fake_handler(*args, **kwargs):
            captured["optional_params"] = kwargs.get(
                "anthropic_messages_optional_request_params"
            )
            return {"id": "msg_test", "type": "message", "content": []}

        monkeypatch.setattr(
            handler.base_llm_http_handler,
            "anthropic_messages_handler",
            fake_handler,
        )

        await handler.anthropic_messages(
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            model="anthropic/claude-sonnet-4-5",
            system="plain system prompt",
            custom_llm_provider="anthropic",
            api_key="fake-key",
        )

        # unchanged when no injection points configured
        assert captured["optional_params"]["system"] == "plain system prompt"

    @pytest.mark.asyncio
    async def test_injection_param_preserved_for_fallback_provider(self, monkeypatch):
        # A non-native provider (Gemini) falls back to the chat/completions
        # adapter. The native Anthropic block rewrite must NOT run here, and the
        # param must survive so the downstream OpenAI-path hook applies caching
        # in the form Gemini understands. Regression guard for the fallback path.
        from litellm.llms.anthropic.experimental_pass_through.messages import handler

        captured = {}

        def fake_adapter(*args, **kwargs):
            captured["kwargs"] = kwargs
            captured["messages"] = kwargs.get("messages")
            return {"id": "msg_test", "type": "message", "content": []}

        monkeypatch.setattr(
            handler.LiteLLMMessagesToCompletionTransformationHandler,
            "anthropic_messages_handler",
            fake_adapter,
        )

        await handler.anthropic_messages(
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
            model="gemini/gemini-2.0-flash",
            custom_llm_provider="gemini",
            cache_control_injection_points=[{"location": "message", "role": "user"}],
            api_key="fake-key",
        )

        # param left untouched for the fallback adapter (not consumed here)
        assert captured["kwargs"].get("cache_control_injection_points") == [
            {"location": "message", "role": "user"}
        ]
        # messages not rewritten into Anthropic cached-block form by our hook
        assert captured["messages"] == [{"role": "user", "content": "hello"}]
