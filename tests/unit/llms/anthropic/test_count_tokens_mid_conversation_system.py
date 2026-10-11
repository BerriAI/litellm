"""
Regression tests: the Anthropic count_tokens transformation must never emit a
``role: "system"`` message at a position the API rejects.

The leading system run is counted through the top-level ``system`` parameter
(#45463). A system message later in the conversation used to leak into
``messages`` unchanged, where Anthropic rejects it (400) and the caller fell
back to the local tokenizer. The chat path already places those messages with
``place_mid_conversation_system``; the count_tokens path now applies the same
placement so the counted request matches the shape chat sends.
"""

from typing import Final

from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)

_CONFIG: Final = AnthropicCountTokensConfig()


def _transform(model: str, messages: list[dict]) -> dict:
    return _CONFIG.transform_request_to_count_tokens(model=model, messages=messages)


class TestMidConversationSystemPlacement:
    def test_unflagged_model_converts_mid_conversation_system_to_user(self):
        """claude-sonnet-4-5 does not accept a mid-conversation system role: the
        system text becomes part of a user turn, no system role stays in messages."""
        result: Final = _transform(
            "claude-sonnet-4-5",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "2+2?"},
            ],
        )
        roles: Final = [message["role"] for message in result["messages"]]
        assert "system" not in roles
        texts: Final = [
            block.get("text")
            for message in result["messages"]
            for block in (
                message["content"] if isinstance(message["content"], list) else [{"text": message["content"]}]
            )
            if isinstance(block, dict)
        ]
        assert "be terse" in texts

    def test_flagged_model_keeps_system_at_a_valid_position(self):
        """claude-opus-4-8 accepts a mid-conversation system role: the message is
        placed at an anchor the API accepts (after a user-type turn), not left
        dangling after an assistant turn."""
        result: Final = _transform(
            "claude-opus-4-8",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "2+2?"},
            ],
        )
        roles: Final = [message["role"] for message in result["messages"]]
        if "system" in roles:
            system_index: Final = roles.index("system")
            assert system_index > 0 and roles[system_index - 1] == "user"

    def test_leading_run_still_lifts_and_later_system_is_placed(self):
        """The leading run keeps moving to ``system`` (with its text intact) while
        a later system message is placed instead of leaking."""
        result: Final = _transform(
            "claude-sonnet-4-5",
            [
                {"role": "system", "content": "you are terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "system", "content": "and literal"},
                {"role": "user", "content": "2+2?"},
            ],
        )
        assert result.get("system") is not None
        system_texts: Final = [
            block.get("text")
            for block in (result["system"] if isinstance(result["system"], list) else [])
            if isinstance(block, dict)
        ]
        assert "you are terse" in system_texts
        roles: Final = [message["role"] for message in result["messages"]]
        assert "system" not in roles

    def test_tool_result_followed_by_system_keeps_tool_result_first(self):
        """A system run after a tool message is emitted after the tool run, so the
        merged user turn still opens with the tool_result block."""
        result: Final = _transform(
            "claude-sonnet-4-5",
            [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}],
                },
                {
                    "role": "tool",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}],
                },
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "thanks"},
            ],
        )
        roles: Final = [message["role"] for message in result["messages"]]
        assert "system" not in roles
        first_tool_index: Final = roles.index("tool")
        # tool content must remain attached to its own message before any converted text
        assert roles[first_tool_index + 1] == "user"
        wire_text: Final = str(result["messages"])
        assert "tool_result" in wire_text and "be terse" in wire_text
