import logging
from typing import Final

import pytest

import litellm
from litellm.anthropic_beta_gated_fields import (
    ANTHROPIC_INLINE_TOOLS_BETA_HEADER,
    ANTHROPIC_MID_CONVERSATION_SYSTEM_CLEAR_AT_BETA_HEADER,
    ANTHROPIC_THINKING_BINDING_CONTROLS_BETA_HEADER,
    BETA_GATED_FIELDS,
    active_beta_values,
    log_beta_gated_strips,
    strip_beta_gated_native_request,
    strip_beta_gated_request_fields,
    strip_beta_gated_system_messages,
    strip_beta_gated_thinking,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_MID_CONVERSATION_OUTPUT_CONFIG_BETA_HEADER,
    ANTHROPIC_MID_CONVERSATION_TOOL_CHANGES_BETA_HEADER,
    ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER,
)

EVERY_BETA: Final = frozenset(BETA_GATED_FIELDS)
BOUND_THINKING: Final = {
    "type": "adaptive",
    "block_binding": {"prefix_mismatch_behavior": "error"},
    "display": "updates",
}


def _system_turn() -> dict[str, object]:
    return {
        "role": "system",
        "clear_at": "end_turn",
        "output_config": {"effort": "low"},
        "content": [
            {"type": "text", "text": "stay terse"},
            {"type": "tool_addition", "tools": [{"name": "lookup"}]},
            {"type": "tool_removal", "tool_names": ["lookup"]},
        ],
    }


def test_thinking_with_every_beta_active_is_returned_as_is() -> None:
    result: Final = strip_beta_gated_thinking(BOUND_THINKING, EVERY_BETA, drop_params=True)
    assert result.thinking is BOUND_THINKING
    assert result.stripped == ()


def test_thinking_loses_only_the_field_whose_beta_is_absent() -> None:
    without_binding: Final = strip_beta_gated_thinking(
        BOUND_THINKING, EVERY_BETA - {ANTHROPIC_THINKING_BINDING_CONTROLS_BETA_HEADER}, drop_params=True
    )
    assert without_binding.thinking == {"type": "adaptive", "display": "updates"}
    assert without_binding.stripped == ("thinking.block_binding",)

    without_display: Final = strip_beta_gated_thinking(
        BOUND_THINKING, EVERY_BETA - {ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER}, drop_params=True
    )
    assert without_display.thinking == {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "error"}}
    assert without_display.stripped == ("thinking.display=updates",)
    assert BOUND_THINKING == {
        "type": "adaptive",
        "block_binding": {"prefix_mismatch_behavior": "error"},
        "display": "updates",
    }, "copy-on-write: the caller's thinking dict must survive for the fallback leg"


def test_thinking_display_other_than_updates_is_not_gated() -> None:
    thinking: Final = {"type": "adaptive", "display": "summarized"}
    result: Final = strip_beta_gated_thinking(thinking, frozenset(), drop_params=True)
    assert result.thinking is thinking


@pytest.mark.parametrize("thinking", [None, "enabled", 12, ["adaptive"]])
def test_non_mapping_thinking_passes_through(thinking: object) -> None:
    result: Final = strip_beta_gated_thinking(thinking, frozenset(), drop_params=True)
    assert result.thinking is thinking
    assert result.stripped == ()


def test_system_message_keeps_every_gated_field_when_the_betas_are_active() -> None:
    messages: Final = [{"role": "user", "content": "hi"}, _system_turn()]
    result: Final = strip_beta_gated_system_messages(messages, EVERY_BETA, drop_params=True)
    assert result.messages is messages
    assert result.stripped == ()


def test_system_message_loses_exactly_the_fields_whose_betas_are_absent() -> None:
    original: Final = _system_turn()
    messages: Final = [{"role": "user", "content": "hi"}, original]
    result: Final = strip_beta_gated_system_messages(
        messages,
        EVERY_BETA
        - {
            ANTHROPIC_MID_CONVERSATION_SYSTEM_CLEAR_AT_BETA_HEADER,
            ANTHROPIC_MID_CONVERSATION_TOOL_CHANGES_BETA_HEADER,
            ANTHROPIC_INLINE_TOOLS_BETA_HEADER,
        },
        drop_params=True,
    )
    assert result.messages == [
        {"role": "user", "content": "hi"},
        {
            "role": "system",
            "output_config": {"effort": "low"},
            "content": [{"type": "text", "text": "stay terse"}],
        },
    ]
    assert result.stripped == (
        "messages[].system.clear_at",
        "messages[].system.content[].tool_addition|tool_removal",
    )
    assert original == _system_turn(), "copy-on-write: the caller's system message must survive"


def test_per_turn_control_beta_also_unlocks_system_output_config() -> None:
    messages: Final = [{"role": "system", "output_config": {"effort": "low"}, "content": "terse"}]
    assert (
        strip_beta_gated_system_messages(
            messages, frozenset({"per-turn-control-2026-07-01"}), drop_params=True
        ).messages
        is messages
    )
    assert (
        strip_beta_gated_system_messages(
            messages, frozenset({ANTHROPIC_MID_CONVERSATION_OUTPUT_CONFIG_BETA_HEADER}), drop_params=True
        ).messages
        is messages
    )
    assert strip_beta_gated_system_messages(messages, frozenset(), drop_params=True).messages == [
        {"role": "system", "content": "terse"}
    ]


def test_inline_tools_beta_also_unlocks_system_tool_changes() -> None:
    messages: Final = [
        {
            "role": "system",
            "content": [{"type": "tool_removal", "tool_names": ["lookup"]}, {"type": "text", "text": "x"}],
        }
    ]
    assert (
        strip_beta_gated_system_messages(
            messages, frozenset({ANTHROPIC_INLINE_TOOLS_BETA_HEADER}), drop_params=True
        ).messages
        is messages
    )
    assert (
        strip_beta_gated_system_messages(
            messages, frozenset({ANTHROPIC_MID_CONVERSATION_TOOL_CHANGES_BETA_HEADER}), drop_params=True
        ).messages
        is messages
    )
    assert strip_beta_gated_system_messages(messages, frozenset(), drop_params=True).messages == [
        {"role": "system", "content": [{"type": "text", "text": "x"}]}
    ]


def test_system_block_with_unhashable_type_is_kept_for_the_provider_to_reject() -> None:
    messages: Final = [{"role": "system", "content": [{"type": [], "text": "x"}]}]
    result: Final = strip_beta_gated_system_messages(messages, frozenset(), drop_params=True)
    assert result.messages is messages
    assert result.stripped == ()


def test_system_message_emptied_by_the_strip_is_dropped() -> None:
    messages: Final = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": [{"type": "tool_removal", "tool_names": ["lookup"]}]},
        {"role": "assistant", "content": "ok"},
    ]
    result: Final = strip_beta_gated_system_messages(messages, frozenset(), drop_params=True)
    assert result.messages == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    assert result.stripped == (
        "messages[].system.content[].tool_addition|tool_removal",
        "messages[].system (emptied by the strip)",
    )


def test_emptied_trailing_system_turn_after_an_assistant_turn_is_sent_as_is() -> None:
    messages: Final = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": [{"type": "tool_removal", "tool_names": ["lookup"]}]},
    ]
    result: Final = strip_beta_gated_system_messages(messages, frozenset(), drop_params=True)
    assert result.messages is messages
    assert result.stripped == ()


def test_emptied_system_turn_is_dropped_when_a_user_turn_still_ends_the_conversation() -> None:
    messages: Final = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        {"role": "system", "content": [{"type": "tool_removal", "tool_names": ["lookup"]}]},
        {"role": "user", "content": "go on"},
    ]
    result: Final = strip_beta_gated_system_messages(messages, frozenset(), drop_params=True)
    assert [message["role"] for message in result.messages] == ["user", "assistant", "user"]


def test_system_turn_emptied_by_the_strip_stays_as_a_directive_when_its_output_config_survives() -> None:
    directive: Final = {
        "role": "system",
        "output_config": {"effort": "low"},
        "content": [{"type": "tool_removal", "tool_names": ["lookup"]}],
    }
    result: Final = strip_beta_gated_system_messages(
        [directive], frozenset({ANTHROPIC_MID_CONVERSATION_OUTPUT_CONFIG_BETA_HEADER}), drop_params=True
    )
    assert result.messages == [{"role": "system", "output_config": {"effort": "low"}, "content": []}]
    assert result.stripped == ("messages[].system.content[].tool_addition|tool_removal",)


def test_system_turn_emptied_by_the_strip_is_dropped_when_its_output_config_is_gated_too() -> None:
    directive: Final = {
        "role": "system",
        "output_config": {"effort": "low"},
        "content": [{"type": "tool_removal", "tool_names": ["lookup"]}],
    }
    result: Final = strip_beta_gated_system_messages([directive], frozenset(), drop_params=True)
    assert result.messages == []
    assert result.stripped[-1] == "messages[].system (emptied by the strip)"


def test_request_strip_replaces_only_thinking_and_messages_and_keeps_the_dict_otherwise() -> None:
    request: Final = {
        "model": "claude",
        "thinking": dict(BOUND_THINKING),
        "messages": [_system_turn()],
        "max_tokens": 5,
    }
    result: Final = strip_beta_gated_request_fields(request, frozenset(), drop_params=True)
    assert result.request == {
        "model": "claude",
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "system", "content": [{"type": "text", "text": "stay terse"}]}],
        "max_tokens": 5,
    }
    assert result.request is not request
    assert request["thinking"] == BOUND_THINKING
    assert request["messages"] == [_system_turn()]


def test_request_strip_is_identity_when_nothing_is_gated() -> None:
    request: Final = {
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "messages": [{"role": "user", "content": "hi"}],
    }
    result: Final = strip_beta_gated_request_fields(request, frozenset(), drop_params=True)
    assert result.request is request
    assert result.stripped == ()


@pytest.mark.parametrize("drop_params", [None, False, "false"])
def test_without_drop_params_every_gated_field_is_sent_as_is(
    drop_params: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    messages: Final = [{"role": "user", "content": "hi"}, _system_turn()]
    request: Final = {"thinking": BOUND_THINKING, "messages": messages}

    assert strip_beta_gated_thinking(BOUND_THINKING, frozenset(), drop_params=drop_params).thinking is BOUND_THINKING
    assert strip_beta_gated_system_messages(messages, frozenset(), drop_params=drop_params).messages is messages
    assert strip_beta_gated_request_fields(request, frozenset(), drop_params=drop_params).request is request
    native: Final = strip_beta_gated_native_request({}, {"thinking": BOUND_THINKING}, messages, drop_params=drop_params)
    assert (native.thinking, list(native.messages), native.thinking_stripped, native.messages_stripped) == (
        BOUND_THINKING,
        messages,
        False,
        False,
    )


def test_global_drop_params_strips_without_a_per_call_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", True)
    assert strip_beta_gated_thinking(BOUND_THINKING, frozenset(), drop_params=None).thinking == {"type": "adaptive"}


def test_active_beta_values_reads_the_header_case_insensitively_and_merges_body_betas() -> None:
    assert active_beta_values({"Anthropic-Beta": "a-2026-01-01, b-2026-01-01 ,"}, ["c-2026-01-01", 7]) == frozenset(
        {"a-2026-01-01", "b-2026-01-01", "c-2026-01-01"}
    )
    assert active_beta_values({}) == frozenset()


def test_every_strip_is_logged_once_at_warning_with_the_fields_model_and_provider(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        log_beta_gated_strips(("thinking.block_binding", "thinking.block_binding"), model="claude", provider="azure_ai")
        log_beta_gated_strips((), model="claude", provider="azure_ai")
    warnings: Final = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert (
        "Dropped thinking.block_binding (unlocked by anthropic-beta thinking-binding-controls-2026-08-01) for "
        "model=claude provider=azure_ai" in warnings[0].getMessage()
    )
