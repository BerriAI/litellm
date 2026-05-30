"""
Unit tests for the compact_20260112 polyfill editor.

Coverage:
- trigger.value < 50k  → AnthropicContextManagementError(400)
- opt-in gate (no summary model)  → summary_model_not_configured
- slice-only path (existing compaction block, under threshold)
- full summary path (over threshold, summary fires)
- summary call raises  → summary_call_failed
- summary response missing <summary> tags  → summary_extraction_failed
- pause_after_compaction: true  → pause_after_compaction_ignored warning, proceeds
- custom instructions  → default prompt is not used even when tools present
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.anthropic.experimental_pass_through.context_management import (
    AnthropicContextManagementError,
    apply_context_management,
)
from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
    _augment_system_with_summary,
    _extract_summary_text,
    _select_last_user_question,
    _slice_around_compaction_block,
    _strip_compaction_blocks,
    apply_client_compaction_block_history,
    apply_compact_20260112,
)
from litellm.llms.anthropic.experimental_pass_through.context_management.result import (
    PolyfillResult,
)

MODEL = "openai/gpt-4o"

_EDIT_SPEC_DEFAULT: Dict[str, Any] = {"type": "compact_20260112"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_messages() -> List[Dict[str, Any]]:
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]},
        {"role": "user", "content": "What is 2+2?"},
    ]


def _messages_with_compaction(summary: str = "prev summary") -> List[Dict[str, Any]]:
    """History that already has a compaction block in an assistant turn."""
    return [
        {"role": "user", "content": "older question"},
        {
            "role": "assistant",
            "content": [{"type": "compaction", "content": summary}],
        },
        {"role": "user", "content": "newer question"},
        {"role": "assistant", "content": [{"type": "text", "text": "newer reply"}]},
        {"role": "user", "content": "latest question"},
    ]


def _make_mock_response(
    content: str,
    prompt_tokens: int = 50,
    completion_tokens: int = 100,
) -> MagicMock:
    response = MagicMock()
    choice = MagicMock()
    message = MagicMock()
    message.content = content
    choice.message = message
    response.choices = [choice]
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    response.usage = usage
    return response


# ---------------------------------------------------------------------------
# Unit: helper functions
# ---------------------------------------------------------------------------


def test_applied_edits_for_response_omits_compact_without_block_or_error():
    """No compaction block and no error: omit the compact_20260112 edit."""
    result = PolyfillResult(
        messages=[],
        system="summary on system",
        applied_edits=[{"type": "compact_20260112"}],
        compaction_block=None,
    )
    assert result.applied_edits_for_response() is None


def test_applied_edits_for_response_includes_compact_when_error_present():
    """Error states must surface to the client so operators can debug."""
    for error in (
        "summary_model_not_configured",
        "summary_call_failed",
        "summary_extraction_failed",
    ):
        result = PolyfillResult(
            messages=[],
            system=None,
            applied_edits=[{"type": "compact_20260112", "error": error}],
            compaction_block=None,
        )
        visible = result.applied_edits_for_response()
        assert visible is not None, error
        assert visible[0]["error"] == error


def test_applied_edits_for_response_includes_compact_when_block_present():
    result = PolyfillResult(
        messages=[],
        system=None,
        applied_edits=[
            {
                "type": "compact_20260112",
                "summary_input_tokens": 10,
                "summary_output_tokens": 5,
            }
        ],
        compaction_block={"type": "compaction", "content": "summary"},
    )
    visible = result.applied_edits_for_response()
    assert visible is not None
    assert visible[0]["type"] == "compact_20260112"
    assert visible[0]["summary_input_tokens"] == 10


def test_slice_around_compaction_block_found():
    messages = _messages_with_compaction("my summary")
    sliced, block = _slice_around_compaction_block(messages)
    assert block is not None
    assert block["type"] == "compaction"
    assert block["content"] == "my summary"
    # Sliced list starts at the assistant turn containing the compaction block
    assert sliced[0]["role"] == "assistant"
    assert len(sliced) == 4  # assistant(compaction), user, assistant, user


def test_slice_around_compaction_block_not_found():
    messages = _simple_messages()
    sliced, block = _slice_around_compaction_block(messages)
    assert block is None
    assert sliced is messages  # same object, no copy


def test_strip_compaction_blocks_removes_block():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "compaction", "content": "summary"},
                {"type": "text", "text": "hello"},
            ],
        }
    ]
    stripped = _strip_compaction_blocks(messages)
    assert len(stripped) == 1
    content = stripped[0]["content"]
    assert all(b["type"] != "compaction" for b in content)
    assert len(content) == 1
    assert content[0]["type"] == "text"


def test_select_last_user_question_strips_tool_result_from_mixed_turn():
    """Mixed [tool_result, text] turn: keep text, drop tool_result blocks."""
    messages = [
        {"role": "user", "content": "earlier"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "res"},
                {"type": "text", "text": "follow-up question"},
            ],
        },
    ]
    selected = _select_last_user_question(messages)
    assert len(selected) == 1
    assert selected[0]["role"] == "user"
    content = selected[0]["content"]
    assert isinstance(content, list)
    assert all(b.get("type") != "tool_result" for b in content)
    assert any(
        b.get("type") == "text" and b.get("text") == "follow-up question"
        for b in content
    )


def test_select_last_user_question_skips_pure_tool_result_turn():
    """Pure tool_result turn: skip and walk back to a real user turn."""
    messages = [
        {"role": "user", "content": "real question"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "a", "name": "x", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "a", "content": "res"}],
        },
    ]
    selected = _select_last_user_question(messages)
    assert len(selected) == 1
    assert selected[0]["content"] == "real question"


def test_select_last_user_question_falls_back_when_no_eligible_turn():
    """Only tool_result-only user turns: emit a synthetic continuation prompt."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "a", "content": "res"}],
        },
    ]
    selected = _select_last_user_question(messages)
    assert len(selected) == 1
    assert selected[0]["role"] == "user"
    assert isinstance(selected[0]["content"], str)


def test_strip_compaction_blocks_drops_compaction_only_turn():
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [{"type": "compaction", "content": "summary"}],
        },
        {"role": "user", "content": "bye"},
    ]
    stripped = _strip_compaction_blocks(messages)
    assert len(stripped) == 2
    assert stripped[0]["role"] == "user"
    assert stripped[1]["role"] == "user"


def test_augment_system_with_summary_none_system():
    result = _augment_system_with_summary(None, "my summary")
    assert isinstance(result, str)
    assert "my summary" in result


def test_augment_system_with_summary_string_system():
    result = _augment_system_with_summary("You are helpful.", "my summary")
    assert isinstance(result, str)
    assert result.startswith("Previous conversation summary:")
    assert "my summary" in result
    assert "You are helpful." in result


def test_augment_system_with_summary_list_system():
    system = [{"type": "text", "text": "existing system"}]
    result = _augment_system_with_summary(system, "my summary")
    assert isinstance(result, list)
    assert result[0]["type"] == "text"
    text = result[0]["text"]
    assert "my summary" in text
    assert "existing system" in text


def test_extract_summary_text_found():
    raw = "Here is the summary:\n<summary>Key points from chat</summary>\nDone."
    assert _extract_summary_text(raw) == "Key points from chat"


def test_extract_summary_text_missing_tags():
    assert _extract_summary_text("No tags here") is None


def test_extract_summary_text_none():
    assert _extract_summary_text(None) is None


def test_extract_summary_text_case_insensitive():
    raw = "<SUMMARY>uppercase tags</SUMMARY>"
    assert _extract_summary_text(raw) == "uppercase tags"


# ---------------------------------------------------------------------------
# Editor: validation
# ---------------------------------------------------------------------------


async def test_trigger_below_minimum_raises():
    with pytest.raises(AnthropicContextManagementError) as exc_info:
        await apply_compact_20260112(
            model=MODEL,
            messages=_simple_messages(),
            tools=None,
            system=None,
            edit_spec={
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 10_000},
            },
        )
    assert exc_info.value.status_code == 400
    assert "50000" in exc_info.value.message


async def test_trigger_at_minimum_does_not_raise():
    """Exactly 50 000 is allowed — only strictly less than 50k is rejected."""
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
        return_value=None,
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=_simple_messages(),
            tools=None,
            system=None,
            edit_spec={
                "type": "compact_20260112",
                "trigger": {"type": "input_tokens", "value": 50_000},
            },
        )
    # Reached opt-in gate (no summary model); no error raised from trigger check
    assert result.applied_edits[0]["error"] == "summary_model_not_configured"


# ---------------------------------------------------------------------------
# Editor: opt-in gate
# ---------------------------------------------------------------------------


async def test_opt_in_gating_no_summary_model_configured():
    messages = _simple_messages()
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
        return_value=None,
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system="system prompt",
            edit_spec=_EDIT_SPEC_DEFAULT,
        )
    assert result.applied_edits[0]["error"] == "summary_model_not_configured"
    assert result.messages == messages
    assert result.system == "system prompt"
    assert result.compaction_block is None
    assert result.iterations_usage is None


async def test_opt_in_gating_no_summary_model_keeps_post_compaction_tail():
    """No summary model + prior compaction block forwards the full tail.

    The prior summary lives on the system prefix; the post-compaction turns it
    does not cover must be forwarded unchanged rather than collapsed to the
    latest user question (which would strip intermediate turns the model needs).
    """
    messages = _messages_with_compaction("prior summary text")

    with patch(
        "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
        return_value=None,
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    assert result.applied_edits[0]["error"] == "summary_model_not_configured"
    assert result.system is not None
    assert "prior summary text" in str(result.system)
    assert result.compaction_block is None
    assert result.iterations_usage is None
    # Post-compaction tail forwarded unchanged (compaction blocks stripped).
    assert [m["role"] for m in result.messages] == ["user", "assistant", "user"]
    assert result.messages[0]["content"] == "newer question"
    assert result.messages[-1]["content"] == "latest question"
    for msg in result.messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                assert block.get("type") != "compaction"


# ---------------------------------------------------------------------------
# Client compaction block without context_management
# ---------------------------------------------------------------------------


def test_client_compaction_block_history_without_context_management():
    """Compaction in messages alone triggers slice-only forwarding.

    The prior summary is prepended to ``system``; the post-compaction tail is
    forwarded unchanged so the model sees the recent turns the summary does
    not cover. Compaction blocks themselves are stripped from messages so
    non-Anthropic backends don't reject them.
    """
    messages = _messages_with_compaction("prior summary text")

    result = apply_client_compaction_block_history(messages=messages, system=None)

    assert result is not None
    assert result.system is not None
    assert "prior summary text" in str(result.system)
    assert result.compaction_block is None
    assert result.applied_edits == []
    # Post-compaction tail: newer question, newer reply, latest question.
    assert [m["role"] for m in result.messages] == ["user", "assistant", "user"]
    assert result.messages[0]["content"] == "newer question"
    assert result.messages[-1]["content"] == "latest question"
    for msg in result.messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                assert block.get("type") != "compaction"


def test_client_compaction_block_history_no_compaction_returns_none():
    result = apply_client_compaction_block_history(
        messages=_simple_messages(), system="base"
    )
    assert result is None


# ---------------------------------------------------------------------------
# Editor: slice-only path
# ---------------------------------------------------------------------------


async def test_slice_only_path_with_existing_compaction_block():
    """Phase A slices; Phase B token count is below threshold; no summary call.

    The prior compaction summary lives on the system prefix; the
    post-compaction tail is forwarded unchanged so the model retains the
    recent turns the summary does not cover. Compaction blocks themselves
    are stripped from messages.
    """
    messages = _messages_with_compaction("prior summary text")

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=500),  # well under threshold
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    # System should have the prior summary prefixed
    assert result.system is not None
    assert "prior summary text" in str(result.system)

    # No new compaction block; no iterations_usage
    assert result.compaction_block is None
    assert result.iterations_usage is None

    # Main call: summary on system + full post-compaction tail (no compaction blocks).
    assert [m["role"] for m in result.messages] == ["user", "assistant", "user"]
    assert result.messages[0]["content"] == "newer question"
    assert result.messages[-1]["content"] == "latest question"
    for msg in result.messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                assert block.get("type") != "compaction"


async def test_slice_only_no_compaction_block_under_threshold():
    """No prior compaction block, and token count is below threshold — pure pass-through."""
    messages = _simple_messages()
    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=500),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    assert result.messages == messages
    assert result.compaction_block is None
    assert result.iterations_usage is None
    assert not result.applied_edits[0].get("error")


# ---------------------------------------------------------------------------
# Editor: full summary path
# ---------------------------------------------------------------------------


async def test_full_summary_path():
    """Over threshold: summary call fires, compaction_block and iterations_usage returned."""
    messages = _simple_messages()
    mock_response = _make_mock_response(
        "<summary>Condensed history</summary>", prompt_tokens=200, completion_tokens=50
    )

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),  # over 150k threshold
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    assert result.compaction_block is not None
    assert result.compaction_block["type"] == "compaction"
    assert result.compaction_block["content"] == "Condensed history"

    assert result.iterations_usage is not None
    assert len(result.iterations_usage) == 1
    assert result.iterations_usage[0]["type"] == "compaction"
    assert result.iterations_usage[0]["input_tokens"] == 200
    assert result.iterations_usage[0]["output_tokens"] == 50

    # System must have summary prefixed
    assert "Condensed history" in str(result.system)

    # applied_edits should have usage fields
    edit = result.applied_edits[0]
    assert edit["type"] == "compact_20260112"
    assert edit.get("summary_input_tokens") == 200
    assert edit.get("summary_output_tokens") == 50

    # Downstream messages must not contain a compaction block
    for msg in result.messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                assert block.get("type") != "compaction"


async def test_full_summary_path_uses_router_when_available():
    """When llm_router is provided, its acompletion method is called instead of litellm."""
    messages = _simple_messages()
    mock_response = _make_mock_response("<summary>Router summary</summary>")
    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(return_value=mock_response)

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="my-summary-model",
        ),
        patch("litellm.token_counter", return_value=200_000),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            llm_router=mock_router,
        )

    mock_router.acompletion.assert_called_once()
    call_kwargs = mock_router.acompletion.call_args.kwargs
    assert call_kwargs["model"] == "my-summary-model"

    assert result.compaction_block is not None
    assert result.compaction_block["content"] == "Router summary"


async def test_litellm_metadata_propagated_to_summary_call():
    """Auth fields from the proxy ``litellm_metadata`` are forwarded to the summary call."""
    messages = _simple_messages()
    mock_response = _make_mock_response("<summary>Summary</summary>")
    parent_litellm_metadata = {
        "user_api_key": "sk-test",
        "user_api_key_team_id": "team-123",
        "user_api_key_user_id": "user-456",
        "litellm_call_id": "call-789",
        "should_not_propagate": "secret",
    }

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_call,
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            litellm_metadata=parent_litellm_metadata,
        )

    call_kwargs = mock_call.call_args.kwargs
    propagated = call_kwargs["metadata"]
    assert propagated["user_api_key"] == "sk-test"
    assert propagated["user_api_key_team_id"] == "team-123"
    assert "should_not_propagate" not in propagated


# ---------------------------------------------------------------------------
# Editor: error paths
# ---------------------------------------------------------------------------


async def test_summary_call_failed():
    """When the summary model raises, applied_edits[0].error == 'summary_call_failed'."""
    messages = _simple_messages()

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            new_callable=AsyncMock,
            side_effect=RuntimeError("network error"),
        ),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    assert result.applied_edits[0]["error"] == "summary_call_failed"
    assert result.compaction_block is None
    assert result.iterations_usage is None
    # Messages passed through (at minimum sliced, no compaction blocks)
    for msg in result.messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                assert block.get("type") != "compaction"


async def test_summary_extraction_failed_no_tags():
    """When summary response has no <summary> tags, applied_edits[0].error == 'summary_extraction_failed'."""
    messages = _simple_messages()
    mock_response = _make_mock_response("I cannot summarize that.")

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            new_callable=AsyncMock,
            return_value=mock_response,
        ),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    assert result.applied_edits[0]["error"] == "summary_extraction_failed"
    assert result.compaction_block is None
    assert result.iterations_usage is None


# ---------------------------------------------------------------------------
# Editor: warnings
# ---------------------------------------------------------------------------


async def test_pause_after_compaction_ignored_warning():
    """pause_after_compaction: true → warning recorded, request proceeds normally."""
    messages = _simple_messages()
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
        return_value=None,
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec={
                "type": "compact_20260112",
                "pause_after_compaction": True,
            },
        )

    edit = result.applied_edits[0]
    assert "pause_after_compaction_ignored" in (edit.get("warnings") or [])
    # Request still proceeds (here it hits opt-in gate because no model configured)
    assert edit.get("error") == "summary_model_not_configured"


async def test_unsupported_trigger_type_falls_back_to_default():
    messages = _simple_messages()
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
        return_value=None,
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec={
                "type": "compact_20260112",
                "trigger": {"type": "output_tokens", "value": 200_000},
            },
        )

    edit = result.applied_edits[0]
    warnings = edit.get("warnings") or []
    assert any("unsupported_trigger_type" in w for w in warnings)


# ---------------------------------------------------------------------------
# Editor: custom instructions
# ---------------------------------------------------------------------------


async def test_custom_instructions_used_verbatim():
    """Custom instructions are used as-is; the default prompt is NOT appended."""
    messages = _simple_messages()
    tools = [{"name": "search", "description": "Search tool"}]
    mock_response = _make_mock_response("<summary>Custom summary</summary>")

    captured_calls: list = []

    async def _fake_call_summary_model(**kwargs):
        captured_calls.append(kwargs)
        return mock_response

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            side_effect=_fake_call_summary_model,
        ),
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=tools,
            system=None,
            edit_spec={
                "type": "compact_20260112",
                "instructions": "Summarize everything briefly.",
            },
        )

    assert len(captured_calls) == 1
    summary_messages = captured_calls[0]["summary_messages"]
    # The custom instruction prompt is appended to the trailing user turn so
    # we don't end up with two consecutive ``role=user`` messages (some
    # providers reject that).
    last_msg = summary_messages[-1]
    assert last_msg["role"] == "user"
    assert "Summarize everything briefly." in last_msg["content"]
    # The "do not call tools" suffix should NOT be in the prompt since custom was set
    assert "do not call" not in last_msg["content"].lower()


async def test_default_instructions_appended_with_no_tool_suffix_when_no_tools():
    """Without tools, default prompt is used but the no-tool-calls suffix is absent."""
    messages = _simple_messages()
    mock_response = _make_mock_response("<summary>Default summary</summary>")

    captured_calls: list = []

    async def _fake_call_summary_model(**kwargs):
        captured_calls.append(kwargs)
        return mock_response

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            side_effect=_fake_call_summary_model,
        ),
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    prompt = captured_calls[0]["summary_messages"][-1]["content"]
    # Should not contain the no-tool-calls guidance
    assert "do not call" not in prompt.lower()


async def test_default_instructions_with_tools_appends_no_tool_suffix():
    """With tools and no custom instructions, the no-tool-calls suffix is appended."""
    messages = _simple_messages()
    tools = [{"name": "search"}]
    mock_response = _make_mock_response("<summary>Tool-aware summary</summary>")

    captured_calls: list = []

    async def _fake_call_summary_model(**kwargs):
        captured_calls.append(kwargs)
        return mock_response

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            side_effect=_fake_call_summary_model,
        ),
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=tools,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    prompt = captured_calls[0]["summary_messages"][-1]["content"]
    assert "tool" in prompt.lower()


async def test_system_prompt_forwarded_to_summary_call_as_string():
    """A bare-string ``system`` is prepended as a system message to the summary call."""
    messages = _simple_messages()
    mock_response = _make_mock_response("<summary>With system</summary>")

    captured_calls: list = []

    async def _fake_call_summary_model(**kwargs):
        captured_calls.append(kwargs)
        return mock_response

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            side_effect=_fake_call_summary_model,
        ),
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system="You are a helpful coding agent. The initial task is to fix bug X.",
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    summary_messages = captured_calls[0]["summary_messages"]
    assert summary_messages[0]["role"] == "system"
    assert "initial task is to fix bug X" in summary_messages[0]["content"]


async def test_system_prompt_forwarded_to_summary_call_as_content_blocks():
    """An Anthropic-shaped list ``system`` is flattened to text and prepended."""
    messages = _simple_messages()
    mock_response = _make_mock_response("<summary>With list system</summary>")

    captured_calls: list = []

    async def _fake_call_summary_model(**kwargs):
        captured_calls.append(kwargs)
        return mock_response

    system_blocks = [
        {"type": "text", "text": "Agent role: code reviewer."},
        {"type": "text", "text": "Initial task: review PR #123."},
    ]

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            side_effect=_fake_call_summary_model,
        ),
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=system_blocks,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    summary_messages = captured_calls[0]["summary_messages"]
    assert summary_messages[0]["role"] == "system"
    content = summary_messages[0]["content"]
    assert "Agent role: code reviewer." in content
    assert "Initial task: review PR #123." in content


async def test_summary_call_carries_prior_compaction_summary_into_system():
    """Multi-round: when a prior compaction block is present, the summary
    model receives the augmented system (with ``Previous conversation
    summary: <prior>``) so it can produce a comprehensive summary that
    incorporates both the prior round's context and the current slice.
    Without this, multi-round compaction would silently drop accumulated
    history each time the polyfill fires.
    """
    messages = _messages_with_compaction(summary="ROUND_ONE_SUMMARY_TEXT")
    mock_response = _make_mock_response("<summary>Round two</summary>")

    captured_calls: list = []

    async def _fake_call_summary_model(**kwargs):
        captured_calls.append(kwargs)
        return mock_response

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            side_effect=_fake_call_summary_model,
        ),
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system="Original agent role.",
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    summary_messages = captured_calls[0]["summary_messages"]
    assert summary_messages[0]["role"] == "system"
    system_content = summary_messages[0]["content"]
    assert "ROUND_ONE_SUMMARY_TEXT" in system_content
    assert "Original agent role." in system_content


async def test_summary_call_omits_system_message_when_system_is_none():
    """No system message is prepended when the caller did not provide one."""
    messages = _simple_messages()
    mock_response = _make_mock_response("<summary>No system</summary>")

    captured_calls: list = []

    async def _fake_call_summary_model(**kwargs):
        captured_calls.append(kwargs)
        return mock_response

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            side_effect=_fake_call_summary_model,
        ),
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    summary_messages = captured_calls[0]["summary_messages"]
    assert all(msg.get("role") != "system" for msg in summary_messages)


async def test_summary_call_does_not_emit_consecutive_user_turns():
    """When the trailing message is already a user turn, the summarization
    prompt is merged into it instead of appended as a second user message.

    Some providers (and strict OpenAI-compatible endpoints) reject two
    consecutive ``role=user`` messages, which would silently fall into the
    ``summary_call_failed`` error path.
    """
    messages = _simple_messages()
    assert messages[-1]["role"] == "user"
    mock_response = _make_mock_response("<summary>x</summary>")

    captured_calls: list = []

    async def _fake_call_summary_model(**kwargs):
        captured_calls.append(kwargs)
        return mock_response

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            side_effect=_fake_call_summary_model,
        ),
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    summary_messages = captured_calls[0]["summary_messages"]
    user_indices = [
        idx for idx, msg in enumerate(summary_messages) if msg.get("role") == "user"
    ]
    # No two adjacent indices.
    assert all(
        b - a > 1 for a, b in zip(user_indices, user_indices[1:])
    ), f"two consecutive user turns produced: {summary_messages}"


async def test_summary_call_sends_default_max_tokens():
    """``max_tokens`` is set on the summary call so providers like Anthropic
    (which require it) don't reject the request and silently fall back to
    ``summary_call_failed``.
    """
    from litellm.llms.anthropic.experimental_pass_through.context_management.constants import (
        COMPACT_SUMMARY_MAX_TOKENS,
    )
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        _call_summary_model,
    )

    captured_kwargs: dict = {}

    class _FakeRouter:
        async def acompletion(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_mock_response("<summary>x</summary>")

    await _call_summary_model(
        summary_model="claude-haiku-4-5",
        summary_messages=[{"role": "user", "content": "hi"}],
        metadata={},
        llm_router=_FakeRouter(),
    )

    assert captured_kwargs.get("max_tokens") == COMPACT_SUMMARY_MAX_TOKENS


async def test_summary_call_honors_max_tokens_override():
    """Operators can override the default summary ``max_tokens`` via
    ``general_settings.context_management_summary_max_tokens``."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        _read_summary_max_tokens_setting,
    )

    captured_kwargs: dict = {}

    class _FakeRouter:
        async def acompletion(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_mock_response("<summary>x</summary>")

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"context_management_summary_max_tokens": 8192},
    ):
        assert _read_summary_max_tokens_setting() == 8192

        from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
            _call_summary_model,
        )

        await _call_summary_model(
            summary_model="claude-haiku-4-5",
            summary_messages=[{"role": "user", "content": "hi"}],
            metadata={},
            llm_router=_FakeRouter(),
            max_tokens=_read_summary_max_tokens_setting(),
        )

    assert captured_kwargs.get("max_tokens") == 8192


def test_summary_max_tokens_setting_falls_back_for_invalid_values():
    """Invalid override values (non-int, non-positive, missing) fall back to
    the compiled default so a typo in ``general_settings`` doesn't break the
    summary call."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.constants import (
        COMPACT_SUMMARY_MAX_TOKENS,
    )
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        _read_summary_max_tokens_setting,
    )

    for bad in ("4096", 0, -1, None, {"value": 1024}):
        with patch(
            "litellm.proxy.proxy_server.general_settings",
            {"context_management_summary_max_tokens": bad},
        ):
            assert (
                _read_summary_max_tokens_setting() == COMPACT_SUMMARY_MAX_TOKENS
            ), f"expected default for invalid override {bad!r}"


async def test_summary_call_sends_default_timeout():
    """``timeout`` is set on the summary call so a slow or unresponsive summary
    model cannot hang the parent ``/v1/messages`` request indefinitely."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.constants import (
        COMPACT_SUMMARY_TIMEOUT_SECONDS,
    )
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        _call_summary_model,
    )

    captured_kwargs: dict = {}

    class _FakeRouter:
        async def acompletion(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_mock_response("<summary>x</summary>")

    await _call_summary_model(
        summary_model="claude-haiku-4-5",
        summary_messages=[{"role": "user", "content": "hi"}],
        metadata={},
        llm_router=_FakeRouter(),
    )

    assert captured_kwargs.get("timeout") == COMPACT_SUMMARY_TIMEOUT_SECONDS


# ---------------------------------------------------------------------------
# Editor: summary model key/team access gate
# ---------------------------------------------------------------------------


def _fake_user_api_key_auth(
    *,
    key_models=None,
    team_models=None,
    team_id=None,
    model_max_budget=None,
    end_user_model_max_budget=None,
    end_user_id=None,
    token=None,
):
    """Build a minimal stand-in for ``UserAPIKeyAuth`` with just the fields
    consulted by ``_check_summary_model_access`` and
    ``_check_summary_model_budget``. Avoids pulling the proxy deps into this
    unit test."""

    class _Auth:
        pass

    auth = _Auth()
    auth.models = list(key_models) if key_models is not None else []
    auth.team_models = list(team_models) if team_models is not None else []
    auth.team_id = team_id
    auth.team_model_aliases = None
    auth.model_max_budget = model_max_budget
    auth.end_user_model_max_budget = end_user_model_max_budget
    auth.end_user_id = end_user_id
    auth.token = token
    return auth


async def test_summary_model_denied_when_key_not_in_allowlist():
    """Caller key restricted to specific models cannot trigger an unauthorized summary model."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>x</summary>"))

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=_fake_user_api_key_auth(key_models=["gpt-4o"]),
        )

    mock_call.assert_not_awaited()
    assert result.compaction_block is None
    assert result.iterations_usage is None
    assert result.applied_edits[0]["type"] == "compact_20260112"
    assert result.applied_edits[0].get("error") == "summary_model_access_denied"


async def test_summary_model_denied_when_team_not_in_allowlist():
    """Team-level model allowlist is enforced even if the key allows all models."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>x</summary>"))

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=_fake_user_api_key_auth(
                key_models=["all-proxy-models"], team_models=["gpt-4o"]
            ),
        )

    mock_call.assert_not_awaited()
    assert result.applied_edits[0].get("error") == "summary_model_access_denied"


async def test_summary_model_allowed_when_in_key_allowlist():
    """Caller key that explicitly allows the summary model is permitted to use it."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>ok</summary>"))

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=_fake_user_api_key_auth(
                key_models=["claude-haiku-4-5", "gpt-4o"]
            ),
        )

    mock_call.assert_awaited_once()
    assert result.compaction_block is not None
    assert result.compaction_block["content"] == "ok"
    assert not result.applied_edits[0].get("error")


async def test_summary_model_allowed_when_no_user_api_key_auth():
    """SDK callers (no proxy auth object) are not gated."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>ok</summary>"))

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
        )

    mock_call.assert_awaited_once()
    assert result.compaction_block is not None


async def test_summary_model_denied_when_user_scope_excludes_it():
    """Personal user allowed-models scope denies the summary model even when
    key/team allowlists permit it."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>x</summary>"))

    auth = _fake_user_api_key_auth(key_models=["all-proxy-models"])
    auth.user_id = "user-123"

    class _User:
        user_id = "user-123"
        models = ["gpt-3.5-turbo"]
        organization_memberships = []

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_user_object",
            AsyncMock(return_value=_User()),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_project_object",
            AsyncMock(return_value=None),
        ),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_not_awaited()
    assert result.applied_edits[0].get("error") == "summary_model_access_denied"


async def test_summary_model_denied_when_project_scope_excludes_it():
    """Project allowed-models scope denies the summary model even when
    key/team allowlists permit it."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>x</summary>"))

    auth = _fake_user_api_key_auth(key_models=["all-proxy-models"])
    auth.project_id = "project-1"

    class _Project:
        project_id = "project-1"
        models = ["gpt-3.5-turbo"]

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_user_object",
            AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_project_object",
            AsyncMock(return_value=_Project()),
        ),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_not_awaited()
    assert result.applied_edits[0].get("error") == "summary_model_access_denied"


async def test_summary_model_denied_when_team_member_scope_excludes_it():
    """Per-team-member allowed-models scope denies the summary model even
    when key/team allowlists permit it."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>x</summary>"))

    auth = _fake_user_api_key_auth(key_models=["all-proxy-models"], team_id="team-1")
    auth.user_id = "user-123"

    class _Budget:
        allowed_models = ["gpt-3.5-turbo"]

    class _Membership:
        litellm_budget_table = _Budget()

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_user_object",
            AsyncMock(return_value=None),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_team_membership",
            AsyncMock(return_value=_Membership()),
        ),
        patch(
            "litellm.proxy.auth.auth_checks.get_project_object",
            AsyncMock(return_value=None),
        ),
        patch("litellm.proxy.proxy_server.prisma_client", MagicMock()),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_not_awaited()
    assert result.applied_edits[0].get("error") == "summary_model_access_denied"


async def test_summary_model_denied_when_key_over_model_budget():
    """A caller whose per-model budget for the summary model is exhausted cannot
    trigger the summary call via compaction."""
    import litellm

    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>x</summary>"))

    auth = _fake_user_api_key_auth(
        key_models=["all-proxy-models"],
        model_max_budget={"claude-haiku-4-5": {"budget_limit": 5}},
        token="hashed-token",
    )

    limiter = MagicMock()
    limiter.is_key_within_model_budget = AsyncMock(
        side_effect=litellm.BudgetExceededError(
            message="over budget", current_cost=10, max_budget=5
        )
    )

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_not_awaited()
    limiter.is_key_within_model_budget.assert_awaited_once()
    assert result.applied_edits[0].get("error") == "summary_model_budget_exceeded"


async def test_summary_model_denied_when_end_user_over_model_budget():
    """End-user per-model budget is enforced for the summary subrequest too."""
    import litellm

    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>x</summary>"))

    auth = _fake_user_api_key_auth(
        key_models=["all-proxy-models"],
        end_user_model_max_budget={"claude-haiku-4-5": {"budget_limit": 5}},
        end_user_id="end-user-1",
        token="hashed-token",
    )

    limiter = MagicMock()
    limiter.is_key_within_model_budget = AsyncMock(return_value=True)
    limiter.is_end_user_within_model_budget = AsyncMock(
        side_effect=litellm.BudgetExceededError(
            message="over budget", current_cost=10, max_budget=5
        )
    )

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_not_awaited()
    limiter.is_end_user_within_model_budget.assert_awaited_once()
    assert result.applied_edits[0].get("error") == "summary_model_budget_exceeded"


async def test_summary_model_allowed_when_within_model_budget():
    """When the per-model budget check passes, the summary call proceeds."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>ok</summary>"))

    auth = _fake_user_api_key_auth(
        key_models=["all-proxy-models"],
        model_max_budget={"claude-haiku-4-5": {"budget_limit": 5}},
        token="hashed-token",
    )

    limiter = MagicMock()
    limiter.is_key_within_model_budget = AsyncMock(return_value=True)
    limiter.is_end_user_within_model_budget = AsyncMock(return_value=True)

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
        patch("litellm.proxy.proxy_server.model_max_budget_limiter", limiter),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_awaited_once()
    limiter.is_key_within_model_budget.assert_awaited_once()
    assert not result.applied_edits[0].get("error")


class _FakeRateLimiter:
    """Minimal stand-in for ``_PROXY_MaxParallelRequestsHandler_v3`` exposing
    just the descriptor-build + read-only check surface the editor consults."""

    def __init__(self, overall_code: str):
        self._overall_code = overall_code
        self.read_only_checked = False

    def _create_rate_limit_descriptors(self, **kwargs):
        return [
            {
                "key": "api_key",
                "value": "hashed-token",
                "rate_limit": {"requests_per_unit": 10},
            }
        ]

    def _add_team_model_rate_limit_descriptor_from_metadata(self, **kwargs):
        return None

    def _add_project_model_rate_limit_descriptor_from_metadata(self, **kwargs):
        return None

    def create_organization_rate_limit_descriptor(self, *args, **kwargs):
        return []

    async def should_rate_limit(self, **kwargs):
        self.read_only_checked = kwargs.get("read_only") is True
        return {"overall_code": self._overall_code}


async def test_summary_model_denied_when_over_rate_limit():
    """A caller already at their configured RPM/TPM for the summary model cannot
    drive an extra summary completion via compaction."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>x</summary>"))

    auth = _fake_user_api_key_auth(key_models=["all-proxy-models"])
    limiter = _FakeRateLimiter("OVER_LIMIT")
    proxy_logging = MagicMock()
    proxy_logging.max_parallel_request_limiter = limiter

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_not_awaited()
    assert limiter.read_only_checked is True
    assert result.compaction_block is None
    assert result.applied_edits[0].get("error") == "summary_model_rate_limit_exceeded"


async def test_summary_model_allowed_when_within_rate_limit():
    """When the read-only rate-limit check is under limit, the summary call proceeds."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>ok</summary>"))

    auth = _fake_user_api_key_auth(key_models=["all-proxy-models"])
    limiter = _FakeRateLimiter("OK")
    proxy_logging = MagicMock()
    proxy_logging.max_parallel_request_limiter = limiter

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_awaited_once()
    assert limiter.read_only_checked is True
    assert result.compaction_block is not None
    assert not result.applied_edits[0].get("error")


async def test_summary_model_rate_limit_skipped_for_legacy_limiter():
    """A limiter without the v3 read-only check surface fails open so the summary
    call still proceeds (its usage is still charged post-call)."""
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>ok</summary>"))

    auth = _fake_user_api_key_auth(key_models=["all-proxy-models"])

    class _LegacyLimiter:
        async def async_pre_call_hook(self, **kwargs):
            return None

    proxy_logging = MagicMock()
    proxy_logging.max_parallel_request_limiter = _LegacyLimiter()

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging),
    ):
        result = await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_awaited_once()
    assert result.compaction_block is not None
    assert not result.applied_edits[0].get("error")


async def test_scoped_budget_metadata_propagated_to_summary_call():
    """The end-user/project scope identifiers and the end-user budget the post-call
    spend and rate-limit hooks key on are forwarded to the summary subrequest, and
    the end-user id is also passed as the top-level ``user`` kwarg the legacy
    limiter hooks read, so the summary tokens debit those scoped budgets/counters."""
    messages = _simple_messages()
    mock_response = _make_mock_response("<summary>Summary</summary>")
    parent_litellm_metadata = {
        "user_api_key": "sk-test",
        "user_api_key_end_user_id": "customer-1",
        "user_api_end_user_max_budget": 10,
        "user_api_key_project_id": "project-9",
    }

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_call,
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            litellm_metadata=parent_litellm_metadata,
        )

    propagated = mock_call.call_args.kwargs["metadata"]
    assert propagated["user_api_key_end_user_id"] == "customer-1"
    assert propagated["user_api_end_user_max_budget"] == 10
    assert propagated["user_api_key_project_id"] == "project-9"


async def test_summary_call_passes_end_user_id_as_top_level_user():
    """``_call_summary_model`` forwards the propagated end-user id as the top-level
    ``user`` kwarg that legacy limiter / prometheus end-user tracking reads."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        _call_summary_model,
    )

    captured_kwargs: dict = {}

    class _FakeRouter:
        async def acompletion(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_mock_response("<summary>x</summary>")

    await _call_summary_model(
        summary_model="claude-haiku-4-5",
        summary_messages=[{"role": "user", "content": "hi"}],
        metadata={"user_api_key_end_user_id": "customer-1"},
        llm_router=_FakeRouter(),
    )

    assert captured_kwargs.get("user") == "customer-1"


async def test_summary_call_omits_user_when_no_end_user_id():
    """No end-user id on the parent request means no ``user`` kwarg is sent."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        _call_summary_model,
    )

    captured_kwargs: dict = {}

    class _FakeRouter:
        async def acompletion(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_mock_response("<summary>x</summary>")

    await _call_summary_model(
        summary_model="claude-haiku-4-5",
        summary_messages=[{"role": "user", "content": "hi"}],
        metadata={},
        llm_router=_FakeRouter(),
    )

    assert "user" not in captured_kwargs


async def test_model_budget_metadata_propagated_to_summary_call():
    """The per-model budget metadata the spend caches rely on is forwarded to the
    summary subrequest so its spend counts against the caller's model budget."""
    messages = _simple_messages()
    mock_response = _make_mock_response("<summary>Summary</summary>")
    parent_litellm_metadata = {
        "user_api_key": "sk-test",
        "user_api_key_model_max_budget": {"claude-haiku-4-5": {"budget_limit": 5}},
        "user_api_key_end_user_model_max_budget": {
            "claude-haiku-4-5": {"budget_limit": 2}
        },
    }

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_call,
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            litellm_metadata=parent_litellm_metadata,
        )

    propagated = mock_call.call_args.kwargs["metadata"]
    assert propagated["user_api_key_model_max_budget"] == {
        "claude-haiku-4-5": {"budget_limit": 5}
    }
    assert propagated["user_api_key_end_user_model_max_budget"] == {
        "claude-haiku-4-5": {"budget_limit": 2}
    }


async def test_summary_call_propagates_allowed_model_region():
    """``allowed_model_region`` from ``user_api_key_auth`` is propagated to the
    summary subrequest as a top-level kwarg so the router applies the same
    region restriction the parent request would.
    """
    messages = _simple_messages()
    mock_call = AsyncMock(return_value=_make_mock_response("<summary>ok</summary>"))

    auth = _fake_user_api_key_auth(key_models=["all-proxy-models"])
    auth.allowed_model_region = "eu"

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._call_summary_model",
            mock_call,
        ),
    ):
        await apply_compact_20260112(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            edit_spec=_EDIT_SPEC_DEFAULT,
            user_api_key_auth=auth,
        )

    mock_call.assert_awaited_once()
    assert mock_call.await_args.kwargs.get("allowed_model_region") == "eu"


async def test_summary_call_omits_allowed_model_region_when_unset():
    """Callers without a region restriction must not get an ``allowed_model_region=None``
    kwarg, which would otherwise force the router to evaluate region filtering.
    """
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        _call_summary_model,
    )

    captured_kwargs: dict = {}

    class _FakeRouter:
        async def acompletion(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_mock_response("<summary>x</summary>")

    await _call_summary_model(
        summary_model="claude-haiku-4-5",
        summary_messages=[{"role": "user", "content": "hi"}],
        metadata={},
        llm_router=_FakeRouter(),
    )

    assert "allowed_model_region" not in captured_kwargs


async def test_summary_call_forwards_allowed_model_region_when_set():
    """When the caller is region-restricted, the kwarg reaches the router."""
    from litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact import (
        _call_summary_model,
    )

    captured_kwargs: dict = {}

    class _FakeRouter:
        async def acompletion(self, **kwargs):
            captured_kwargs.update(kwargs)
            return _make_mock_response("<summary>x</summary>")

    await _call_summary_model(
        summary_model="claude-haiku-4-5",
        summary_messages=[{"role": "user", "content": "hi"}],
        metadata={},
        llm_router=_FakeRouter(),
        allowed_model_region="eu",
    )

    assert captured_kwargs.get("allowed_model_region") == "eu"


# ---------------------------------------------------------------------------
# Dispatcher integration: compact_20260112 via apply_context_management
# ---------------------------------------------------------------------------


async def test_dispatcher_routes_compact_edit():
    """compact_20260112 in the dispatcher resolves to opt-in gate when no model set."""
    messages = _simple_messages()
    with patch(
        "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
        return_value=None,
    ):
        result = await apply_context_management(
            model=MODEL,
            messages=messages,
            tools=None,
            system=None,
            context_management_spec={"edits": [{"type": "compact_20260112"}]},
        )

    assert len(result.applied_edits) == 1
    assert result.applied_edits[0]["type"] == "compact_20260112"
    assert result.applied_edits[0].get("error") == "summary_model_not_configured"


async def test_dispatcher_trigger_below_minimum_raises_through():
    """AnthropicContextManagementError from the editor bubbles up through the dispatcher."""
    with pytest.raises(AnthropicContextManagementError):
        await apply_context_management(
            model=MODEL,
            messages=_simple_messages(),
            tools=None,
            system=None,
            context_management_spec={
                "edits": [
                    {
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": 1_000},
                    }
                ]
            },
        )


# ---------------------------------------------------------------------------
# _run_polyfill_if_enabled: drop_params gate
# ---------------------------------------------------------------------------


async def test_run_polyfill_skipped_when_drop_params_true():
    """When drop_params=True the polyfill must be skipped (returns None)."""
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        _run_polyfill_if_enabled,
    )

    result = await _run_polyfill_if_enabled(
        model=MODEL,
        messages=_simple_messages(),
        tools=None,
        system=None,
        context_management_spec={"edits": [{"type": "compact_20260112"}]},
        litellm_metadata={},
        drop_params=True,
        llm_router=None,
    )
    assert result is None


async def test_run_polyfill_skipped_when_spec_empty():
    """Empty context_management_spec must also return None (no polyfill work)."""
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        _run_polyfill_if_enabled,
    )

    result = await _run_polyfill_if_enabled(
        model=MODEL,
        messages=_simple_messages(),
        tools=None,
        system=None,
        context_management_spec=None,
        litellm_metadata={},
        drop_params=False,
        llm_router=None,
    )
    assert result is None


async def test_prepare_context_managed_request_forwards_proxy_litellm_metadata():
    """The handler must hand the polyfill the proxy ``litellm_metadata`` (which
    carries ``user_api_key`` / ``user_api_key_team_id`` / ...), not the
    Anthropic-shape ``metadata`` arg (which only carries ``user_id``). Otherwise
    the summary subcall lands on the router with no parent attribution, and
    those tokens go unbilled to the caller's key/team."""
    from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
        _prepare_context_managed_request,
    )

    captured_summary_metadata: Dict[str, Any] = {}

    class _RouterStub:
        async def acompletion(self, **kwargs):
            captured_summary_metadata.update(kwargs.get("litellm_metadata", {}))
            return _make_mock_response("<summary>s</summary>")

    with (
        patch(
            "litellm.llms.anthropic.experimental_pass_through.context_management.editors.compact._read_summary_model_setting",
            return_value="claude-haiku-4-5",
        ),
        patch("litellm.token_counter", return_value=200_000),
    ):
        result = await _prepare_context_managed_request(
            model=MODEL,
            messages=_simple_messages(),
            tools=None,
            system=None,
            context_management_spec={"edits": [_EDIT_SPEC_DEFAULT]},
            litellm_metadata={
                "user_api_key": "sk-parent",
                "user_api_key_team_id": "team-abc",
                "user_api_key_user_id": "user-xyz",
                "litellm_call_id": "call-1",
            },
            drop_params=False,
            llm_router=_RouterStub(),
        )

    assert result is not None
    assert captured_summary_metadata.get("user_api_key") == "sk-parent"
    assert captured_summary_metadata.get("user_api_key_team_id") == "team-abc"
    assert captured_summary_metadata.get("user_api_key_user_id") == "user-xyz"
    assert captured_summary_metadata.get("litellm_call_id") == "call-1"
    # Anthropic-shape ``metadata.user_id`` must not leak in as a propagated field.
    assert "user_id" not in captured_summary_metadata


# ---------------------------------------------------------------------------
# Endpoint error format: AnthropicContextManagementError → Anthropic 400 body
# ---------------------------------------------------------------------------


def test_anthropic_context_management_error_format():
    """AnthropicContextManagementError must produce an Anthropic-format body via
    AnthropicExceptionMapping.transform_to_anthropic_error — the same path the
    /v1/messages endpoint takes when it catches this exception."""
    from litellm.anthropic_interface.exceptions import AnthropicExceptionMapping

    body = AnthropicExceptionMapping.transform_to_anthropic_error(
        status_code=400,
        raw_message="trigger.value must be at least 50000 tokens",
        request_id=None,
    )

    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "50000" in body["error"]["message"]


def test_anthropic_context_management_error_attrs():
    """AnthropicContextManagementError carries status_code and message correctly."""
    err = AnthropicContextManagementError(
        status_code=400,
        message="trigger.value must be at least 50000 tokens",
    )

    assert err.status_code == 400
    assert "50000" in err.message


# ---------------------------------------------------------------------------
# Endpoint integration: /v1/messages → Anthropic 400 on context management error
# ---------------------------------------------------------------------------


def test_endpoint_returns_anthropic_400_on_context_management_error():
    """The /v1/messages endpoint must catch AnthropicContextManagementError and
    return an Anthropic-format 400 JSONResponse — not a 500 ProxyException."""
    import sys
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy.anthropic_endpoints.endpoints import router
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    # Stub proxy_server to avoid apscheduler/heavy proxy deps imported lazily
    # inside the route handler at request time.
    mock_proxy_server = MagicMock()
    mock_proxy_server.general_settings = {}
    mock_proxy_server.llm_router = None
    mock_proxy_server.proxy_config = MagicMock()
    mock_proxy_server.proxy_logging_obj = MagicMock()
    mock_proxy_server.user_api_base = None
    mock_proxy_server.user_max_tokens = None
    mock_proxy_server.user_model = None
    mock_proxy_server.user_request_timeout = None
    mock_proxy_server.user_temperature = None
    mock_proxy_server.version = "test"

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        with patch(
            "litellm.proxy.anthropic_endpoints.endpoints.ProxyBaseLLMRequestProcessing"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.base_process_llm_request = AsyncMock(
                side_effect=AnthropicContextManagementError(
                    status_code=400,
                    message="trigger.value must be at least 50000 tokens",
                )
            )
            mock_cls.return_value = mock_instance

            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[user_api_key_auth] = lambda: MagicMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/v1/messages",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer test-key"},
            )

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert "50000" in body["error"]["message"]


def test_endpoint_runs_failure_hook_on_500_context_management_error():
    """A 500-level AnthropicContextManagementError (internal polyfill failure)
    must invoke post_call_failure_hook for spend/alerting parity, while still
    returning the Anthropic-format error body."""
    import sys
    from unittest.mock import AsyncMock, MagicMock, patch

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy.anthropic_endpoints.endpoints import router
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

    failure_hook = AsyncMock()
    mock_proxy_server = MagicMock()
    mock_proxy_server.general_settings = {}
    mock_proxy_server.llm_router = None
    mock_proxy_server.proxy_config = MagicMock()
    mock_proxy_server.proxy_logging_obj = MagicMock()
    mock_proxy_server.proxy_logging_obj.post_call_failure_hook = failure_hook
    mock_proxy_server.user_api_base = None
    mock_proxy_server.user_max_tokens = None
    mock_proxy_server.user_model = None
    mock_proxy_server.user_request_timeout = None
    mock_proxy_server.user_temperature = None
    mock_proxy_server.version = "test"

    with patch.dict(sys.modules, {"litellm.proxy.proxy_server": mock_proxy_server}):
        with patch(
            "litellm.proxy.anthropic_endpoints.endpoints.ProxyBaseLLMRequestProcessing"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.base_process_llm_request = AsyncMock(
                side_effect=AnthropicContextManagementError(
                    status_code=500,
                    message="context_management polyfill failed: boom",
                )
            )
            mock_cls.return_value = mock_instance

            app = FastAPI()
            app.include_router(router)
            app.dependency_overrides[user_api_key_auth] = lambda: MagicMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/v1/messages",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer test-key"},
            )

    assert response.status_code == 500
    body = response.json()
    assert body["type"] == "error"
    failure_hook.assert_awaited_once()
