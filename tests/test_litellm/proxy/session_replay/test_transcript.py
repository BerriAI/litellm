from litellm.proxy.session_replay.transcript import (
    RecordedBody,
    attach_turn,
    build_transcript,
    tool_use_ids,
)

RECORDED_BODY = {
    "model": "tin-claude-router",
    "max_tokens": 32000,
    "thinking": {"type": "adaptive"},
    "stream": True,
    "litellm_session_id": "sess-1",
    "litellm_trace_id": "sess-1",
    "headers": {"x-claude-code-session-id": "sess-1"},
    "litellm_metadata": {"user_api_key_hash": "secret-hash"},
    "metadata": {"user_id": "device-blob"},
    "provider_specific_header": {"extra_headers": {"anthropic-beta": "claude-code"}},
    "system": [{"type": "text", "text": "You are Claude Code"}],
    "tools": [
        {"name": "Read", "description": "Read a file", "input_schema": {"type": "object", "properties": {}}},
        {
            "name": "Edit",
            "description": "Edit a file...[litellm_truncated]...tail",
            "input_schema": {"type": "object", "properties": {}},
        },
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "<system-reminder>\ntoday is tuesday\n</system-reminder>"},
                {"type": "text", "text": "Plan my week"},
            ],
        },
        {"role": "system", "content": "Available agent types for the Agent tool"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "on it"},
                {"type": "tool_use", "id": "toolu_ORIGINAL", "name": "Read"},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_ORIGINAL", "content": "file body"}]},
    ],
}


def _transcript():
    return build_transcript(RecordedBody.model_validate(RECORDED_BODY))


def test_system_role_message_is_hoisted_out_of_messages():
    """The Anthropic Messages API rejects role=system inside messages with a 400, and the
    recorded body carries one, so leaving it in place makes every replay unreplayable."""
    transcript = _transcript()

    assert all(turn.role == "user" for turn in transcript.user_turns)
    assert "Available agent types for the Agent tool" in "".join(block.text or "" for block in transcript.system)


def test_human_asks_strip_system_reminders_and_exclude_tool_results():
    transcript = _transcript()

    assert transcript.human_asks == ("Plan my week",)


def test_sampling_params_keep_provider_knobs_and_drop_proxy_injected_keys():
    transcript = _transcript()
    keys = {key for key, _ in transcript.sampling_params}

    assert {"max_tokens", "thinking"} <= keys
    assert keys.isdisjoint(
        {
            "litellm_session_id",
            "litellm_trace_id",
            "headers",
            "litellm_metadata",
            "metadata",
            "provider_specific_header",
        }
    )
    assert keys.isdisjoint({"model", "messages", "system", "tools", "stream"})


def test_truncation_markers_are_counted_so_fidelity_loss_is_visible():
    assert _transcript().truncated_strings == 1


def test_unattachable_tool_result_turn_is_dropped_not_degraded():
    """An arm that called no tools has no tool_use block for a recorded tool_result to
    reference, and injecting the recording's tool output as prose would hand the arm
    answers to questions it never asked."""
    transcript = _transcript()
    tool_turn = next(turn for turn in transcript.user_turns if "tool_result" in turn.block_types())

    assert attach_turn(tool_turn, ()) is None


def test_tool_results_rebind_to_the_arms_own_tool_use_ids():
    transcript = _transcript()
    tool_turn = next(turn for turn in transcript.user_turns if "tool_result" in turn.block_types())

    attached = attach_turn(tool_turn, ("toolu_ARM",))

    assert attached is not None
    assert [block.tool_use_id for block in attached.blocks()] == ["toolu_ARM"]


def test_tool_calls_the_recording_never_answered_get_stubbed():
    """Anthropic rejects an assistant tool_use with no matching tool_result, so an arm that
    called more tools than the recording answered would wedge on the next turn."""
    transcript = _transcript()
    tool_turn = next(turn for turn in transcript.user_turns if "tool_result" in turn.block_types())

    attached = attach_turn(tool_turn, ("toolu_ARM", "toolu_EXTRA"))

    assert attached is not None
    assert [block.tool_use_id for block in attached.blocks()] == ["toolu_ARM", "toolu_EXTRA"]


def test_plain_user_turn_passes_through_untouched():
    transcript = _transcript()
    plain = next(turn for turn in transcript.user_turns if "tool_result" not in turn.block_types())

    assert attach_turn(plain, ("toolu_ARM",)) is plain


def test_tool_use_ids_reads_only_tool_use_blocks():
    transcript = _transcript()
    assistant_blocks = RecordedBody.model_validate(RECORDED_BODY).messages[2].blocks()

    assert tool_use_ids(assistant_blocks) == ("toolu_ORIGINAL",)
    assert tool_use_ids(transcript.system) == ()
