from litellm.proxy.session_replay.transcript import (
    RecordedBody,
    RecordedMessage,
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


def test_plain_user_turn_passes_through_when_no_tool_call_is_outstanding():
    transcript = _transcript()
    plain = next(turn for turn in transcript.user_turns if "tool_result" not in turn.block_types())

    assert attach_turn(plain, ()) is plain


def test_plain_user_turn_still_answers_an_outstanding_tool_call():
    """Anthropic rejects a plain user message straight after an unanswered tool_use, so an arm
    that called a tool where the recording next has ordinary text would 400 on the following turn."""
    transcript = _transcript()
    plain = next(turn for turn in transcript.user_turns if "tool_result" not in turn.block_types())

    attached = attach_turn(plain, ("toolu_ARM",))

    assert attached is not None
    kinds = [block.type for block in attached.blocks()]
    assert kinds[0] == "tool_result"
    assert attached.blocks()[0].tool_use_id == "toolu_ARM"
    assert "text" in kinds
    assert "Plan my week" in attached.text()


def test_surplus_recorded_results_are_dropped_not_piled_onto_the_last_id():
    """Mapping extras onto the final pending id repeats one tool_use_id in a single message,
    which the provider rejects."""
    turn = RecordedMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "orig_1", "content": "a"},
                {"type": "tool_result", "tool_use_id": "orig_2", "content": "b"},
                {"type": "tool_result", "tool_use_id": "orig_3", "content": "c"},
            ],
        }
    )

    attached = attach_turn(turn, ("toolu_ARM",))

    assert attached is not None
    ids = [block.tool_use_id for block in attached.blocks()]
    assert ids == ["toolu_ARM"]
    assert len(ids) == len(set(ids))


def test_tool_use_ids_reads_only_tool_use_blocks():
    transcript = _transcript()
    assistant_blocks = RecordedBody.model_validate(RECORDED_BODY).messages[2].blocks()

    assert tool_use_ids(assistant_blocks) == ("toolu_ORIGINAL",)
    assert tool_use_ids(transcript.system) == ()


def test_transport_fields_never_reach_the_replay():
    """The recorded body is caller-controlled and the spend-log snapshot strips api_key but not
    api_base, so forwarding unrecognized keys would let a seeded session redirect an admin's
    replay traffic to an attacker-chosen endpoint."""
    hostile = dict(RECORDED_BODY) | {
        "api_base": "https://attacker.example",
        "api_key": "sk-leak",
        "base_url": "https://attacker.example",
        "custom_llm_provider": "openai",
        "extra_headers": {"x-exfil": "1"},
        "vertex_project": "someone-elses",
    }

    transcript = build_transcript(RecordedBody.model_validate(hostile))
    keys = {key for key, _ in transcript.sampling_params}

    assert keys.isdisjoint(
        {"api_base", "api_key", "base_url", "custom_llm_provider", "extra_headers", "vertex_project"}
    )
    assert {"max_tokens", "thinking"} <= keys


def test_tool_result_content_may_be_a_list_of_blocks():
    """Anthropic tool results carry either a string or a list of content blocks, and rejecting
    the list form would make every session using it unreplayable."""
    turn = RecordedMessage.model_validate(
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "orig_1",
                    "content": [{"type": "text", "text": "file body"}],
                }
            ],
        }
    )

    attached = attach_turn(turn, ("toolu_ARM",))

    assert attached is not None
    assert attached.blocks()[0].tool_use_id == "toolu_ARM"
