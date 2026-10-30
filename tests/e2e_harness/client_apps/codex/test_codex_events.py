from __future__ import annotations

from pathlib import Path
from typing import Final

from client_apps.codex.events import (
    CommandExecution,
    FileChange,
    ItemError,
    Reasoning,
    Transcript,
    UnknownEvent,
    UnknownItem,
    WebSearch,
    parse_transcript,
)

TRANSCRIPTS: Final = Path(__file__).resolve().parent / "transcripts"


def _transcript(name: str) -> Transcript:
    return parse_transcript((TRANSCRIPTS / name).read_text())


def test_every_recorded_line_parses() -> None:
    unparsed: Final = {path.name: _transcript(path.name).unparsed for path in sorted(TRANSCRIPTS.glob("*.jsonl"))}

    assert len(unparsed) == 11
    assert all(lines == () for lines in unparsed.values()), unparsed


def test_file_change_lists_the_touched_path_once_completed() -> None:
    transcript: Final = _transcript("file_change.jsonl")
    changes: Final = tuple(item for item in transcript.completed_items if isinstance(item, FileChange))

    assert len(changes) == 1
    assert changes[0].status == "completed"
    assert [(update.path, update.kind) for update in changes[0].changes] == [
        ("/tmp/codex-session/workdir/notes.txt", "add")
    ]
    assert transcript.final_message == "Created `notes.txt` containing only `q7w8e9`."
    assert transcript.failure is None


def test_shell_command_keeps_exit_code_and_output_of_the_completed_item() -> None:
    transcript: Final = _transcript("shell_command.jsonl")
    commands: Final = tuple(item for item in transcript.completed_items if isinstance(item, CommandExecution))

    assert [(command.exit_code, command.aggregated_output) for command in commands] == [(0, "c887862269b3\n")]
    assert transcript.final_message == "c887862269b3"


def test_completed_web_search_carries_its_query() -> None:
    searches: Final = tuple(
        item for item in _transcript("web_search.jsonl").completed_items if isinstance(item, WebSearch)
    )

    assert [search.query for search in searches] == ["LiteLLM documentation official"]


def test_web_search_that_only_started_is_not_a_completed_item() -> None:
    transcript: Final = _transcript("web_search_never_completed.jsonl")

    assert not any(isinstance(item, WebSearch) for item in transcript.completed_items)
    assert transcript.final_message == "docs.litellm.ai"
    assert transcript.usage is not None


def test_fallback_metadata_warning_is_an_item_not_a_turn_failure() -> None:
    transcript: Final = _transcript("fallback_metadata.jsonl")
    errors: Final = tuple(item for item in transcript.completed_items if isinstance(item, ItemError))

    assert len(errors) == 1
    assert "fallback metadata" in errors[0].message
    assert transcript.failure is None
    assert any(isinstance(item, Reasoning) and item.text for item in transcript.completed_items)
    assert transcript.usage is not None
    assert transcript.usage.reasoning_output_tokens == 205
    assert transcript.final_message == "15"


def test_turn_failed_reports_the_proxy_error_and_no_usage() -> None:
    transcript: Final = _transcript("turn_failed.jsonl")

    assert transcript.failure is not None
    assert "Invalid model name passed in model=gpt-6.1-sol-does-not-exist" in transcript.failure
    assert transcript.usage is None
    assert transcript.final_message is None


def test_rate_limited_turn_reports_the_final_error_not_a_retry_notice() -> None:
    transcript: Final = _transcript("rate_limited.jsonl")

    assert transcript.failure is not None
    assert transcript.failure.startswith("rate limit exceeded")
    assert "Reconnecting" not in transcript.failure


def test_a_turn_that_recovers_after_a_retry_notice_has_no_failure() -> None:
    transcript: Final = parse_transcript(
        "\n".join(
            (
                '{"type":"thread.started","thread_id":"t-1"}',
                '{"type":"turn.started"}',
                '{"type":"error","message":"Reconnecting... 1/5 (rate limit exceeded: 429)"}',
                '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"done"}}',
                '{"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":1}}',
            )
        )
    )

    assert transcript.failure is None
    assert transcript.final_message == "done"


def test_a_stream_that_ends_on_an_error_without_turn_events_is_a_failure() -> None:
    transcript: Final = parse_transcript(
        '{"type":"thread.started","thread_id":"t-1"}\n{"type":"error","message":"stream disconnected"}\n'
    )

    assert transcript.failure == "stream disconnected"


def test_resumed_turn_keeps_the_thread_and_reports_cached_input() -> None:
    first: Final = _transcript("first_turn.jsonl")
    resumed: Final = _transcript("resumed_turn.jsonl")

    assert first.thread_id is not None
    assert resumed.thread_id == first.thread_id
    assert first.usage is not None
    assert resumed.usage is not None
    assert first.usage.cached_input_tokens == 0
    assert resumed.usage.cached_input_tokens == 10870


def test_unknown_shapes_parse_as_unknown_and_garbage_lines_are_kept_aside() -> None:
    transcript: Final = parse_transcript(
        "\n".join(
            (
                '{"type":"thread.started","thread_id":"t-1"}',
                '{"type":"turn.compacted"}',
                '{"type":"item.completed","item":{"id":"item_0","type":"collab_tool_call"}}',
                '{"type":"item.completed","item":{"id":"item_1","type":"agent_message","text":"done"}}',
                "not json",
                "",
            )
        )
    )

    assert transcript.thread_id == "t-1"
    assert any(isinstance(event, UnknownEvent) and event.type == "turn.compacted" for event in transcript.events)
    assert [item.type for item in transcript.completed_items if isinstance(item, UnknownItem)] == ["collab_tool_call"]
    assert transcript.final_message == "done"
    assert [line.text for line in transcript.unparsed] == ["not json"]
