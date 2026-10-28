"""Unit tests for the tool-search replay assertion in `http_probe`.

Markerless harness tests: they exercise probe plumbing over hand-built
`Result` values, not a product feature, so they run without a proxy and carry
no `e2e` marker.

The red paths are what these are for. A live cell only ever executes the green
one, so a broken diagnostic in the failure branch would sit undetected until
the day the provider actually rejects the history, which is the day the
diagnostic has to be right.
"""

from __future__ import annotations

from e2e_http import Result, Success, UnknownApiError
from models import (
    AnthropicContentBlock,
    AnthropicMessagesResponse,
    AnthropicToolResultTurn,
    ChatMessage,
)

from claude_code.http_probe import (
    ToolSearchReplay,
    _replay_history,
    assert_tool_search_replay_shape,
)

_REJECTED: Result[AnthropicMessagesResponse] = UnknownApiError(
    status_code=400,
    body="server_tool_use blocks are not supported",
)
_ACCEPTED: Result[AnthropicMessagesResponse] = Success(
    status_code=200,
    data=AnthropicMessagesResponse(content=[AnthropicContentBlock(type="text", text="done")]),
)


def _replay(block_types: tuple[str, ...], second_turn: Result[AnthropicMessagesResponse]) -> ToolSearchReplay:
    answer = AnthropicMessagesResponse(
        content=[AnthropicContentBlock(type=block_type, id="srvtoolu_01") for block_type in block_types]
    )
    return ToolSearchReplay(
        first_turn=Success(status_code=200, data=answer),
        history=_replay_history(answer),
        second_turn=second_turn,
    )


def test_accepts_a_replayed_server_tool_pair() -> None:
    replay = _replay(("text", "server_tool_use", "tool_search_tool_result"), _ACCEPTED)
    assert assert_tool_search_replay_shape(replay) is None


def test_reports_the_status_when_the_replayed_history_is_rejected() -> None:
    replay = _replay(("server_tool_use", "tool_search_tool_result"), _REJECTED)
    error = assert_tool_search_replay_shape(replay)
    assert error is not None
    assert "status 400" in error
    assert "server_tool_use" in error


def test_a_turn_truncated_before_the_result_block_is_not_a_pass() -> None:
    replay = _replay(("server_tool_use",), _ACCEPTED)
    error = assert_tool_search_replay_shape(replay)
    assert error is not None
    assert "tool_search_tool_result" in error


def test_a_history_with_no_server_tool_block_is_not_a_pass() -> None:
    replay = _replay(("text",), _ACCEPTED)
    error = assert_tool_search_replay_shape(replay)
    assert error is not None
    assert "server_tool_use" in error


def test_a_failed_first_turn_is_reported_as_the_first_turn() -> None:
    replay = ToolSearchReplay(first_turn=_REJECTED, history=(), second_turn=None)
    error = assert_tool_search_replay_shape(replay)
    assert error is not None
    assert error.startswith("first turn: ")


def test_a_pending_tool_use_is_answered_with_the_id_the_model_returned() -> None:
    answer = AnthropicMessagesResponse(
        content=[
            AnthropicContentBlock(type="server_tool_use", id="srvtoolu_01"),
            AnthropicContentBlock(type="tool_search_tool_result", id=None),
            AnthropicContentBlock(type="tool_use", id="toolu_99"),
        ]
    )
    last_turn = _replay_history(answer)[-1]
    assert isinstance(last_turn, AnthropicToolResultTurn)
    assert [block.tool_use_id for block in last_turn.content] == ["toolu_99"]


def test_a_turn_with_no_pending_tool_use_gets_a_plain_follow_up() -> None:
    answer = AnthropicMessagesResponse(
        content=[
            AnthropicContentBlock(type="server_tool_use", id="srvtoolu_01"),
            AnthropicContentBlock(type="tool_search_tool_result"),
        ]
    )
    last_turn = _replay_history(answer)[-1]
    assert isinstance(last_turn, ChatMessage)
    assert last_turn.role == "user"
