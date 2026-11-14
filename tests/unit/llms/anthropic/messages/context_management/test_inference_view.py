import logging
from typing import Final
from unittest.mock import patch

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.llms.anthropic.messages.context_management.inference_view import (
    CountContextEdits,
    InferenceRequestView,
    InferenceRequestViewError,
    inference_request_view,
)

_NO_EDITS: Final = CountContextEdits(spec=None, model="gemini/gemini-2.5-flash", tools=None)


@pytest.mark.parametrize(
    "system", ("Be brief.", [{"type": "text", "text": "Be brief."}]), ids=("string-system", "text-block-system")
)
async def test_anthropic_view_keeps_only_the_turns_after_a_client_compaction_block(system: JsonValue) -> None:
    view: Final = await inference_request_view(
        (
            {"role": "user", "content": "first question " * 50},
            {"role": "assistant", "content": [{"type": "compaction", "content": "Summary of the earlier turns."}]},
            {"role": "user", "content": "latest question"},
        ),
        system,
        "anthropic",
        _NO_EDITS,
    )

    assert isinstance(view, InferenceRequestView)
    assert view.messages == [{"role": "user", "content": "latest question"}]
    assert "Summary of the earlier turns." in str(view.system)
    assert "Be brief." in str(view.system)


async def test_anthropic_view_flattens_a_replayed_web_search_result_into_text() -> None:
    snippet: Final = "LiteLLM ships a weekly release candidate and a stable line."
    view: Final = await inference_request_view(
        (
            {"role": "user", "content": "What is new?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {"query": "news"}},
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://docs.litellm.ai/release_notes",
                                "title": "Release notes",
                                "encrypted_content": "",
                                "snippet": snippet,
                            }
                        ],
                    },
                ],
            },
        ),
        None,
        "anthropic",
        _NO_EDITS,
    )

    assert isinstance(view, InferenceRequestView)
    assistant_blocks: Final = TypeAdapter(list[dict[str, str]]).validate_python(view.messages[1]["content"])
    assert [block["type"] for block in assistant_blocks] == ["text"]
    assert snippet in assistant_blocks[0]["text"]


async def test_openai_view_is_the_request_unchanged() -> None:
    messages: Final = ({"role": "user", "content": [{"type": "compaction", "content": "kept"}]},)

    view: Final = await inference_request_view(messages, "Be brief.", "openai", _NO_EDITS)

    assert isinstance(view, InferenceRequestView)
    assert view.messages == messages
    assert view.system == "Be brief."


_TOOL_OUTPUT: Final = "line of a long file that the agent read earlier. " * 180
_READ_FILE_TOOL: Final[dict[str, object]] = {
    "name": "read_file",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}
_CLEAR_EDIT: Final[JsonValue] = {
    "type": "clear_tool_uses_20250919",
    "trigger": {"type": "input_tokens", "value": 500},
    "keep": {"type": "tool_uses", "value": 1},
}
_CLEAR_OLD_TOOL_RESULTS: Final[JsonValue] = {"edits": [_CLEAR_EDIT]}


def _agent_turns(rounds: int, first_round: int = 0) -> tuple[dict[str, JsonValue], ...]:
    return (
        *(
            turn
            for index in range(first_round, first_round + rounds)
            for turn in (
                {"role": "user", "content": f"Read file {index}."},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": f"toolu_{index}", "name": "read_file", "input": {"path": f"{index}"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": f"toolu_{index}", "content": _TOOL_OUTPUT}],
                },
            )
        ),  # comprehension-ok: flattens each round into its three turns
        {"role": "user", "content": "Summarize what you read."},
    )


def _tool_outputs_kept(view: InferenceRequestView) -> tuple[bool, ...]:
    return tuple(
        _TOOL_OUTPUT in str(message["content"]) for message in view.messages if "tool_result" in str(message["content"])
    )


def _edits(spec: JsonValue) -> CountContextEdits:
    return CountContextEdits(spec=spec, model="gemini/gemini-2.5-flash", tools=(_READ_FILE_TOOL,))


async def test_anthropic_view_clears_the_tool_results_inference_clears() -> None:
    view: Final = await inference_request_view(_agent_turns(3), None, "anthropic", _edits(_CLEAR_OLD_TOOL_RESULTS))

    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (False, False, True)


async def test_anthropic_view_applies_clear_edits_without_calling_the_compaction_summary_model() -> None:
    spec: Final[JsonValue] = {
        "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50000}}, _CLEAR_EDIT]
    }
    turns: Final = (
        {"role": "user", "content": "first question " * 30000},
        {"role": "assistant", "content": "first answer"},
        *_agent_turns(3),
    )

    with (
        patch(
            "litellm.llms.anthropic.messages.context_management.editors.compact._read_summary_model_setting",
            return_value="gemini/gemini-2.5-flash",
        ),
        patch.object(litellm, "acompletion", side_effect=AssertionError("count_tokens must not summarize")) as summary,
    ):
        view: Final = await inference_request_view(turns, None, "anthropic", _edits(spec))

    assert summary.call_count == 0
    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (False, False, True)


async def test_anthropic_view_counts_the_unedited_body_without_an_error_log_when_an_edit_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        patch.object(litellm, "token_counter", side_effect=RuntimeError("tokenizer unavailable")),
        caplog.at_level(logging.DEBUG, logger="LiteLLM"),
    ):
        view: Final = await inference_request_view(_agent_turns(3), None, "anthropic", _edits(_CLEAR_OLD_TOOL_RESULTS))

    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (True, True, True)
    assert any("skipping edits" in record.getMessage() for record in caplog.records)
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


async def test_anthropic_view_counts_the_turns_after_a_compaction_block_when_an_edit_fails() -> None:
    turns: Final = (
        *_agent_turns(3)[:-1],
        {"role": "assistant", "content": [{"type": "compaction", "content": "Read files 0 to 2."}]},
        *_agent_turns(2, first_round=3),
    )

    with patch.object(litellm, "token_counter", side_effect=RuntimeError("tokenizer unavailable")):
        view: Final = await inference_request_view(turns, None, "anthropic", _edits(_CLEAR_OLD_TOOL_RESULTS))

    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (True, True)
    assert "Read files 0 to 2." in str(view.system)


async def test_anthropic_view_clears_only_the_turns_after_a_compaction_block_when_clear_runs_before_compact() -> None:
    clear_before_compact: Final[JsonValue] = {
        "edits": [
            {
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "tool_uses", "value": 3},
                "keep": {"type": "tool_uses", "value": 1},
            },
            {"type": "compact_20260112"},
        ]
    }
    turns: Final = (
        *_agent_turns(3)[:-1],
        {"role": "assistant", "content": [{"type": "compaction", "content": "Read files 0 to 2."}]},
        *_agent_turns(2, first_round=3),
    )

    with patch.object(litellm, "token_counter", side_effect=RuntimeError("tokenizer unavailable")):
        view: Final = await inference_request_view(turns, None, "anthropic", _edits(clear_before_compact))

    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (True, True)
    assert "Read files 0 to 2." in str(view.system)


async def test_anthropic_view_leaves_a_system_that_is_not_text_blocks_and_its_turns_to_the_provider_check() -> None:
    image_system: Final[JsonValue] = [{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]

    view: Final = await inference_request_view(
        _agent_turns(3), image_system, "anthropic", _edits(_CLEAR_OLD_TOOL_RESULTS)
    )

    assert isinstance(view, InferenceRequestView)
    assert _tool_outputs_kept(view) == (True, True, True)
    assert view.system == image_system


async def test_anthropic_view_rejects_a_compact_trigger_inference_rejects() -> None:
    low_trigger: Final[JsonValue] = {
        "edits": [{"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 1000}}]
    }

    view: Final = await inference_request_view(_agent_turns(1), None, "anthropic", _edits(low_trigger))

    assert isinstance(view, InferenceRequestViewError)
    assert view.status_code == 400
