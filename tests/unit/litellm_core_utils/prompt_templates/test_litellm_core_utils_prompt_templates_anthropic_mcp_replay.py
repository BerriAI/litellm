from typing import Final

from litellm.litellm_core_utils.prompt_templates.anthropic_mcp_replay import mcp_replay_plan


def _mcp_call(call_id: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "ask_question",
            "arguments": "{}",
            "provider_specific_fields": {"server_name": "deepwiki"},
        },
    }


def _mcp_result(call_id: str, text: str) -> dict[str, object]:
    return {
        "type": "mcp_tool_result",
        "tool_use_id": call_id,
        "is_error": False,
        "content": [{"type": "text", "text": text}],
    }


def _assistant(calls: list[str], results: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "Working on it.",
        "tool_calls": [_mcp_call(call_id) for call_id in calls],
        "provider_specific_fields": {"tool_results": results or []},
    }


_USER: Final = {"role": "user", "content": "Go on."}


def test_every_unanswered_or_repeated_mcp_call_of_a_message_is_dropped():
    plan: Final = mcp_replay_plan(
        [_USER, _assistant(["mcptoolu_X"]), _USER, _assistant(["mcptoolu_X", "mcptoolu_Y"]), _USER, _assistant([])],
        replays_mcp_calls=True,
    )

    assert (plan.dropped_ids_at(1), plan.dropped_ids_at(3)) == (
        frozenset({"mcptoolu_X"}),
        frozenset({"mcptoolu_X", "mcptoolu_Y"}),
    )


def test_an_mcp_call_answered_in_a_later_message_takes_the_first_later_result():
    first: Final = _mcp_result("mcptoolu_X", "first answer")
    second: Final = _mcp_result("mcptoolu_X", "second answer")
    plan: Final = mcp_replay_plan(
        [_USER, _assistant(["mcptoolu_X"]), _assistant([], [first]), _USER, _assistant([], [second]), _USER],
        replays_mcp_calls=True,
    )

    assert (plan.lone_results_at(2), plan.lone_results_at(4), plan.answered_later_ids) == (
        (first,),
        (),
        frozenset({"mcptoolu_X"}),
    )
