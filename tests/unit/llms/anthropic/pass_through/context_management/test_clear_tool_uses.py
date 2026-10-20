"""
Unit tests for the in-gateway `clear_tool_uses_20250919` polyfill editor.
"""

from copy import deepcopy
from itertools import chain
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.llms.anthropic.pass_through.adapters.handler import _run_polyfill_if_enabled
from litellm.llms.anthropic.pass_through.context_management import (
    AnthropicContextManagementError,
    PolyfillResult,
    apply_context_management,
)
from litellm.llms.anthropic.pass_through.context_management.constants import (
    CLEARED_TOOL_RESULT_PLACEHOLDER,
    COMPACT_SUMMARY_MODEL_SETTING_KEY,
)
from litellm.llms.anthropic.pass_through.context_management.editors.clear_tool_uses import (
    apply_clear_tool_uses_20250919,
)
from litellm.types.llms.anthropic import AppliedEdit

MODEL = "xai/grok-4"


def _make_pair(tool_use_id: str, result_text: str, location: str = "Mumbai", name: str = "get_weather"):
    """Return an (assistant, user) message pair with one tool_use + tool_result."""
    assistant_msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": name,
                "input": {"location": location},
            }
        ],
    }
    user_msg = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": result_text,
            }
        ],
    }
    return assistant_msg, user_msg


def _make_history(n_pairs: int, result_filler: str = "x" * 200):
    messages = [{"role": "user", "content": "Compare weather across cities."}]
    for i in range(n_pairs):
        assistant_msg, user_msg = _make_pair(
            tool_use_id=f"toolu_{i:02d}",
            result_text=f"Result {i}: {result_filler}",
            location=f"City{i}",
        )
        messages.append(assistant_msg)
        messages.append(user_msg)
    return messages


def test_below_trigger_returns_unchanged():
    """If trigger threshold isn't exceeded, editor is a no-op."""
    messages = _make_history(n_pairs=2)
    original = deepcopy(messages)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "input_tokens", "value": 10_000_000},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is None
    assert new_messages == original


def test_keep_preserves_most_recent_pairs():
    """With keep=2 and 5 pairs, the 3 oldest pairs are cleared."""
    messages = _make_history(n_pairs=5)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 2},
        },
    )
    assert applied is not None
    assert applied["type"] == "clear_tool_uses_20250919"
    assert applied["cleared_tool_uses"] == 3

    # Tool results for the first 3 pairs should be the placeholder, last 2 untouched.
    cleared_ids = {"toolu_00", "toolu_01", "toolu_02"}
    kept_ids = {"toolu_03", "toolu_04"}
    for msg in new_messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get("type") != "tool_result":
                continue
            if block["tool_use_id"] in cleared_ids:
                assert block["content"] == CLEARED_TOOL_RESULT_PLACEHOLDER
            elif block["tool_use_id"] in kept_ids:
                assert "Result" in block["content"]


def test_tool_use_input_is_not_cleared():
    """clear_tool_inputs defaults to false — tool_use.input must remain intact."""
    messages = _make_history(n_pairs=3)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    # Every tool_use block still has its original `input`.
    for msg in new_messages:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if block.get("type") == "tool_use":
                assert block["input"] == {"location": block["input"]["location"]}
                assert block["input"]["location"].startswith("City")


def test_message_array_length_and_roles_preserved():
    messages = _make_history(n_pairs=4)
    original_roles = [m["role"] for m in messages]
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    assert len(new_messages) == len(messages)
    assert [m["role"] for m in new_messages] == original_roles


def test_defaults_applied_when_knobs_omitted():
    """No trigger/keep specified — defaults are 100k input_tokens / 3 tool_uses."""
    messages = _make_history(n_pairs=2)
    # Below 100k tokens; should not fire.
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={"type": "clear_tool_uses_20250919"},
    )
    assert applied is None
    assert new_messages == messages


def test_tool_uses_trigger_variant():
    """Trigger by raw count of tool_use blocks, not tokens."""
    messages = _make_history(n_pairs=4)
    _, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 2},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    # 4 total - 1 kept = 3 cleared
    assert applied["cleared_tool_uses"] == 3


def test_cleared_input_tokens_is_nonnegative():
    messages = _make_history(n_pairs=4)
    _, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 1},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    assert applied["cleared_input_tokens"] >= 0


def _blocks_of(messages: list[dict[str, object]], block_type: str) -> tuple[dict[str, object], ...]:
    contents: Final = (message["content"] for message in messages)
    blocks: Final = chain.from_iterable(content for content in contents if isinstance(content, list))
    return tuple(block for block in blocks if isinstance(block, dict) and block["type"] == block_type)


def _result_contents(messages: list[dict[str, object]]) -> dict[object, object]:
    return {block["tool_use_id"]: block["content"] for block in _blocks_of(messages, "tool_result")}


def _cleared_ids(messages: list[dict[str, object]]) -> set[object]:
    placeholders: Final = (CLEARED_TOOL_RESULT_PLACEHOLDER, [{"type": "text", "text": CLEARED_TOOL_RESULT_PLACEHOLDER}])
    return {tool_use_id for tool_use_id, content in _result_contents(messages).items() if content in placeholders}


def _result_tokens(text: str) -> int:
    return litellm.token_counter(model=MODEL, text=text)


def test_clear_at_least_clears_in_batches_and_leaves_an_incomplete_batch():
    """Oldest first, a batch closes once it reaches clear_at_least; the rest waits."""
    messages = _make_history(n_pairs=8)
    one_result = _result_tokens(_result_contents(messages)["toolu_00"])
    edit_spec = {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "tool_uses", "value": 0},
        "keep": {"type": "tool_uses", "value": 1},
        # Three results fill a batch; two don't.
        "clear_at_least": {"type": "input_tokens", "value": 2 * one_result + 1},
    }
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL, messages=messages, tools=None, system=None, edit_spec=edit_spec
    )
    # 7 eligible: two batches of 3; the 7th waits for its batch to fill.
    assert applied is not None
    assert applied["cleared_tool_uses"] == 6
    assert _cleared_ids(new_messages) == {f"toolu_{i:02d}" for i in range(6)}


def test_clear_at_least_keeps_the_cleared_prefix_stable_between_requests():
    """A new tool use doesn't move the cleared prefix until the next batch fills (prompt-cache friendly)."""
    history = _make_history(n_pairs=10)
    one_result = _result_tokens(_result_contents(history)["toolu_00"])
    edit_spec = {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "tool_uses", "value": 0},
        "keep": {"type": "tool_uses", "value": 1},
        "clear_at_least": {"type": "input_tokens", "value": 2 * one_result + 1},
    }
    cleared: Final = tuple(
        _cleared_ids(
            apply_clear_tool_uses_20250919(
                model=MODEL, messages=history[: 1 + 2 * n_pairs], tools=None, system=None, edit_spec=edit_spec
            )[0]
        )
        for n_pairs in (8, 9, 10)
    )
    assert cleared[0] == cleared[1] == {f"toolu_{i:02d}" for i in range(6)}
    assert cleared[2] == {f"toolu_{i:02d}" for i in range(9)}


def test_clear_at_least_out_of_reach_leaves_messages_unchanged():
    messages = _make_history(n_pairs=3)
    original = deepcopy(messages)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_at_least": {"type": "input_tokens", "value": 999_999_999},
        },
    )
    assert applied is None
    assert new_messages == original


def test_a_batch_closes_when_it_frees_exactly_clear_at_least():
    messages = _make_history(n_pairs=6)
    clear_one_use = {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "tool_uses", "value": 0},
        "keep": {"type": "tool_uses", "value": 5},
    }
    _, one = apply_clear_tool_uses_20250919(
        model=MODEL, messages=messages, tools=None, system=None, edit_spec=clear_one_use
    )
    assert one is not None
    assert one["cleared_tool_uses"] == 1
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            **clear_one_use,
            "keep": {"type": "tool_uses", "value": 1},
            "clear_at_least": {"type": "input_tokens", "value": 2 * one["cleared_input_tokens"]},
        },
    )
    assert _cleared_ids(new_messages) == {f"toolu_{i:02d}" for i in range(4)}
    assert applied is not None
    assert applied["cleared_tool_uses"] == 4


def test_exclude_tools_are_never_cleared_and_do_not_count_toward_keep():
    """keep counts clearable uses only: with keep=1 the newest get_weather result stays, not the newest search."""
    messages = [{"role": "user", "content": "Plan a trip."}]
    for i, name in enumerate(["get_weather", "search", "get_weather", "get_weather", "search"]):
        messages.extend(_make_pair(f"toolu_{i:02d}", f"Result {i}: " + "x" * 200, name=name))
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "exclude_tools": ["search"],
        },
    )
    assert applied is not None
    assert _cleared_ids(new_messages) == {"toolu_00", "toolu_02"}
    assert applied["cleared_tool_uses"] == 2


def test_history_of_only_excluded_tools_is_left_unchanged():
    messages = [{"role": "user", "content": "Plan a trip."}]
    for i in range(4):
        messages.extend(_make_pair(f"toolu_{i:02d}", f"Result {i}: " + "x" * 200, name="search"))
    original = deepcopy(messages)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "exclude_tools": ["search"],
        },
    )
    assert applied is None
    assert new_messages == original


def _clear_with_keep_two(messages: list[dict[str, object]]) -> tuple[list[dict[str, object]], AppliedEdit | None]:
    return apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 2},
        },
    )


def test_tiny_results_are_never_cleared_and_do_not_count_toward_keep():
    messages = _make_history(6)
    messages[2]["content"][0]["content"] = "(Bash completed with no output)"
    messages[12]["content"][0]["content"] = [{"type": "text", "text": "\u9ce5" * 15}]
    new_messages, applied = _clear_with_keep_two(messages)
    assert _cleared_ids(new_messages) == {"toolu_01", "toolu_02"}
    assert applied is not None
    assert applied["cleared_tool_uses"] == 2


def _clear_with_keep_two_and_inputs(
    messages: list[dict[str, object]], clear_tool_inputs: bool | list[str]
) -> tuple[list[dict[str, object]], AppliedEdit | None]:
    return apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 2},
            "clear_tool_inputs": clear_tool_inputs,
        },
    )


def test_tiny_results_count_and_clear_when_their_tool_inputs_are_cleared():
    for clear_tool_inputs in (True, ["get_weather"]):
        messages = _make_history(6)
        messages[2]["content"][0]["content"] = "(Bash completed with no output)"
        messages[12]["content"][0]["content"] = "ok"
        new_messages, applied = _clear_with_keep_two_and_inputs(messages, clear_tool_inputs)
        assert _cleared_ids(new_messages) == {"toolu_00", "toolu_01", "toolu_02", "toolu_03"}
        assert applied is not None
        assert applied["cleared_tool_uses"] == 4


def test_tiny_results_stay_out_when_clear_tool_inputs_names_other_tools():
    messages = _make_history(6)
    messages[2]["content"][0]["content"] = "(Bash completed with no output)"
    messages[12]["content"][0]["content"] = "ok"
    new_messages, applied = _clear_with_keep_two_and_inputs(messages, ["read_file"])
    assert _cleared_ids(new_messages) == {"toolu_01", "toolu_02"}
    assert applied is not None
    assert applied["cleared_tool_uses"] == 2


def test_results_over_46_bytes_and_empty_results_still_count():
    # The 46 byte cutoff was measured on api.anthropic.com on 2026-10-19.
    messages = _make_history(5)
    messages[2]["content"][0]["content"] = "x" * 47
    messages[4]["content"][0]["content"] = ""
    messages[6]["content"][0]["content"] = [
        {"type": "text", "text": "ok"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
    ]
    new_messages, applied = _clear_with_keep_two(messages)
    assert _cleared_ids(new_messages) == {"toolu_00", "toolu_01", "toolu_02"}
    assert new_messages[6]["content"][0]["content"] == [{"type": "text", "text": CLEARED_TOOL_RESULT_PLACEHOLDER}]
    assert applied is not None
    assert applied["cleared_tool_uses"] == 3


def test_the_46_byte_cutoff_includes_46_and_counts_utf8_bytes():
    # The 46 byte cutoff was measured on api.anthropic.com on 2026-10-19.
    messages = _make_history(5)
    messages[2]["content"][0]["content"] = "x" * 46
    messages[4]["content"][0]["content"] = "鳥" * 16
    new_messages, applied = _clear_with_keep_two(messages)
    assert _cleared_ids(new_messages) == {"toolu_01", "toolu_02"}
    assert applied is not None
    assert applied["cleared_tool_uses"] == 2


def test_a_result_holding_a_lone_surrogate_is_sized_like_any_other_text():
    messages = _make_history(5)
    messages[2]["content"][0]["content"] = "\ud800"
    new_messages, applied = _clear_with_keep_two(messages)
    assert _cleared_ids(new_messages) == {"toolu_01", "toolu_02"}
    assert applied is not None


def _tool_inputs(messages: list[dict[str, object]]) -> dict[object, object]:
    return {block["id"]: block["input"] for block in _blocks_of(messages, "tool_use")}


def test_clear_tool_inputs_true_empties_the_inputs_of_cleared_uses_only():
    messages = _make_history(n_pairs=3)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_tool_inputs": True,
        },
    )
    assert applied is not None
    assert _tool_inputs(new_messages) == {"toolu_00": {}, "toolu_01": {}, "toolu_02": {"location": "City2"}}
    # The caller's messages are not mutated.
    assert _tool_inputs(messages)["toolu_00"] == {"location": "City0"}


def test_clear_tool_inputs_list_clears_only_the_named_tools():
    messages = [{"role": "user", "content": "Plan a trip."}]
    for i, name in enumerate(["get_weather", "search", "get_weather"]):
        messages.extend(_make_pair(f"toolu_{i:02d}", f"Result {i}: " + "x" * 200, location=f"City{i}", name=name))
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_tool_inputs": ["search"],
        },
    )
    assert applied is not None
    assert _cleared_ids(new_messages) == {"toolu_00", "toolu_01"}
    assert _tool_inputs(new_messages) == {
        "toolu_00": {"location": "City0"},
        "toolu_01": {},
        "toolu_02": {"location": "City2"},
    }


def test_a_tool_use_with_a_non_string_id_is_left_for_the_provider_to_reject():
    messages = _make_history(n_pairs=4)
    messages[1]["content"][0]["id"] = ["toolu_00"]
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_tool_inputs": True,
        },
    )
    assert applied is not None
    assert new_messages[1] == messages[1]
    assert _cleared_ids(new_messages) == {"toolu_01", "toolu_02"}
    assert (new_messages[3]["content"][0]["input"], new_messages[5]["content"][0]["input"]) == ({}, {})


async def _malformed_spec_error(edits: list[dict[str, object]]) -> AnthropicContextManagementError:
    with pytest.raises(AnthropicContextManagementError) as raised:
        await apply_context_management(
            model=MODEL,
            messages=_make_history(n_pairs=3),
            tools=None,
            system=None,
            context_management_spec={"edits": edits},
        )
    return raised.value


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("clear_at_least", {"type": "tool_uses", "value": 5}),
        ("clear_at_least", {"value": 5}),
        ("clear_at_least", {"type": "input_tokens", "value": -1}),
        ("clear_at_least", {"type": "input_tokens", "value": 1.5}),
        ("clear_at_least", {"type": "input_tokens", "value": "1200"}),
        ("clear_at_least", {"type": "input_tokens", "value": True}),
        ("exclude_tools", "get_weather"),
        ("exclude_tools", [1]),
        ("clear_tool_inputs", "get_weather"),
    ],
)
async def test_malformed_fields_are_a_400_even_below_the_trigger(field: str, value: object) -> None:
    error = await _malformed_spec_error(
        [{"type": "clear_tool_uses_20250919", "trigger": {"type": "tool_uses", "value": 99}, field: value}]
    )
    assert error.status_code == 400
    assert error.message.startswith(f"context_management.edits.0.clear_tool_uses_20250919.{field}")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exclude_tools", "memory", "exclude_tools: Input should be a valid array"),
        ("clear_at_least", 1200, "clear_at_least: Input does not match the expected shape."),
        ("clear_at_least", {"type": "input_tokens", "value": "1200"}, "value: Input should be a valid integer"),
        ("clear_tool_inputs", 5, "clear_tool_inputs.bool: Input should be a valid boolean, unable to interpret input"),
    ],
)
async def test_malformed_field_400_names_the_edit_and_field_as_the_api_does(
    field: str, value: object, message: str
) -> None:
    # api.anthropic.com, measured 2026-10-19, answers these four with these messages for the edit at index 1.
    error = await _malformed_spec_error(
        [{"type": "clear_thinking_20251015"}, {"type": "clear_tool_uses_20250919", field: value}]
    )
    assert error.message.startswith("context_management.edits.1.clear_tool_uses_20250919.")
    assert error.message.endswith(message)


_VALID_EDIT: Final = {
    "type": "clear_tool_uses_20250919",
    "trigger": {"type": "tool_uses", "value": 0},
    "keep": {"type": "tool_uses", "value": 1},
}
_MALFORMED_EDIT: Final = {**_VALID_EDIT, "exclude_tools": "get_weather"}


@pytest.mark.parametrize(
    ("field", "value"),
    [("exclude_tools", "get_weather"), ("clear_at_least", {"type": "input_tokens", "value": "1200"})],
)
async def test_drop_params_drops_only_the_malformed_field(field: str, value: object) -> None:
    async def run(drop_invalid_edits: bool) -> PolyfillResult | None:
        return await _run_polyfill_if_enabled(
            model=MODEL,
            messages=_make_history(n_pairs=3),
            tools=None,
            system=None,
            context_management_spec={"edits": [{**_VALID_EDIT, field: value, "clear_tool_inputs": True}]},
            litellm_metadata={},
            additional_drop_params=None,
            llm_router=None,
            drop_invalid_edits=drop_invalid_edits,
        )

    with pytest.raises(
        AnthropicContextManagementError, match=rf"^context_management\.edits\.0\.clear_tool_uses_20250919\.{field}"
    ):
        await run(drop_invalid_edits=False)
    result: Final = await run(drop_invalid_edits=True)
    assert result is not None
    assert _cleared_ids(result.messages) == {"toolu_00", "toolu_01"}
    assert _tool_inputs(result.messages) == {"toolu_00": {}, "toolu_01": {}, "toolu_02": {"location": "City2"}}


async def test_malformed_edit_is_rejected_before_a_compact_summary_is_paid_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.proxy.proxy_server import general_settings

    monkeypatch.setitem(general_settings, COMPACT_SUMMARY_MODEL_SETTING_KEY, "summary-model")
    router: Final = MagicMock()
    router.acompletion = AsyncMock(side_effect=RuntimeError("summary model unavailable"))
    with pytest.raises(AnthropicContextManagementError):
        await apply_context_management(
            model=MODEL,
            messages=[{"role": "user", "content": "hello " * 60_000}],
            tools=None,
            system=None,
            context_management_spec={
                "edits": [
                    {"type": "compact_20260112", "trigger": {"type": "input_tokens", "value": 50_000}},
                    _MALFORMED_EDIT,
                ]
            },
            llm_router=router,
        )
    router.acompletion.assert_not_awaited()


@pytest.mark.parametrize(("clear_tool_inputs", "cleared"), [("yes", True), ("true", True), (1, True), ("no", False)])
def test_clear_tool_inputs_reads_boolean_strings_and_numbers_like_the_api(
    clear_tool_inputs: object, cleared: bool
) -> None:
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=_make_history(n_pairs=3),
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_tool_inputs": clear_tool_inputs,
        },
    )
    assert applied is not None
    assert _tool_inputs(new_messages)["toolu_00"] == ({} if cleared else {"location": "City0"})


@pytest.mark.parametrize(("trigger_value", "triggers"), [(2, False), (1, True)])
def test_tool_uses_trigger_does_not_count_excluded_tools(trigger_value: int, triggers: bool) -> None:
    messages = [{"role": "user", "content": "Plan a trip."}]
    for i, name in enumerate(["search", "get_weather", "search", "get_weather"]):
        messages.extend(_make_pair(f"toolu_{i:02d}", f"Result {i}: " + "x" * 200, name=name))
    _, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": trigger_value},
            "keep": {"type": "tool_uses", "value": 0},
            "exclude_tools": ["search"],
        },
    )
    assert (applied is not None) == triggers


@pytest.mark.parametrize(("clear_tool_inputs", "triggers"), [(False, False), (True, True)])
def test_tool_uses_trigger_counts_tiny_results_only_when_their_inputs_are_cleared(
    clear_tool_inputs: bool, triggers: bool
) -> None:
    messages = _make_history(n_pairs=3)
    messages[2]["content"][0]["content"] = "ok"
    _, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 2},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_tool_inputs": clear_tool_inputs,
        },
    )
    assert (applied is not None) == triggers


def test_clear_at_least_counts_the_tool_inputs_it_clears():
    messages = [{"role": "user", "content": "Write the files."}]
    for i in range(4):
        assistant_msg, user_msg = _make_pair(f"toolu_{i:02d}", "ok", name="write_file")
        assistant_msg["content"][0]["input"] = {"path": f"file{i}.txt", "content": "lorem ipsum " * 300}
        messages.extend((assistant_msg, user_msg))
    one_input = litellm.token_counter(model=MODEL, text="lorem ipsum " * 300)
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_at_least": {"type": "input_tokens", "value": 3 * one_input // 2},
            "clear_tool_inputs": True,
        },
    )
    assert applied is not None
    assert _cleared_ids(new_messages) == {"toolu_00", "toolu_01"}
    assert [_tool_inputs(new_messages)[f"toolu_{i:02d}"] == {} for i in range(4)] == [True, True, False, False]
    assert applied["cleared_input_tokens"] >= 3 * one_input // 2


_ONE_PIXEL_PNG: Final = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def test_clear_at_least_counts_image_results():
    messages = _make_history(n_pairs=4)
    image: Final = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _ONE_PIXEL_PNG}}
    for user_msg in messages[2::2]:
        user_msg["content"][0]["content"] = [image]
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
            "clear_at_least": {"type": "input_tokens", "value": 1},
        },
    )
    assert applied is not None
    assert _cleared_ids(new_messages) == {"toolu_00", "toolu_01", "toolu_02"}


def test_tool_result_list_content_shape_preserved():
    """When tool_result.content is a list of blocks, replacement returns a list shape."""
    messages = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_a", "name": "f", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_a",
                    "content": [{"type": "text", "text": "huge result: " + "x" * 200}],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_b", "name": "f", "input": {}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_b",
                    "content": [{"type": "text", "text": "keep me: " + "y" * 200}],
                }
            ],
        },
    ]
    new_messages, applied = apply_clear_tool_uses_20250919(
        model=MODEL,
        messages=messages,
        tools=None,
        system=None,
        edit_spec={
            "type": "clear_tool_uses_20250919",
            "trigger": {"type": "tool_uses", "value": 0},
            "keep": {"type": "tool_uses", "value": 1},
        },
    )
    assert applied is not None
    cleared_block = new_messages[2]["content"][0]
    assert isinstance(cleared_block["content"], list)
    assert cleared_block["content"][0]["type"] == "text"
    assert cleared_block["content"][0]["text"] == CLEARED_TOOL_RESULT_PLACEHOLDER
