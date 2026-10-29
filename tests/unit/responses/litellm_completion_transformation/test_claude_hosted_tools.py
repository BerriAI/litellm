import json
from collections.abc import Mapping
from typing import Final

import pytest

from litellm.exceptions import UnsupportedParamsError
from litellm.responses.litellm_completion_transformation.claude_hosted_tools import (
    claude_computer_member_call,
    openai_computer_call,
)


def _openai_action(member: str, member_input: Mapping[str, object]) -> dict[str, object]:
    computer_call: Final = openai_computer_call("toolu_1", member, json.dumps(member_input), "completed")
    assert computer_call is not None
    return computer_call.action.model_dump(exclude_none=True)


@pytest.mark.parametrize(
    "member, member_input",
    [
        ("screenshot", {}),
        ("type", {"text": "hello world"}),
        ("key", {"text": "ctrl+shift+Return"}),
        ("key", {"text": "a"}),
        ("left_click", {"coordinate": [10, 20]}),
        ("right_click", {"coordinate": [10, 20], "text": "shift"}),
        ("middle_click", {"coordinate": [10, 20]}),
        ("double_click", {"coordinate": [10, 20], "text": "alt"}),
        ("mouse_move", {"coordinate": [10, 20]}),
        ("left_click_drag", {"start_coordinate": [1, 2], "coordinate": [30, 40]}),
        ("scroll", {"coordinate": [10, 20], "scroll_direction": "down", "scroll_amount": 3}),
        ("scroll", {"coordinate": [10, 20], "scroll_direction": "up", "scroll_amount": 2}),
        ("scroll", {"coordinate": [10, 20], "scroll_direction": "left", "scroll_amount": 1}),
        ("scroll", {"coordinate": [10, 20], "scroll_direction": "right", "scroll_amount": 4, "text": "shift"}),
    ],
)
def test_claude_member_call_survives_the_trip_through_an_openai_computer_call(
    member: str, member_input: Mapping[str, object]
) -> None:
    assert claude_computer_member_call(_openai_action(member, member_input)) == (member, member_input)


def test_scroll_amount_in_wheel_clicks_becomes_a_pixel_distance_in_the_scroll_direction() -> None:
    down: Final = _openai_action("scroll", {"coordinate": [5, 5], "scroll_direction": "down", "scroll_amount": 2})
    up: Final = _openai_action("scroll", {"coordinate": [5, 5], "scroll_direction": "up", "scroll_amount": 2})

    assert down["scroll_x"] == up["scroll_x"] == 0
    assert down["scroll_y"] == -up["scroll_y"] > 0


def test_xdotool_key_names_become_openai_key_names_and_back() -> None:
    keypress: Final = _openai_action("key", {"text": "ctrl+Return"})

    assert keypress == {"type": "keypress", "keys": ["CTRL", "ENTER"]}
    assert claude_computer_member_call({"type": "keypress", "keys": ["ctrl", "enter"]}) == (
        "key",
        {"text": "ctrl+Return"},
    )


def test_a_replayed_wait_waits_a_fixed_second_since_openai_wait_has_no_duration() -> None:
    assert _openai_action("wait", {"duration": 5}) == {"type": "wait"}
    assert claude_computer_member_call({"type": "wait"}) == ("wait", {"duration": 1})


def test_a_diagonal_openai_scroll_keeps_its_dominant_axis_and_at_least_one_wheel_click() -> None:
    assert claude_computer_member_call({"type": "scroll", "x": 1, "y": 2, "scroll_x": 30, "scroll_y": -40}) == (
        "scroll",
        {"coordinate": [1, 2], "scroll_direction": "up", "scroll_amount": 1},
    )
    assert claude_computer_member_call({"type": "scroll", "x": 1, "y": 2, "scroll_x": 250, "scroll_y": 0}) == (
        "scroll",
        {"coordinate": [1, 2], "scroll_direction": "right", "scroll_amount": 2},
    )


@pytest.mark.parametrize(
    "member, member_input",
    [
        ("left_click", {}),
        ("scroll", {"coordinate": [1, 2]}),
        ("left_click_drag", {"coordinate": [1, 2]}),
        ("type", {}),
        ("zoom", {"region": [0, 0, 10, 10]}),
    ],
)
def test_member_call_with_no_openai_action_gives_no_computer_call(
    member: str, member_input: Mapping[str, object]
) -> None:
    assert openai_computer_call("toolu_1", member, json.dumps(member_input), "completed") is None


def test_malformed_member_arguments_give_no_computer_call() -> None:
    assert openai_computer_call("toolu_1", "left_click", '{"coordinate": "top left"}', "completed") is None


@pytest.mark.parametrize(
    "action",
    [
        {"type": "click", "button": "back", "x": 1, "y": 2},
        {"type": "drag", "path": [{"x": 1, "y": 2}]},
        {"type": "teleport"},
        None,
    ],
    ids=["back_button", "single_point_drag", "unknown_type", "missing"],
)
def test_computer_call_input_claude_cannot_run_answers_400(action: object) -> None:
    with pytest.raises(UnsupportedParamsError) as rejected:
        claude_computer_member_call(action)

    assert rejected.value.status_code == 400
