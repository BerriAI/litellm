from typing import Final

from litellm.responses.bridge_check import bridge_check_request_params


def test_a_dropped_effort_inside_extra_body_is_not_read_by_the_route_decision():
    request: Final = {
        "extra_body": {"reasoning_effort": "low", "reasoningSummary": "auto"},
        "additional_drop_params": ["reasoning_effort"],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }

    params: Final = bridge_check_request_params(request)

    assert params["extra_body"] == {"reasoningSummary": "auto"}, params
    assert params["tools"] is request["tools"]


def test_a_dropped_top_level_alias_is_removed_too():
    params: Final = bridge_check_request_params({"reasoningSummary": "auto", "additional_drop_params": ["reasoningSummary"]})

    assert "reasoningSummary" not in params, params


def test_a_request_without_drops_is_returned_whole():
    request: Final = {"extra_body": {"reasoning_effort": "low"}, "tools": []}

    params: Final = bridge_check_request_params(request)

    assert params == request
    assert params["extra_body"] == {"reasoning_effort": "low"}


def test_a_non_list_additional_drop_params_drops_nothing():
    request: Final = {"extra_body": {"reasoning_effort": "low"}, "additional_drop_params": "reasoning_effort"}

    assert bridge_check_request_params(request)["extra_body"] == {"reasoning_effort": "low"}
