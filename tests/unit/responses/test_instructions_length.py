from typing import Final

import pytest

from litellm.responses.instructions_length import RESPONSES_INSTRUCTIONS_MAX_LENGTH, with_long_instructions_in_input

_OVER_LIMIT: Final = "a" * (RESPONSES_INSTRUCTIONS_MAX_LENGTH + 1)
_USER_HI: Final = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
_SYSTEM_ITEM: Final = {"type": "message", "role": "system", "content": [{"type": "input_text", "text": _OVER_LIMIT}]}


@pytest.mark.parametrize(
    ("request_body", "expected"),
    [
        (
            {"instructions": _OVER_LIMIT, "input": [_USER_HI], "truncation": "auto"},
            {"input": [_SYSTEM_ITEM, _USER_HI], "truncation": "auto"},
        ),
        (
            {"input": [_USER_HI], "extra_body": {"instructions": _OVER_LIMIT, "truncation": "auto"}},
            {"input": [_SYSTEM_ITEM, _USER_HI], "extra_body": {"truncation": "auto"}},
        ),
        (
            {"instructions": _OVER_LIMIT, "input": [_USER_HI], "extra_body": {"input": [_USER_HI]}},
            {"input": [_SYSTEM_ITEM, _USER_HI], "extra_body": {"input": [_SYSTEM_ITEM, _USER_HI]}},
        ),
    ],
    ids=["top_level_instructions", "extra_body_instructions", "caller_input_override_in_extra_body"],
)
def test_instructions_too_long_for_the_responses_api_go_first_in_the_input_the_request_sends(
    request_body: dict[str, object], expected: dict[str, object]
) -> None:
    assert with_long_instructions_in_input(request_body) == expected


@pytest.mark.parametrize(
    "request_body",
    [
        {"instructions": "a" * RESPONSES_INSTRUCTIONS_MAX_LENGTH, "input": [_USER_HI]},
        {"instructions": _OVER_LIMIT, "input": "hi"},
    ],
    ids=["at_limit", "string_input_is_left_for_the_provider_to_reject"],
)
def test_requests_the_limit_does_not_apply_to_are_sent_unchanged(request_body: dict[str, object]) -> None:
    assert with_long_instructions_in_input(request_body) is request_body
