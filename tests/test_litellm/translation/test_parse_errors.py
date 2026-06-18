"""Malformed requests surface a descriptive error value, never an exception.

These pin the failures-as-values contract: each bad shape must come back as an
``Error`` whose ``.summary`` names the problem, so the FastAPI edge has one
string to return and nothing in the package raises.
"""

import pytest

from litellm.translation import translate_chat_request

MODEL = "claude-3-5-sonnet-20241022"
_USER = {"role": "user", "content": "hi"}


def _req(**overrides) -> dict:
    base = {"model": MODEL, "messages": [_USER]}
    base.update(overrides)
    return base


CASES = [
    ("missing model and messages", {}, ["model", "messages"]),
    ("non-object message", _req(messages=[123]), ["each message must be an object"]),
    (
        "unsupported role",
        _req(messages=[{"role": "function", "content": "x"}]),
        ["unsupported message role"],
    ),
    (
        "content wrong type",
        _req(messages=[{"role": "user", "content": 123}]),
        ["string or an array"],
    ),
    (
        "content part not object",
        _req(messages=[{"role": "user", "content": [123]}]),
        ["each content part must be an object"],
    ),
    (
        "text part missing text",
        _req(messages=[{"role": "user", "content": [{"type": "text"}]}]),
        ["missing 'text'"],
    ),
    (
        "unsupported part type",
        _req(messages=[{"role": "user", "content": [{"type": "audio"}]}]),
        ["unsupported content part type"],
    ),
    (
        "image missing url",
        _req(
            messages=[
                {"role": "user", "content": [{"type": "image_url", "image_url": {}}]}
            ]
        ),
        ["missing a 'url'"],
    ),
    (
        "image not a data uri",
        _req(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://x/y.png"}}
                    ],
                }
            ]
        ),
        ["only base64 data"],
    ),
    (
        "tool message missing id",
        _req(messages=[{"role": "tool", "content": "x"}]),
        ["tool message requires"],
    ),
    ("tools not an array", _req(tools=5), ["'tools' must be an array"]),
    (
        "tool missing function",
        _req(tools=[{"type": "function"}]),
        ["must have a 'function' object"],
    ),
    (
        "tool function missing name",
        _req(tools=[{"type": "function", "function": {}}]),
        ["requires a 'name'"],
    ),
    (
        "tool_call missing id",
        _req(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "f", "arguments": "{}"}}],
                }
            ]
        ),
        ["requires 'id' and 'function.name'"],
    ),
    (
        "tool_calls not an array",
        _req(messages=[{"role": "assistant", "content": None, "tool_calls": 5}]),
        ["'tool_calls' must be an array"],
    ),
    ("tool_choice unsupported", _req(tool_choice=5), ["unsupported tool_choice"]),
    (
        "tool_choice object missing name",
        _req(tool_choice={"foo": 1}),
        ["requires 'function.name'"],
    ),
]


@pytest.mark.parametrize(
    "name, request_body, expected", CASES, ids=[case[0] for case in CASES]
)
def test_malformed_request_is_an_error_value(name, request_body, expected) -> None:
    result = translate_chat_request(request_body, "anthropic")
    assert result.is_error(), f"{name} should have failed to parse"
    summary = result.error.summary
    for fragment in expected:
        assert fragment in summary, f"{name}: {fragment!r} not in {summary!r}"
