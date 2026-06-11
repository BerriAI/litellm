"""The fail-closed contract of the OpenAI-chat inbound parser.

Every inbound field is accounted for: a field outside the supported surface
comes back as a typed ``unsupported`` error (the seam falls back to v1), a
malformed known field as a ``boundary`` error listing every failure, and
nothing raises. These pin the audit-F1 guarantee: flag-on can never silently
drop prompt caching, thinking, structured outputs, or any other feature.
"""

import pytest

from litellm.translation.inbound.openai_chat import parse_request

MODEL = "claude-sonnet-4-5"
_USER = {"role": "user", "content": "hi"}


def _req(**overrides) -> dict:
    return {"model": MODEL, "messages": [_USER], **overrides}


UNSUPPORTED_CASES = [
    ("unknown top-level field", _req(seed=42), ["seed"]),
    ("several unknown fields", _req(logit_bias={}, n=2), ["logit_bias", "n"]),
    ("web_search_options", _req(web_search_options={}), ["web_search_options"]),
    ("stream_options", _req(stream_options={"include_usage": True}), ["stream_options"]),
    (
        "unknown role",
        _req(messages=[{"role": "developer", "content": "x"}]),
        ["messages.0"],
    ),
    (
        "unknown content part type",
        _req(messages=[{"role": "user", "content": [{"type": "input_audio"}]}]),
        ["messages.0"],
    ),
    (
        "unknown message field",
        _req(messages=[{"role": "user", "content": "x", "weird": 1}]),
        ["weird"],
    ),
    (
        "http image needs v1 download",
        _req(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "http://x/y.png"}}
                    ],
                }
            ]
        ),
        ["http://"],
    ),
    (
        "legacy function_call",
        _req(
            messages=[
                {
                    "role": "assistant",
                    "function_call": {"name": "f", "arguments": "{}"},
                }
            ]
        ),
        ["function_call"],
    ),
    (
        "server tool calls",
        _req(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "srvtoolu_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                }
            ]
        ),
        ["srvtoolu"],
    ),
    (
        "malformed tool arguments need v1 repair",
        _req(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{not json"},
                        }
                    ],
                }
            ]
        ),
        ["repair"],
    ),
    (
        "legacy schema definitions",
        _req(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {"type": "object", "definitions": {}},
                    },
                }
            ]
        ),
        ["defs"],
    ),
    (
        "tool defer_loading",
        _req(
            tools=[
                {
                    "type": "function",
                    "function": {"name": "f"},
                    "defer_loading": True,
                }
            ]
        ),
        ["defer_loading"],
    ),
]


BOUNDARY_CASES = [
    ("missing model and messages", {}, ["model", "messages"]),
    ("non-object message", _req(messages=[123]), ["messages"]),
    ("content wrong type", _req(messages=[{"role": "user", "content": 123}]), ["content"]),
    (
        "text part missing text",
        _req(messages=[{"role": "user", "content": [{"type": "text"}]}]),
        ["text"],
    ),
    (
        "tool message missing id",
        _req(messages=[{"role": "tool", "content": "x"}]),
        ["tool_call_id"],
    ),
    ("tools not an array", _req(tools=5), ["tools"]),
    (
        "tool_call missing id",
        _req(
            messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"type": "function", "function": {"name": "f"}}],
                }
            ]
        ),
        ["id"],
    ),
    ("temperature wrong type", _req(temperature="hot"), ["temperature"]),
    ("stop wrong element type", _req(stop=[1]), ["stop"]),
    (
        "image not a data uri",
        _req(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "zzz"}}],
                }
            ]
        ),
        ["base64"],
    ),
]


@pytest.mark.parametrize(
    "name, body, fragments",
    UNSUPPORTED_CASES,
    ids=[case[0] for case in UNSUPPORTED_CASES],
)
def test_outside_surface_is_typed_unsupported(name, body, fragments) -> None:
    result = parse_request(body)
    assert result.is_error(), f"{name} should not parse"
    assert result.error.tag == "unsupported", f"{name}: {result.error.summary}"
    for fragment in fragments:
        assert fragment in result.error.summary, (name, result.error.summary)


@pytest.mark.parametrize(
    "name, body, fragments",
    BOUNDARY_CASES,
    ids=[case[0] for case in BOUNDARY_CASES],
)
def test_malformed_known_field_is_boundary_error(name, body, fragments) -> None:
    result = parse_request(body)
    assert result.is_error(), f"{name} should not parse"
    summary = result.error.summary
    for fragment in fragments:
        assert fragment in summary, (name, summary)


def test_explicit_top_level_nulls_match_absent() -> None:
    explicit = parse_request(_req(temperature=None, tools=None, seed=None))
    absent = parse_request(_req())
    assert explicit.is_ok() and absent.is_ok()
    assert explicit.ok == absent.ok


def test_ignored_fields_match_v1_observable_behavior() -> None:
    """v1 reads none of these, so accepting-and-ignoring equals v1 exactly."""
    result = parse_request(
        _req(
            messages=[
                {"role": "user", "content": "hi", "name": "alice"},
                {
                    "role": "assistant",
                    "content": "yo",
                    "refusal": None,
                    "annotations": [],
                    "audio": None,
                },
            ]
        )
    )
    assert result.is_ok(), result.error.summary if result.is_error() else None


def test_consecutive_same_role_messages_merge() -> None:
    result = parse_request(
        _req(
            messages=[
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "assistant", "content": "c"},
                {"role": "assistant", "content": "d"},
            ]
        )
    )
    assert result.is_ok()
    messages = result.ok.messages
    assert [m.role for m in messages] == ["user", "assistant"]
    assert len(messages[0].content) == 2
    assert len(messages[1].content) == 2


def test_tool_messages_merge_into_user_turn() -> None:
    result = parse_request(
        _req(
            messages=[
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"x": 1}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                {"role": "tool", "tool_call_id": "c1b", "content": "ok2"},
            ]
        )
    )
    assert result.is_ok()
    messages = result.ok.messages
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert [b.tag for b in messages[2].content] == ["tool_result", "tool_result"]
    arguments = messages[1].content[0].tool_use.arguments.value
    assert arguments == {"x": 1}


def test_non_function_tool_calls_skipped_like_v1() -> None:
    result = parse_request(
        _req(
            messages=[
                {
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": [
                        {"id": "c1", "type": "custom"},
                    ],
                }
            ]
        )
    )
    assert result.is_ok()
    assert [b.tag for b in result.ok.messages[0].content] == ["text"]
