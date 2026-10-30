import sys
from dataclasses import replace
from io import BytesIO, TextIOWrapper
from typing import Final

import pytest
from pydantic import JsonValue

from litellm.proxy.mcp_server.code_worker import (
    WORKER_MESSAGE,
    Call,
    CodeExecutionLimits,
    Done,
    Reply,
    Start,
    encode_message,
    load_json,
    main,
)

pytestmark: Final = pytest.mark.usefixtures("mcp_code_runtime")


def run_worker(monkeypatch: pytest.MonkeyPatch, request: bytes) -> list[Call | Done]:
    source: Final = TextIOWrapper(BytesIO(request))
    destination: Final = TextIOWrapper(BytesIO())
    monkeypatch.setattr(sys, "stdin", source)
    monkeypatch.setattr(sys, "stdout", destination)
    main()
    return [
        WORKER_MESSAGE.validate_python(load_json(line.decode())) for line in destination.buffer.getvalue().splitlines()
    ]


@pytest.mark.parametrize(
    ("codepoint", "expected"), [(0xD800, "\ud800"), (0xDC00, "\udc00"), (0x1F680, "🚀"), (0x732B, "猫")]
)
def test_native_worker_preserves_unicode_output(monkeypatch: pytest.MonkeyPatch, codepoint: int, expected: str) -> None:
    limits: Final = CodeExecutionLimits()
    request: Final = Start(code=f"text('kept'); text(String.fromCodePoint({codepoint}));", limits=limits)
    assert run_worker(monkeypatch, encode_message(request, limits.message_bytes)) == [Done(output=("kept", expected))]


def test_native_worker_roundtrips_surrogates_in_tool_arguments_and_results(monkeypatch: pytest.MonkeyPatch) -> None:
    limits: Final = CodeExecutionLimits()
    request: Final = Start(
        code="text('kept'); text(await tools.call_tool({tool_id:'echo',arguments:{value:String.fromCharCode(0xd800)}}));",
        limits=limits,
    )
    value: Final[dict[str, JsonValue]] = {"value": "\ud800"}
    messages: Final = run_worker(
        monkeypatch,
        encode_message(request, limits.message_bytes) + encode_message(Reply(id=0, value=value), limits.message_bytes),
    )
    assert messages == [Call(id=0, tool_id="echo", arguments=value), Done(output=("kept", value))]


def test_native_worker_chains_full_tool_results(monkeypatch: pytest.MonkeyPatch) -> None:
    limits: Final = CodeExecutionLimits()
    request: Final = Start(
        code="""
        const first = await tools.call_tool({tool_id: 'lookup', arguments: {id: '9007199254740993'}});
        text(await tools.call_tool({tool_id: 'echo', arguments: first.structuredContent}));
        """,
        limits=limits,
    )
    envelope: Final[dict[str, JsonValue]] = {
        "content": [{"type": "text", "text": "kept"}],
        "structuredContent": {"id": "9007199254740993"},
        "isError": True,
    }
    messages: Final = run_worker(
        monkeypatch,
        b"".join(
            encode_message(message, limits.message_bytes)
            for message in (request, Reply(id=0, value=envelope), Reply(id=1, value=envelope))
        ),
    )
    assert messages == [
        Call(id=0, tool_id="lookup", arguments={"id": "9007199254740993"}),
        Call(id=1, tool_id="echo", arguments={"id": "9007199254740993"}),
        Done(output=(envelope,)),
    ]


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ('text("before"); throw new Error("script failed")', "script failed"),
        ("text(BigInt(1))", "json values"),
        ("text(9007199254740992)", "integer"),
        ("text(() => 1)", "json values"),
    ],
)
def test_native_worker_reports_script_errors(monkeypatch: pytest.MonkeyPatch, code: str, error: str) -> None:
    limits: Final = CodeExecutionLimits()
    messages: Final = run_worker(monkeypatch, encode_message(Start(code=code, limits=limits), limits.message_bytes))
    result: Final = messages[-1]
    assert isinstance(result, Done)
    assert result.error is not None and error in result.error.lower()
    assert result.output == (("before",) if "before" in code else ())


def test_native_worker_rejects_oversized_output(monkeypatch: pytest.MonkeyPatch) -> None:
    limits: Final = replace(CodeExecutionLimits(), output_bytes=32)
    result: Final = run_worker(
        monkeypatch, encode_message(Start(code='text("猫".repeat(50))', limits=limits), limits.message_bytes)
    )[-1]
    assert isinstance(result, Done)
    assert result.error is not None and "byte limit" in result.error
    assert result.output == ()


@pytest.mark.parametrize(
    "reply", [b"", b"not json\n", b'{"id":0,"value":{"number":NaN}}\n', b'{"id":0,"value":{"id":9007199254740993}}\n']
)
def test_native_worker_fails_closed_on_bad_parent_reply(monkeypatch: pytest.MonkeyPatch, reply: bytes) -> None:
    limits: Final = CodeExecutionLimits()
    request: Final = Start(code="text(await tools.call_tool({tool_id:'lookup', arguments:{}}))", limits=limits)
    messages: Final = run_worker(monkeypatch, encode_message(request, limits.message_bytes) + reply)
    assert messages[0] == Call(id=0, tool_id="lookup", arguments={})
    assert messages[-1] == Done(error="JavaScript worker failed")


def test_native_worker_rejects_oversized_start(monkeypatch: pytest.MonkeyPatch) -> None:
    assert run_worker(monkeypatch, b"x" * (CodeExecutionLimits().message_bytes + 1)) == [
        Done(error="JavaScript worker failed")
    ]


def test_native_worker_limits_outgoing_tool_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    limits: Final = replace(CodeExecutionLimits(), message_bytes=128)
    request: Final = Start(
        code="await tools.call_tool({tool_id:'echo', arguments:{text:'x'.repeat(256)}})", limits=limits
    )
    messages: Final = run_worker(monkeypatch, encode_message(request, CodeExecutionLimits().message_bytes))
    assert len(messages) == 1 and isinstance(messages[0], Done)
    assert messages[0].error is not None
