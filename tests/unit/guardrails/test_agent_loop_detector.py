"""Unit tests for AgentLoopDetector guardrail."""

from __future__ import annotations

import itertools
import json
from typing import Final

import pytest

import litellm
from litellm.exceptions import GuardrailRaisedException
from litellm.proxy.guardrails.guardrail_hooks.agent_loop_detector import (
    AgentLoopDetector,
)


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------


def _openai_turn(tool_calls: list[tuple[str, str]]) -> dict:
    """Build an OpenAI-format assistant message with the given (name, json_args) pairs."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
            for i, (name, args) in enumerate(tool_calls)
        ],
    }


def _anthropic_turn(tool_calls: list[tuple[str, dict]]) -> dict:
    """Build an Anthropic-format assistant message with tool_use content blocks."""
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": f"tu_{i}",
                "name": name,
                "input": inp,
            }
            for i, (name, inp) in enumerate(tool_calls)
        ],
    }


async def _run(detector: AgentLoopDetector, messages: list[dict]) -> None:
    """Run the guardrail; propagates GuardrailRaisedException on loop detection."""
    await detector.async_pre_call_hook(
        user_api_key_dict=None,  # type: ignore[arg-type]
        cache=None,  # type: ignore[arg-type]
        data={"messages": messages},
        call_type="completion",
    )


# ---------------------------------------------------------------------------
# Fingerprint detection — OpenAI format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fingerprint_loop_detected_openai() -> None:
    detector = AgentLoopDetector(fingerprint_threshold=3)
    messages = [
        _openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')]),
        _openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')]),
        _openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')]),
    ]
    with pytest.raises(GuardrailRaisedException, match="AgentLoopDetector"):
        await _run(detector, messages)


@pytest.mark.asyncio
async def test_fingerprint_loop_not_triggered_below_threshold() -> None:
    detector = AgentLoopDetector(fingerprint_threshold=3)
    messages = [
        _openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')]),
        _openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')]),
    ]
    # Only 2 repetitions, threshold is 3 — should not raise
    await _run(detector, messages)


@pytest.mark.asyncio
async def test_no_loop_with_varied_calls() -> None:
    detector = AgentLoopDetector(fingerprint_threshold=3)
    messages = [
        _openai_turn([("read_file", '{"path": "/tmp/a.txt"}')]),
        _openai_turn([("write_file", '{"path": "/tmp/b.txt", "content": "hello"}')]),
        _openai_turn([("run_tests", "{}")]),
        _openai_turn([("read_file", '{"path": "/tmp/c.txt"}')]),
    ]
    await _run(detector, messages)


# ---------------------------------------------------------------------------
# Fingerprint detection — Anthropic format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fingerprint_loop_detected_anthropic() -> None:
    detector = AgentLoopDetector(fingerprint_threshold=3)
    messages = [
        _anthropic_turn([("bash", {"command": "ls -la"})]),
        _anthropic_turn([("bash", {"command": "ls -la"})]),
        _anthropic_turn([("bash", {"command": "ls -la"})]),
    ]
    with pytest.raises(GuardrailRaisedException, match="AgentLoopDetector"):
        await _run(detector, messages)


@pytest.mark.asyncio
async def test_different_args_not_a_loop_anthropic() -> None:
    detector = AgentLoopDetector(fingerprint_threshold=3)
    messages = [
        _anthropic_turn([("bash", {"command": "ls -la /tmp"})]),
        _anthropic_turn([("bash", {"command": "ls -la /home"})]),
        _anthropic_turn([("bash", {"command": "ls -la /var"})]),
    ]
    # Same tool name but different args -> different fingerprints -> no loop
    await _run(detector, messages)


# ---------------------------------------------------------------------------
# Sequence detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequence_loop_detected() -> None:
    # Set fingerprint_threshold high so only sequence detection fires.
    detector = AgentLoopDetector(fingerprint_threshold=10, sequence_min_len=2)
    # Pattern: read_file -> write_file, repeated 2x consecutively (different args
    # each time so fingerprint detection won't trigger).
    messages = [
        _openai_turn([("read_file", '{"path": "/tmp/a.txt"}')]),
        _openai_turn([("write_file", '{"path": "/tmp/b.txt"}')]),
        _openai_turn([("read_file", '{"path": "/tmp/c.txt"}')]),
        _openai_turn([("write_file", '{"path": "/tmp/d.txt"}')]),
    ]
    with pytest.raises(GuardrailRaisedException, match="AgentLoopDetector"):
        await _run(detector, messages)


@pytest.mark.asyncio
async def test_sequence_no_loop_non_consecutive() -> None:
    detector = AgentLoopDetector(fingerprint_threshold=10, sequence_min_len=2)
    # Same two tools appear twice but NOT as consecutive repetitions.
    messages = [
        _openai_turn([("read_file", "{}")]),
        _openai_turn([("write_file", "{}")]),
        _openai_turn([("run_tests", "{}")]),  # breaks the sequence
        _openai_turn([("read_file", "{}")]),
        _openai_turn([("write_file", "{}")]),
    ]
    await _run(detector, messages)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_messages_no_error() -> None:
    detector = AgentLoopDetector()
    await _run(detector, [])


@pytest.mark.asyncio
async def test_no_tool_calls_in_messages() -> None:
    detector = AgentLoopDetector()
    messages = [
        {"role": "user", "content": "Fix this bug."},
        {"role": "assistant", "content": "Sure, let me look at it."},
    ]
    await _run(detector, messages)


@pytest.mark.asyncio
async def test_single_tool_call_no_error() -> None:
    detector = AgentLoopDetector()
    messages = [_openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')])]
    await _run(detector, messages)


@pytest.mark.asyncio
async def test_configurable_threshold() -> None:
    detector = AgentLoopDetector(fingerprint_threshold=5, sequence_min_len=6, sequence_max_len=6)
    turn = _openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')])
    # 4 repetitions — should not trigger with threshold=5
    await _run(detector, [turn] * 4)
    # 5 repetitions — should trigger
    with pytest.raises(GuardrailRaisedException, match="AgentLoopDetector"):
        await _run(detector, [turn] * 5)


@pytest.mark.asyncio
async def test_window_size_limits_inspection() -> None:
    detector = AgentLoopDetector(fingerprint_threshold=3, window_size=2)
    # 3 identical calls exist, but window_size=2 means only the last 2 are
    # inspected, so the count stays at 2 (below threshold=3).
    messages = [
        _openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')]),
        _openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')]),
        _openai_turn([("read_file", '{"path": "/tmp/foo.txt"}')]),
    ]
    await _run(detector, messages)


@pytest.mark.asyncio
async def test_fingerprint_extraction_openai() -> None:
    """Unit test for the internal _extract_tool_fingerprints helper."""
    detector = AgentLoopDetector()
    messages = [
        _openai_turn([("tool_a", '{"x": 1}'), ("tool_b", '{"y": 2}')]),
        _openai_turn([("tool_c", '{"z": 3}')]),
    ]
    pairs = detector._extract_tool_fingerprints(messages)
    names = [n for n, _ in pairs]
    assert names == ["tool_a", "tool_b", "tool_c"]


@pytest.mark.asyncio
async def test_fingerprint_extraction_anthropic() -> None:
    """Unit test for the internal _extract_tool_fingerprints helper — Anthropic."""
    detector = AgentLoopDetector()
    messages = [
        _anthropic_turn([("bash", {"command": "echo hi"})]),
        _anthropic_turn([("read_file", {"path": "/tmp/x"})]),
    ]
    pairs = detector._extract_tool_fingerprints(messages)
    names = [n for n, _ in pairs]
    assert names == ["bash", "read_file"]


def _tool_exchange(path: str) -> list[dict[str, object]]:
    return [
        _openai_turn([("read_file", json.dumps({"path": path}))]),
        {"role": "tool", "tool_call_id": "call_0", "content": f"contents of {path}"},
    ]


def _sdk_request(paths: tuple[str, ...]) -> dict[str, object]:
    messages: Final = [
        {"role": "user", "content": "Fix this bug."},
        *itertools.chain.from_iterable(_tool_exchange(path) for path in paths),
    ]
    return {
        "model": "gpt-4o-mini",
        "messages": messages,
        "mock_response": "ok",
        "guardrails": ["agent-loop-detector"],
    }


@pytest.fixture
def registered_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    detector: Final = AgentLoopDetector(guardrail_name="agent-loop-detector", fingerprint_threshold=3)
    monkeypatch.setattr(litellm, "callbacks", [detector])


@pytest.mark.asyncio
@pytest.mark.usefixtures("registered_detector")
async def test_acompletion_halts_a_tool_call_loop_when_a_guardrails_list_is_passed() -> None:
    with pytest.raises(GuardrailRaisedException, match="loop detected"):
        await litellm.acompletion(**_sdk_request(("/tmp/foo.txt",) * 3))


@pytest.mark.asyncio
@pytest.mark.usefixtures("registered_detector")
async def test_acompletion_lets_varied_tool_calls_through_when_a_guardrails_list_is_passed() -> None:
    response: Final = await litellm.acompletion(**_sdk_request(("/tmp/a.txt", "/tmp/b.txt", "/tmp/c.txt")))
    assert response.choices[0].message.content == "ok", response
