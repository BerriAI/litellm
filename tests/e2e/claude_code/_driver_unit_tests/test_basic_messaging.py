"""Unit tests for the shared `run_basic_messaging_cell` helper.

These tests mock `run_claude_models_parallel` so they exercise the
helper's branching (env-missing guard, per-model pass/fail/empty-text,
streaming wire check) without spawning the real CLI. The streaming
check is the regression we care about: a proxy that buffers the
upstream stream must turn the cell red, not green.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from claude_code import _basic_messaging
from claude_code._basic_messaging import (
    MIN_STREAM_DELTA_EVENTS,
    _count_stream_event_deltas,
    run_basic_messaging_cell,
)
from claude_code.cli_driver import DriverResult


class _FakeResult:
    """Stand-in for the test's `compat_result` fixture.

    Records every `set` / `add` payload so assertions can inspect what
    the cell reported, in order, without needing the real
    `pytest_runtest_logreport` plumbing from `conftest.py`.
    """

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.single: Optional[Dict[str, Any]] = None

    def set(self, payload: Mapping[str, Any]) -> None:
        self.single = dict(payload)

    def add(self, payload: Mapping[str, Any]) -> None:
        self.rows.append(dict(payload))


def _streamed_events(n_deltas: int = 5) -> List[Dict[str, Any]]:
    """Build a stream-json event list that *looks* streamed.

    Includes `n_deltas` `stream_event` records (matching what
    `--include-partial-messages` produces) plus the usual
    `system`/`assistant`/`result` boilerplate the CLI always emits.
    """
    events: List[Dict[str, Any]] = [{"type": "system", "subtype": "init"}]
    for i in range(n_deltas):
        events.append(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "delta": {"type": "text_delta", "text": str(i)},
                },
            }
        )
    events.append(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "1\n2\n3"}]},
        }
    )
    events.append({"type": "result"})
    return events


def _buffered_events() -> List[Dict[str, Any]]:
    """Event list a buffering proxy would produce: zero `stream_event`s."""
    return [
        {"type": "system", "subtype": "init"},
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "1\n2\n3"}]},
        },
        {"type": "result"},
    ]


def _install_fake_runner(monkeypatch, *, outcomes_by_model):
    """Patch `run_claude_models_parallel` to return canned outcomes.

    Captures the kwargs the cell passed in so tests can assert on
    `extra_args` (which is how the streaming variant opts into
    `--include-partial-messages`).
    """
    captured: Dict[str, Any] = {}

    def fake(*, models, prompt, base_url, api_key, extra_args=None, **_kwargs):
        captured["models"] = list(models)
        captured["prompt"] = prompt
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["extra_args"] = list(extra_args) if extra_args else []
        return {model: outcomes_by_model[model] for model in models}

    monkeypatch.setattr(_basic_messaging, "run_claude_models_parallel", fake)
    return captured


@pytest.fixture(autouse=True)
def _proxy_env(monkeypatch):
    monkeypatch.setenv("LITELLM_PROXY_BASE_URL", "http://localhost:4000")
    monkeypatch.setenv("LITELLM_PROXY_API_KEY", "sk-test")


def test_count_stream_event_deltas_only_counts_records_with_event_payload():
    events = [
        {"type": "system"},
        {"type": "stream_event", "event": {"type": "message_start"}},
        {"type": "stream_event", "event": {"type": "content_block_delta"}},
        {"type": "stream_event"},
        {"type": "stream_event", "event": None},
        {"type": "stream_event", "event": "not-a-dict"},
        {"type": "assistant"},
        {"type": "result"},
    ]
    assert _count_stream_event_deltas(events) == 2


def test_verify_streaming_passes_when_proxy_streams(monkeypatch):
    fake_result = _FakeResult()
    model = "claude-haiku-4-5"
    outcome = DriverResult(text="1\n2\n3", events=_streamed_events(n_deltas=5))
    captured = _install_fake_runner(monkeypatch, outcomes_by_model={model: outcome})

    run_basic_messaging_cell(
        compat_result=fake_result,
        models=[model],
        prompt="Count from 1 to 5, one number per line.",
        verify_streaming=True,
    )

    assert captured["extra_args"] == ["--include-partial-messages"]
    assert fake_result.rows == [{"status": "pass"}]


def test_verify_streaming_fails_when_proxy_buffers(monkeypatch):
    fake_result = _FakeResult()
    model = "claude-haiku-4-5"
    outcome = DriverResult(text="1\n2\n3", events=_buffered_events())
    _install_fake_runner(monkeypatch, outcomes_by_model={model: outcome})

    with pytest.raises(pytest.fail.Exception):
        run_basic_messaging_cell(
            compat_result=fake_result,
            models=[model],
            prompt="Count from 1 to 5, one number per line.",
            verify_streaming=True,
        )

    assert len(fake_result.rows) == 1
    row = fake_result.rows[0]
    assert row["status"] == "fail"
    assert "stream_event" in row["error"]
    assert f"< {MIN_STREAM_DELTA_EVENTS}" in row["error"]


def test_non_streaming_variant_omits_partial_messages_flag(monkeypatch):
    """Default `verify_streaming=False` keeps the non-streaming wire identical."""
    fake_result = _FakeResult()
    model = "claude-haiku-4-5"
    outcome = DriverResult(text="pong", events=_buffered_events())
    captured = _install_fake_runner(monkeypatch, outcomes_by_model={model: outcome})

    run_basic_messaging_cell(
        compat_result=fake_result,
        models=[model],
        prompt="Reply with the single word 'pong' and nothing else.",
    )

    assert captured["extra_args"] == []
    assert fake_result.rows == [{"status": "pass"}]


def test_verify_streaming_requires_all_models_to_stream(monkeypatch):
    """If any one tier buffers, the cell fails — same all-must-pass shape as
    the non-streaming check."""
    fake_result = _FakeResult()
    outcomes = {
        "claude-haiku-4-5": DriverResult(text="ok", events=_streamed_events(5)),
        "claude-sonnet-5": DriverResult(text="ok", events=_buffered_events()),
        "claude-opus-4-8": DriverResult(text="ok", events=_streamed_events(5)),
    }
    _install_fake_runner(monkeypatch, outcomes_by_model=outcomes)

    with pytest.raises(pytest.fail.Exception):
        run_basic_messaging_cell(
            compat_result=fake_result,
            models=list(outcomes.keys()),
            prompt="Count from 1 to 5, one number per line.",
            verify_streaming=True,
        )

    statuses = [row["status"] for row in fake_result.rows]
    assert statuses == ["pass", "fail", "pass"]
