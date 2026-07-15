"""Unit tests for the shared `run_basic_messaging_cell` helper.

These tests inject a fake ``ClaudeRunner`` and a fake env mapping so
they exercise the helper's branching (env-missing guard, per-model
pass/fail/empty-text, streaming wire check) without spawning the real
CLI or touching ``os.environ``. The streaming check is the regression
we care about: a proxy that buffers the upstream stream must turn the
cell red, not green.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import pytest

from claude_code._basic_messaging import (
    MIN_STREAM_DELTA_EVENTS,
    _count_stream_event_deltas,
    run_basic_messaging_cell,
)
from claude_code.cli_driver import DriverResult


_MASTER_KEY = "sk-master-must-not-leak"
_COMPAT_CLI_KEY = "sk-compat-cli-session-key"

_PROXY_ENV: Mapping[str, str] = {
    "LITELLM_PROXY_URL": "http://localhost:4000",
    "LITELLM_MASTER_KEY": _MASTER_KEY,
}


def _static_cli_key_provider() -> str:
    return _COMPAT_CLI_KEY


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


def _make_fake_runner(*, outcomes_by_model):
    """Build an injectable runner that returns canned outcomes and
    records the kwargs the helper passed in.

    Returns a ``(runner, captured)`` pair; ``captured`` is a dict the
    test can assert against without any global mutation, which is why
    we prefer DI over ``monkeypatch.setattr``: the helper takes a
    ``runner=`` kwarg, so tests bind their fake directly."""
    captured: Dict[str, Any] = {}

    def runner(*, models, prompt, base_url, api_key, extra_args=None, **_kwargs):
        captured["models"] = list(models)
        captured["prompt"] = prompt
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        captured["extra_args"] = list(extra_args) if extra_args else []
        return {model: outcomes_by_model[model] for model in models}

    return runner, captured


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


def test_verify_streaming_passes_when_proxy_streams():
    fake_result = _FakeResult()
    model = "claude-haiku-4-5"
    outcome = DriverResult(text="1\n2\n3", events=_streamed_events(n_deltas=5))
    runner, captured = _make_fake_runner(outcomes_by_model={model: outcome})

    run_basic_messaging_cell(
        compat_result=fake_result,
        models=[model],
        prompt="Count from 1 to 5, one number per line.",
        verify_streaming=True,
        env=_PROXY_ENV,
        runner=runner,
        cli_key_provider=_static_cli_key_provider,
    )

    assert captured["extra_args"] == ["--include-partial-messages"]
    assert fake_result.rows == [{"status": "pass"}]


def test_verify_streaming_fails_when_proxy_buffers():
    fake_result = _FakeResult()
    model = "claude-haiku-4-5"
    outcome = DriverResult(text="1\n2\n3", events=_buffered_events())
    runner, _captured = _make_fake_runner(outcomes_by_model={model: outcome})

    with pytest.raises(pytest.fail.Exception):
        run_basic_messaging_cell(
            compat_result=fake_result,
            models=[model],
            prompt="Count from 1 to 5, one number per line.",
            verify_streaming=True,
            env=_PROXY_ENV,
            runner=runner,
            cli_key_provider=_static_cli_key_provider,
        )

    assert len(fake_result.rows) == 1
    row = fake_result.rows[0]
    assert row["status"] == "fail"
    assert "stream_event" in row["error"]
    assert f"< {MIN_STREAM_DELTA_EVENTS}" in row["error"]


def test_non_streaming_variant_omits_partial_messages_flag():
    """Default `verify_streaming=False` keeps the non-streaming wire identical."""
    fake_result = _FakeResult()
    model = "claude-haiku-4-5"
    outcome = DriverResult(text="pong", events=_buffered_events())
    runner, captured = _make_fake_runner(outcomes_by_model={model: outcome})

    run_basic_messaging_cell(
        compat_result=fake_result,
        models=[model],
        prompt="Reply with the single word 'pong' and nothing else.",
        env=_PROXY_ENV,
        runner=runner,
        cli_key_provider=_static_cli_key_provider,
    )

    assert captured["extra_args"] == []
    assert fake_result.rows == [{"status": "pass"}]


def test_verify_streaming_requires_all_models_to_stream():
    """If any one tier buffers, the cell fails — same all-must-pass shape as
    the non-streaming check."""
    fake_result = _FakeResult()
    outcomes = {
        "claude-haiku-4-5": DriverResult(text="ok", events=_streamed_events(5)),
        "claude-sonnet-4-6": DriverResult(text="ok", events=_buffered_events()),
        "claude-opus-4-7": DriverResult(text="ok", events=_streamed_events(5)),
    }
    runner, _captured = _make_fake_runner(outcomes_by_model=outcomes)

    with pytest.raises(pytest.fail.Exception):
        run_basic_messaging_cell(
            compat_result=fake_result,
            models=list(outcomes.keys()),
            prompt="Count from 1 to 5, one number per line.",
            verify_streaming=True,
            env=_PROXY_ENV,
            runner=runner,
            cli_key_provider=_static_cli_key_provider,
        )

    statuses = [row["status"] for row in fake_result.rows]
    assert statuses == ["pass", "fail", "pass"]


def test_missing_proxy_env_hard_fails_regardless_of_runner():
    """The env guard fires before the runner is called, and takes the
    env from the injected mapping (not os.environ). Passing an empty
    env dict must hard-fail even if a happy runner is bound."""
    fake_result = _FakeResult()
    runner, captured = _make_fake_runner(outcomes_by_model={})

    with pytest.raises(pytest.fail.Exception):
        run_basic_messaging_cell(
            compat_result=fake_result,
            models=["claude-haiku-4-5"],
            prompt="whatever",
            env={},
            runner=runner,
            cli_key_provider=_static_cli_key_provider,
        )

    assert captured == {}, "runner must not be called when env resolution fails"


def test_missing_cli_key_hard_fails_without_falling_back_to_master_key():
    """SECURITY-critical: if the compat fixture never bound a CLI key
    (returns ``None``), the helper must hard-fail rather than reach for
    the master key from env. A silent fallback would leak admin
    capabilities to the ``claude`` CLI subprocess."""
    fake_result = _FakeResult()
    runner, captured = _make_fake_runner(outcomes_by_model={})

    with pytest.raises(pytest.fail.Exception) as excinfo:
        run_basic_messaging_cell(
            compat_result=fake_result,
            models=["claude-haiku-4-5"],
            prompt="whatever",
            env=_PROXY_ENV,
            runner=runner,
            cli_key_provider=lambda: None,
        )

    assert "compat CLI key" in str(excinfo.value)
    assert captured == {}, (
        "runner must not be called (and thus never receive the master "
        "key) when the CLI key provider returns None"
    )


def test_runner_receives_compat_cli_key_not_master_key():
    """SECURITY-critical: the ``api_key`` handed to the CLI runner must
    be the fixture-minted compat CLI key, never the master key from
    env. This is the invariant that keeps a compromised ``claude`` CLI
    package from getting proxy admin capabilities."""
    fake_result = _FakeResult()
    model = "claude-haiku-4-5"
    outcome = DriverResult(text="ok", events=_buffered_events())
    runner, captured = _make_fake_runner(outcomes_by_model={model: outcome})

    run_basic_messaging_cell(
        compat_result=fake_result,
        models=[model],
        prompt="whatever",
        env=_PROXY_ENV,
        runner=runner,
        cli_key_provider=_static_cli_key_provider,
    )

    assert captured["api_key"] == _COMPAT_CLI_KEY
    assert captured["api_key"] != _MASTER_KEY, (
        "api_key handed to runner must be the compat CLI key, not the "
        "master key - master key would give a compromised CLI admin "
        "access to the proxy"
    )
