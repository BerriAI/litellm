"""Unit tests for the Claude Code CLI Driver.

These tests mock the subprocess so they run anywhere — no network, no
`claude` install, no API keys. They cover the behavior contract:
argument assembly, environment overlay, stream-JSON parsing, exit-code
plumbing, and the structured failure modes (CLI not found, timeout).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import pytest

from tests.claude_code.cli_driver import (
    ClaudeCLIError,
    DriverResult,
    failure_diagnostic,
    run_claude,
)


@dataclass
class _Completed:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _make_runner(*, stdout: str = "", returncode: int = 0, stderr: str = ""):
    captured = {}

    def runner(cmd, env, capture_output, text, timeout, check):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["timeout"] = timeout
        return _Completed(returncode=returncode, stdout=stdout, stderr=stderr)

    return runner, captured


def test_run_claude_assembles_command_correctly():
    runner, captured = _make_runner(
        stdout='{"type":"assistant","message":{"content":[{"type":"text","text":"ok"}]}}\n'
    )
    run_claude(
        prompt="hello",
        model="claude-haiku-4-5",
        base_url="http://localhost:4000",
        api_key="sk-test",
        runner=runner,
    )
    cmd = captured["cmd"]
    assert cmd[0] == "claude"
    assert "--print" in cmd
    assert "--output-format" in cmd
    assert "stream-json" in cmd
    assert "--model" in cmd
    assert "claude-haiku-4-5" in cmd
    assert cmd[-1] == "hello"


def test_run_claude_overlays_proxy_env():
    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-7",
        base_url="http://proxy.example:4000",
        api_key="sk-abc",
        runner=runner,
    )
    env = captured["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://proxy.example:4000"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-abc"


def test_run_claude_extra_env_takes_precedence_over_os_environ(monkeypatch):
    monkeypatch.setenv("FOO", "from-os")
    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-7",
        base_url="http://localhost",
        api_key="sk-abc",
        extra_env={"FOO": "from-arg"},
        runner=runner,
    )
    assert captured["env"]["FOO"] == "from-arg"


def test_run_claude_parses_stream_json_assistant_text():
    events = [
        {"type": "system", "session_id": "abc"},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ]
            },
        },
        {"type": "result", "usage": {"input_tokens": 10, "output_tokens": 2}},
    ]
    stdout = "\n".join(json.dumps(e) for e in events) + "\n"
    runner, _ = _make_runner(stdout=stdout)
    result = run_claude(
        prompt="hi",
        model="claude-haiku-4-5",
        base_url="http://localhost",
        api_key="sk-abc",
        runner=runner,
    )
    assert isinstance(result, DriverResult)
    assert result.text == "Hello world"
    assert len(result.events) == 3
    assert result.usage == {"input_tokens": 10, "output_tokens": 2}
    assert result.exit_code == 0


def test_run_claude_handles_string_message_content():
    """Some CLI versions emit `message.content` as a plain string."""
    stdout = (
        json.dumps({"type": "assistant", "message": {"content": "bare text"}}) + "\n"
    )
    runner, _ = _make_runner(stdout=stdout)
    result = run_claude(
        prompt="hi",
        model="m",
        base_url="http://x",
        api_key="k",
        runner=runner,
    )
    assert result.text == "bare text"


def test_run_claude_skips_malformed_lines():
    stdout = (
        "not-json\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "x"}]},
            }
        )
        + "\n"
        + "{also-bad\n"
    )
    runner, _ = _make_runner(stdout=stdout)
    result = run_claude(
        prompt="hi",
        model="m",
        base_url="http://x",
        api_key="k",
        runner=runner,
    )
    assert result.text == "x"
    assert len(result.events) == 1


def test_run_claude_propagates_nonzero_exit_code():
    runner, _ = _make_runner(stdout="", returncode=2, stderr="auth failed")
    result = run_claude(
        prompt="hi",
        model="m",
        base_url="http://x",
        api_key="k",
        runner=runner,
    )
    assert result.exit_code == 2
    assert result.stderr == "auth failed"
    assert result.text == ""


def test_run_claude_raises_on_missing_cli():
    def runner(*args, **kwargs):
        raise FileNotFoundError(2, "no such file", "claude")

    with pytest.raises(ClaudeCLIError, match="claude CLI not found"):
        run_claude(
            prompt="hi",
            model="m",
            base_url="http://x",
            api_key="k",
            runner=runner,
        )


def test_run_claude_raises_on_timeout():
    def runner(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    with pytest.raises(ClaudeCLIError, match="timed out"):
        run_claude(
            prompt="hi",
            model="m",
            base_url="http://x",
            api_key="k",
            timeout=1,
            runner=runner,
        )


def test_run_claude_validates_required_params():
    runner, _ = _make_runner()
    with pytest.raises(ValueError, match="prompt"):
        run_claude(
            prompt="",
            model="m",
            base_url="http://x",
            api_key="k",
            runner=runner,
        )
    with pytest.raises(ValueError, match="model"):
        run_claude(
            prompt="hi",
            model="",
            base_url="http://x",
            api_key="k",
            runner=runner,
        )
    with pytest.raises(ValueError, match="base_url"):
        run_claude(
            prompt="hi",
            model="m",
            base_url="",
            api_key="k",
            runner=runner,
        )
    with pytest.raises(ValueError, match="api_key"):
        run_claude(
            prompt="hi",
            model="m",
            base_url="http://x",
            api_key="",
            runner=runner,
        )


# ---------------------------------------------------------------------------
# failure_diagnostic
#
# Regression coverage for the bring-up incident where the proxy was started
# with the wrong config and tests reported only `claude CLI exited 1` while
# the actual 400 from LiteLLM was sitting in stdout. The helper must surface
# api_status, the assistant text (where API errors land), stderr, and the
# exit code together — and gracefully degrade when individual pieces are
# missing.
# ---------------------------------------------------------------------------


def test_failure_diagnostic_surfaces_api_error_text_from_stdout():
    """The CLI hides 4xx/5xx from the proxy in `assistant.message.content` text."""
    api_error_text = (
        'API Error: 400 {"error":{"message":"litellm.BadRequestError: '
        "You passed in model=claude-haiku-4-5. There are no healthy "
        'deployments..."}}'
    )
    result = DriverResult(
        text=api_error_text,
        events=[
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": api_error_text}]},
            },
            {
                "type": "result",
                "is_error": True,
                "api_error_status": 400,
                "result": api_error_text,
            },
        ],
        exit_code=1,
        stderr="",
    )

    diag = failure_diagnostic(result)

    assert "exit=1" in diag
    assert "api_status=400" in diag
    assert "There are no healthy deployments" in diag


def test_failure_diagnostic_falls_back_to_stderr_when_no_text():
    result = DriverResult(text="", events=[], exit_code=2, stderr="boom\n")
    diag = failure_diagnostic(result)
    assert "exit=2" in diag
    assert "stderr=boom" in diag


def test_failure_diagnostic_handles_completely_empty_result():
    """A run that produced literally nothing should still yield a useful string."""
    result = DriverResult(text="", events=[], exit_code=137, stderr="")
    diag = failure_diagnostic(result)
    assert "exit=137" in diag
    assert "no diagnostic output" in diag


def test_failure_diagnostic_truncates_long_text():
    """Don't let a 5MB HTML 502 page from a load balancer wreck the matrix JSON."""
    huge = "x" * 5000
    result = DriverResult(text=huge, events=[], exit_code=1, stderr="")
    diag = failure_diagnostic(result, max_len=100)
    assert "truncated" in diag
    # Allow some slack for the prefix/suffix/separator characters.
    assert len(diag) < 300


def test_failure_diagnostic_ignores_non_int_api_error_status():
    """The CLI sometimes emits api_error_status as a string; don't crash."""
    result = DriverResult(
        text="oops",
        events=[{"type": "result", "api_error_status": "n/a"}],
        exit_code=1,
        stderr="",
    )
    diag = failure_diagnostic(result)
    assert "api_status" not in diag
    assert "text=oops" in diag


def test_failure_diagnostic_uses_last_result_event_status():
    """If multiple `result` events appear, the most recent status wins."""
    result = DriverResult(
        text="",
        events=[
            {"type": "result", "api_error_status": 500},
            {"type": "assistant", "message": {"content": []}},
            {"type": "result", "api_error_status": 429},
        ],
        exit_code=1,
        stderr="",
    )
    diag = failure_diagnostic(result)
    assert "api_status=429" in diag
    assert "500" not in diag
