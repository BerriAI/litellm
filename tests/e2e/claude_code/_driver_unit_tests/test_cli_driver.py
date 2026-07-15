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

from claude_code.cli_driver import (
    ClaudeCLIError,
    DriverResult,
    failure_diagnostic,
    run_claude,
    run_claude_models_parallel,
)


@dataclass
class _Completed:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def _make_runner(*, stdout: str = "", returncode: int = 0, stderr: str = ""):
    captured = {}

    def runner(cmd, env, capture_output, text, timeout, check, input=None):
        captured["cmd"] = cmd
        captured["env"] = env
        captured["timeout"] = timeout
        captured["input"] = input
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
    # prompt is the last positional after the `--` end-of-options marker.
    assert cmd[-2:] == ["--", "hello"]


def test_run_claude_places_extra_args_before_prompt():
    """`claude --print` expects the prompt as the final positional arg.

    Flags appearing after the prompt are ignored or eaten by the prompt
    parser (especially variadic flags like `--allowed-tools <tools...>`),
    which silently broke the tool_use, vision, and web_search cells
    before the fix. Pin the ordering: every flag (including
    caller-supplied `extra_args`) must precede the `--` end-of-options
    marker, which itself precedes the prompt.
    """
    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="say hi",
        model="claude-haiku-4-5",
        base_url="http://localhost:4000",
        api_key="sk-test",
        extra_args=["--allowed-tools", "Bash"],
        runner=runner,
    )
    cmd = captured["cmd"]
    # Prompt is last, `--` immediately precedes it, and the caller's
    # extra_args sit somewhere earlier in the command.
    assert cmd[-2:] == ["--", "say hi"]
    assert "--allowed-tools" in cmd
    assert cmd.index("--allowed-tools") < cmd.index("--")


def test_run_claude_overlays_proxy_env():
    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-8",
        base_url="http://proxy.example:4000",
        api_key="sk-abc",
        runner=runner,
    )
    env = captured["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://proxy.example:4000"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-abc"


def test_run_claude_extra_env_is_added_to_subprocess_env():
    """Caller-supplied extra_env entries land on the subprocess env."""
    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-8",
        base_url="http://localhost",
        api_key="sk-abc",
        extra_env={"MAX_THINKING_TOKENS": "4096"},
        runner=runner,
    )
    assert captured["env"]["MAX_THINKING_TOKENS"] == "4096"


def test_run_claude_inherits_only_allowlisted_os_environ(monkeypatch):
    """Process-runtime vars (PATH) flow through; credentials don't.

    The `claude` CLI is a Node binary installed dynamically from npm in
    CI. If the package were ever compromised, inheriting the entire
    parent environment would hand it every credential the surrounding
    proxy job loads (AWS keys, Azure Foundry key, GitHub token, etc.).
    Pin the contract: only the small allowlist of runtime vars is
    inherited; everything else is dropped unless the caller passes it
    explicitly via extra_env.

    `HOME` is *not* on the allowlist anymore — see the dedicated
    isolated-HOME test below for the reason.
    """
    monkeypatch.setenv("PATH", "/usr/bin:/usr/local/bin")
    monkeypatch.setenv("HOME", "/home/runner")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "totally-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-proxy-only")
    monkeypatch.setenv("AZURE_FOUNDRY_API_KEY", "azure-secret")
    monkeypatch.setenv("VERTEXAI_CREDENTIALS", '{"private_key": "leak"}')
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_xxx")

    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-8",
        base_url="http://localhost",
        api_key="sk-abc",
        runner=runner,
    )
    env = captured["env"]
    assert env["PATH"] == "/usr/bin:/usr/local/bin"
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "AZURE_FOUNDRY_API_KEY" not in env
    assert "VERTEXAI_CREDENTIALS" not in env
    assert "GITHUB_TOKEN" not in env


def test_run_claude_uses_isolated_per_invocation_home(monkeypatch, tmp_path):
    """`claude` subprocess never sees the runtime user's real $HOME.

    The CLI needs *a* HOME (it caches per-session state under
    `$HOME/.claude/projects/<sha>/`), but it has no business reading
    the runtime user's real one. On the cron VM the runtime user is a
    real interactive account with a populated home directory
    (~/.config/gh/hosts.yml carrying a GitHub token, ~/.ssh/, etc.);
    handing /home/mateo to a compromised npm package — or to a
    model-directed `Read` tool call during the PDF/vision cells —
    would let it exfiltrate those files. We hand the CLI a fresh
    empty per-invocation tmpdir instead.
    """
    monkeypatch.setenv("HOME", "/home/runner")

    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-8",
        base_url="http://localhost",
        api_key="sk-abc",
        runner=runner,
    )
    env = captured["env"]
    assert "HOME" in env, "claude CLI needs HOME to find ~/.claude session dir"
    assert (
        env["HOME"] != "/home/runner"
    ), "HOME must not leak the parent process's HOME to claude"
    # The isolated HOME is a fresh tmpdir prefixed `claude-cli-home-`;
    # see `_make_isolated_home` in cli_driver.py. It exists during the
    # subprocess call and is removed afterwards (cleanup runs in a
    # `finally`, so by the time this assertion runs the dir is gone —
    # we only check the *prefix* of the path string we captured).
    assert "claude-cli-home-" in env["HOME"]


def test_run_claude_isolated_home_is_distinct_per_invocation(monkeypatch):
    """Two consecutive calls get two different isolated HOMEs.

    Reusing a single tmpdir across calls would defeat the isolation
    in the parallel matrix run (a compromised CLI could plant a file
    in HOME on one model's run and read it on the next). Pin: each
    `run_claude` invocation gets its own freshly-created HOME.
    """
    monkeypatch.setenv("HOME", "/home/runner")

    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-8",
        base_url="http://localhost",
        api_key="sk-abc",
        runner=runner,
    )
    home_a = captured["env"]["HOME"]

    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-8",
        base_url="http://localhost",
        api_key="sk-abc",
        runner=runner,
    )
    home_b = captured["env"]["HOME"]

    assert home_a != home_b


def test_run_claude_isolated_home_cleaned_up_after_run(monkeypatch):
    """The per-invocation HOME tmpdir is rm-rf'd when run_claude returns.

    Without cleanup, a long matrix run would accumulate one tmpdir
    per cell × per model × per CLI call (~75 dirs per cron run,
    growing without bound across days).
    """
    import os as _os

    monkeypatch.setenv("HOME", "/home/runner")

    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-8",
        base_url="http://localhost",
        api_key="sk-abc",
        runner=runner,
    )
    isolated_home = captured["env"]["HOME"]
    assert not _os.path.exists(
        isolated_home
    ), f"isolated HOME {isolated_home!r} should be removed after run_claude returns"


def test_run_claude_isolated_home_cleaned_up_on_subprocess_failure(monkeypatch):
    """Cleanup runs even when the CLI subprocess raises.

    If the CLI is missing or times out, `run_claude` raises
    `ClaudeCLIError` — but the per-invocation HOME tmpdir must still
    be removed (the `finally` clause), otherwise long failure-prone
    runs leak tmpdirs.
    """
    import os as _os

    monkeypatch.setenv("HOME", "/home/runner")

    captured: dict = {}

    def runner(cmd, env, capture_output, text, timeout, check, input=None):
        captured["env"] = env
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    with pytest.raises(ClaudeCLIError):
        run_claude(
            prompt="hi",
            model="claude-opus-4-8",
            base_url="http://localhost",
            api_key="sk-abc",
            runner=runner,
        )

    isolated_home = captured["env"]["HOME"]
    assert not _os.path.exists(
        isolated_home
    ), f"isolated HOME {isolated_home!r} should be removed even on timeout"


def test_run_claude_extra_env_can_pass_through_otherwise_blocked_var(monkeypatch):
    """The allowlist applies to inherited os.environ; extra_env is the
    sanctioned way for a test to opt-in to passing something extra."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-os")
    runner, captured = _make_runner(stdout="")
    run_claude(
        prompt="hi",
        model="claude-opus-4-8",
        base_url="http://localhost",
        api_key="sk-abc",
        extra_env={"ANTHROPIC_API_KEY": "from-arg"},
        runner=runner,
    )
    assert captured["env"]["ANTHROPIC_API_KEY"] == "from-arg"


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
    with pytest.raises(ValueError, match="stdin_input"):
        run_claude(
            prompt=None,
            stdin_input="",
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


# ---------------------------------------------------------------------------
# run_claude_models_parallel
#
# The matrix runs three Claude tiers per cell, so the parallel helper has to
# (a) invoke `run_claude` once per model, (b) preserve each model's outcome
# separately, and (c) return errors as values rather than raising — callers
# need both the failed and the succeeded model results to report per-cell
# rows accurately.
# ---------------------------------------------------------------------------


def test_run_claude_models_parallel_returns_one_result_per_model():
    """Each model gets its own DriverResult keyed under the helper's dict."""
    seen_models: List[str] = []

    def runner(cmd, env, capture_output, text, timeout, check, input=None):
        # The model id is two slots after `--model` in the assembled command.
        idx = cmd.index("--model")
        model = cmd[idx + 1]
        seen_models.append(model)
        return _Completed(
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": f"reply-{model}"}]
                    },
                }
            )
            + "\n",
        )

    outcomes = run_claude_models_parallel(
        models=["a", "b", "c"],
        prompt="hi",
        base_url="http://x",
        api_key="k",
        runner=runner,
    )

    assert set(outcomes.keys()) == {"a", "b", "c"}
    for model in ("a", "b", "c"):
        result = outcomes[model]
        assert isinstance(result, DriverResult)
        assert result.text == f"reply-{model}"
        assert result.exit_code == 0
    assert sorted(seen_models) == ["a", "b", "c"]


def test_run_claude_models_parallel_returns_errors_as_values():
    """A model whose CLI is missing surfaces as a ClaudeCLIError, not a raise."""

    def runner(cmd, env, capture_output, text, timeout, check, input=None):
        idx = cmd.index("--model")
        model = cmd[idx + 1]
        if model == "boom":
            raise FileNotFoundError(2, "no such file", "claude")
        return _Completed(
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "ok"}]},
                }
            )
            + "\n",
        )

    outcomes = run_claude_models_parallel(
        models=["ok-model", "boom"],
        prompt="hi",
        base_url="http://x",
        api_key="k",
        runner=runner,
    )

    assert isinstance(outcomes["ok-model"], DriverResult)
    assert outcomes["ok-model"].text == "ok"
    assert isinstance(outcomes["boom"], ClaudeCLIError)
    assert "claude CLI not found" in str(outcomes["boom"])


def test_run_claude_models_parallel_preserves_nonzero_exit_codes():
    """Mixed success/failure on exit code should not collapse into one verdict."""

    def runner(cmd, env, capture_output, text, timeout, check, input=None):
        idx = cmd.index("--model")
        model = cmd[idx + 1]
        if model == "fail":
            return _Completed(returncode=2, stdout="", stderr="auth failed")
        return _Completed(
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "assistant",
                    "message": {"content": [{"type": "text", "text": "ok"}]},
                }
            )
            + "\n",
        )

    outcomes = run_claude_models_parallel(
        models=["ok-model", "fail"],
        prompt="hi",
        base_url="http://x",
        api_key="k",
        runner=runner,
    )

    assert outcomes["ok-model"].exit_code == 0
    assert outcomes["fail"].exit_code == 2
    assert outcomes["fail"].stderr == "auth failed"


def test_run_claude_models_parallel_rejects_empty_models():
    with pytest.raises(ValueError, match="non-empty"):
        run_claude_models_parallel(
            models=[],
            prompt="hi",
            base_url="http://x",
            api_key="k",
        )


def test_run_claude_models_parallel_stamps_duration_on_each_result():
    """Each DriverResult carries the per-model wall time so callers can
    attribute slow cells without re-timing the work themselves.

    The fake runner sleeps for very different durations per model so
    we can prove each result is timing its own work (not the batch
    wall time). We use generous absolute bounds because thread-pool
    scheduling on a loaded CI box adds noise on the order of tens of
    milliseconds.
    """
    import time

    def runner(cmd, env, capture_output, text, timeout, check, input=None):
        idx = cmd.index("--model")
        model = cmd[idx + 1]
        time.sleep(0.05 if model == "fast" else 0.40)
        return _Completed(returncode=0, stdout="")

    outcomes = run_claude_models_parallel(
        models=["fast", "slow"],
        prompt="hi",
        base_url="http://x",
        api_key="k",
        runner=runner,
    )

    fast_ms = outcomes["fast"].duration_ms
    slow_ms = outcomes["slow"].duration_ms
    assert fast_ms is not None and slow_ms is not None
    # 50ms sleep ⇒ ~50–250ms after scheduling overhead; 400ms sleep ⇒
    # 400–700ms. We just need the two distributions to be non-overlapping
    # so we know each row's duration is its own work, not the batch's.
    assert fast_ms < 300, fast_ms
    assert slow_ms >= 350, slow_ms
    assert slow_ms > fast_ms


def test_run_claude_models_parallel_breakdown_logs_to_stderr(capsys):
    """The breakdown helper must emit a per-model timing block so users
    can answer "why didn't parallel help?" without re-instrumenting."""

    def runner(cmd, env, capture_output, text, timeout, check, input=None):
        return _Completed(returncode=0, stdout="")

    run_claude_models_parallel(
        models=["model-x", "model-y"],
        prompt="hi",
        base_url="http://x",
        api_key="k",
        runner=runner,
    )

    captured = capsys.readouterr()
    assert "[parallel] per-model wall time:" in captured.err
    assert "model-x" in captured.err
    assert "model-y" in captured.err
    assert "speedup=" in captured.err
    assert "slowest=" in captured.err


def test_run_claude_models_parallel_breakdown_marks_cli_errors(capsys):
    """When a model raises ClaudeCLIError, the breakdown should still
    show its row tagged as `cli-error` rather than crashing or omitting it."""

    def runner(cmd, env, capture_output, text, timeout, check, input=None):
        idx = cmd.index("--model")
        if cmd[idx + 1] == "boom":
            raise FileNotFoundError(2, "no such file", "claude")
        return _Completed(returncode=0, stdout="")

    run_claude_models_parallel(
        models=["ok-model", "boom"],
        prompt="hi",
        base_url="http://x",
        api_key="k",
        runner=runner,
    )

    captured = capsys.readouterr()
    assert "ok-model" in captured.err
    assert "boom" in captured.err
    assert "cli-error" in captured.err


def test_run_claude_models_parallel_forwards_extra_args_and_env():
    """Shared kwargs must reach every per-model invocation unchanged."""
    captured_envs: List[dict] = []
    captured_cmds: List[List[str]] = []

    def runner(cmd, env, capture_output, text, timeout, check, input=None):
        captured_envs.append(env)
        captured_cmds.append(cmd)
        return _Completed(returncode=0, stdout="")

    run_claude_models_parallel(
        models=["a", "b"],
        prompt="hi",
        base_url="http://x",
        api_key="k",
        extra_env={"MAX_THINKING_TOKENS": "4096"},
        extra_args=["--allowed-tools", "Bash"],
        runner=runner,
    )

    assert all(env["MAX_THINKING_TOKENS"] == "4096" for env in captured_envs)
    for cmd in captured_cmds:
        assert "--allowed-tools" in cmd
        assert "Bash" in cmd


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
