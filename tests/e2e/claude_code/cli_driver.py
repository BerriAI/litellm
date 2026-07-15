"""Claude Code CLI Driver.

A thin wrapper around the `claude` CLI in headless mode. Every compatibility
test consumes only this module — tests must never shell out directly. This
keeps the subprocess assembly, stream-JSON parsing, and result shape in a
single place that can be unit-tested with a mocked subprocess.

The driver is deliberately small: it knows how to invoke the CLI, drain its
stream-JSON output, and return a structured `DriverResult`. Higher-level
matrix concerns (status aggregation, manifest lookup, JSON serialization)
live in `matrix_builder.py`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from claude_code.rate_limiter import (
    RateLimiter,
    get_default_limiter,
    infer_provider,
)

CLAUDE_CLI_DEFAULT = "claude"
# 120s is fine for a single isolated CLI call against an unloaded
# upstream, but the matrix run launches up to 75 concurrent calls and
# upstreams can take several minutes to respond under that contention.
# We expose the timeout as an env var so binary-search runs can grow
# it without touching the test code.
DEFAULT_TIMEOUT_SECONDS = float(
    os.environ.get("LITELLM_COMPAT_CLI_TIMEOUT_SECONDS") or 120
)

# Env vars the `claude` Node CLI legitimately needs to function:
# locating its own binary + node, basic locale/terminal plumbing.
# Deliberately excludes every credential-bearing var that the
# surrounding CI job sets for the proxy (ANTHROPIC_API_KEY, AWS_*,
# AZURE_*, VERTEXAI_CREDENTIALS, GITHUB_TOKEN, OPENAI_API_KEY,
# DATABASE_URL, ...). The CLI talks to the proxy via the explicit
# ANTHROPIC_BASE_URL/ANTHROPIC_AUTH_TOKEN we set below — it has no
# business reading the proxy's upstream credentials, and a
# compromised CLI release shouldn't be able to exfiltrate them out
# of the CI environment.
#
# `HOME` is intentionally NOT in this list: see `_make_isolated_home`
# below. We give the CLI a fresh empty per-invocation HOME so a
# compromised claude package or a model-directed `Read` tool call
# can't reach files like `~/.config/gh/hosts.yml`, `~/.ssh/id_rsa`,
# or `~/.bash_history` on the cron VM (and on the CircleCI executor
# the same isolation prevents accidentally exposing checkout-adjacent
# files even though the runner home is ephemeral there).
_CLI_ENV_ALLOWLIST: tuple = (
    "PATH",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "NODE_PATH",
    "NVM_DIR",
    "NVM_BIN",
)


def _make_isolated_home() -> str:
    """Create a fresh empty HOME directory for a single `claude` subprocess.

    The CLI needs *a* writable HOME (it caches per-session state under
    `$HOME/.claude/projects/<sha>/`), but it has no legitimate need
    for the *user's* HOME. Handing it the real one means a compromised
    `@anthropic-ai/claude-code` release, or a model-directed `Read`
    tool call during a PDF/vision cell, can read host files like
    `~/.config/gh/hosts.yml` (GitHub CLI host token), `~/.ssh/`,
    `~/.bash_history`, or any other dotfile under the runtime user's
    home. On the cron VM the runtime user is a real interactive
    account (`mateo`) with a populated home directory, so this is a
    real exfiltration surface.

    The directory is created under `tempfile.gettempdir()` (which is
    `/tmp` on Linux; under systemd's `PrivateTmp=true` that's a
    per-service tmpfs that the service user can't otherwise reach).
    Caller is responsible for `shutil.rmtree`-ing it after the
    subprocess exits.
    """
    return tempfile.mkdtemp(prefix="claude-cli-home-")


class ClaudeCLIError(RuntimeError):
    """Raised when the `claude` CLI cannot be invoked or returns a fatal error."""


@dataclass
class DriverResult:
    """Structured outcome of a single `claude` CLI invocation.

    `text` is the assistant's final user-visible reply (joined across any
    intermediate `assistant` events for non-streaming runs). `events` is the
    raw list of stream-JSON objects emitted by the CLI, preserved so test
    authors can write feature-specific assertions (tool calls, cache hits,
    usage) without re-parsing stdout.
    """

    text: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    exit_code: int = 0
    stderr: str = ""
    usage: Optional[Dict[str, Any]] = None
    duration_ms: Optional[int] = None


def run_claude(
    *,
    prompt: Optional[str],
    model: str,
    base_url: str,
    api_key: str,
    extra_env: Optional[Mapping[str, str]] = None,
    extra_args: Optional[Sequence[str]] = None,
    stdin_input: Optional[str] = None,
    cli_path: str = CLAUDE_CLI_DEFAULT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Optional[Any] = None,
    rate_limiter: Optional[RateLimiter] = None,
) -> DriverResult:
    """Invoke `claude` once in headless stream-JSON mode and return the result.

    The CLI is pointed at a LiteLLM proxy via `ANTHROPIC_BASE_URL` /
    `ANTHROPIC_AUTH_TOKEN`, so the same code path exercises every provider
    column — only the model id and the proxy's routing differ between
    invocations.

    `runner` is an injection seam used by the unit tests: by default we call
    `subprocess.run`, but the test suite swaps in a fake that yields canned
    stream-JSON. Production callers should never set it.

    `rate_limiter` is the second injection seam: the cross-process
    token-bucket limiter throttles outbound calls per provider so a
    fully-parallel matrix run doesn't trip 429s. Defaults to the
    process-wide singleton; unit tests pass a no-op limiter or one
    backed by a tmp dir to keep tests hermetic.
    """
    if prompt is None and stdin_input is None:
        raise ValueError("must supply either `prompt` or `stdin_input`")
    if prompt is not None and stdin_input is not None:
        raise ValueError("must supply only one of `prompt` or `stdin_input`, not both")
    if prompt is not None and not prompt:
        raise ValueError("prompt must be a non-empty string when provided")
    if stdin_input is not None and not stdin_input:
        raise ValueError("stdin_input must be a non-empty string when provided")
    if not model:
        raise ValueError("model must be a non-empty string")
    if not base_url:
        raise ValueError("base_url must be a non-empty string")
    if not api_key:
        raise ValueError("api_key must be a non-empty string")

    # `claude --print` takes the prompt as the **last positional argument**.
    # Flags must come before it, otherwise they're parsed as part of the
    # prompt (or silently dropped, depending on the CLI version) and the
    # tool_use / vision cells fail with confusing "no tool_use observed"
    # errors. Build the flag list first, then append the prompt last.
    #
    # When `extra_args` contains a *variadic* flag like `--allowed-tools
    # WebSearch` (commander.js's `<tools...>`), the parser greedily
    # consumes every subsequent token as part of the variadic list — so
    # the prompt would be eaten as a tool name. Inserting `--` before
    # the prompt terminates option parsing and leaves the prompt as a
    # plain positional, which works for variadic and non-variadic flags
    # alike.
    cmd: List[str] = [
        cli_path,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
    ]
    if extra_args:
        cmd.extend(extra_args)
    if prompt is not None:
        cmd.append("--")
        cmd.append(prompt)

    # Build a minimal env for the CLI subprocess: only the allowlisted
    # process-runtime vars from os.environ, plus the explicit proxy
    # creds, plus any caller-supplied overrides. See _CLI_ENV_ALLOWLIST
    # above for the security rationale.
    env: Dict[str, str] = {
        key: os.environ[key] for key in _CLI_ENV_ALLOWLIST if key in os.environ
    }
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_AUTH_TOKEN"] = api_key
    # Hand the CLI a fresh empty HOME so a compromised claude package
    # or a model-directed Read tool call can't see the runtime user's
    # real dotfiles. Created here, removed in the `finally` below
    # regardless of how the subprocess exits.
    isolated_home = _make_isolated_home()
    env["HOME"] = isolated_home
    if extra_env:
        env.update(extra_env)

    # Throttle by provider *before* launching the CLI. Doing this here
    # (rather than per-test) means every code path that lands on
    # `run_claude` is rate-limited automatically — including
    # `run_claude_models_parallel`, which is the hot path during the
    # full matrix run.
    limiter = rate_limiter if rate_limiter is not None else get_default_limiter()
    provider = infer_provider(model)
    limiter.acquire(provider)

    run_fn = runner or subprocess.run
    try:
        try:
            completed = run_fn(
                cmd,
                env=env,
                input=stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ClaudeCLIError(
                f"claude CLI not found at {cli_path!r}; install with `npm i -g @anthropic-ai/claude-code`"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ClaudeCLIError(f"claude CLI timed out after {timeout}s") from exc
    finally:
        # Best-effort cleanup. If the subprocess wrote a `.claude/`
        # session dir under the isolated HOME, we remove it here so
        # parallel matrix runs don't accumulate per-call tmpdirs.
        # `ignore_errors=True` because rmtree races with any
        # not-yet-reaped child (SIGTERM'd `claude` on a host-side
        # timeout) are benign — the next matrix run starts from a
        # fresh tmpdir anyway.
        shutil.rmtree(isolated_home, ignore_errors=True)

    events = _parse_stream_json(completed.stdout or "")
    text = _extract_assistant_text(events)
    usage = _extract_usage(events)

    return DriverResult(
        text=text,
        events=events,
        exit_code=completed.returncode,
        stderr=completed.stderr or "",
        usage=usage,
    )


ModelResult = Union[DriverResult, ClaudeCLIError]


def run_claude_models_parallel(
    *,
    models: Sequence[str],
    prompt: Optional[str],
    base_url: str,
    api_key: str,
    extra_env: Optional[Mapping[str, str]] = None,
    extra_args: Optional[Sequence[str]] = None,
    stdin_input: Optional[str] = None,
    cli_path: str = CLAUDE_CLI_DEFAULT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, ModelResult]:
    """Invoke `run_claude` for every `models[i]` concurrently and collect outcomes.

    Each `claude` CLI invocation is a long-lived subprocess that spends
    almost all of its time waiting on the upstream API; running the
    three Claude tiers in parallel cuts the per-cell wall time roughly
    threefold without changing what each invocation does.

    Threads (rather than asyncio) are the right primitive here because
    `subprocess.run` releases the GIL while it waits, and we want to
    keep the synchronous CLI driver unchanged so unit tests can keep
    injecting a fake `runner`.

    Returns a dict keyed by model id. Each value is either the
    `DriverResult` produced by `run_claude` or the `ClaudeCLIError`
    that aborted that model's run — callers decide how to map either
    into a `compat_result` entry. The shared kwargs (prompt, env, args,
    timeout, runner) are forwarded verbatim so the per-model wire is
    identical to what the sequential path produces.
    """
    if not models:
        raise ValueError("models must be a non-empty sequence")

    def _one(model: str) -> Tuple[str, ModelResult, float]:
        # Per-model wall clock: this is what the matrix run actually pays for.
        # We record it whether the run succeeded or raised so the breakdown
        # log below covers both code paths and surfaces "which model is the
        # long pole?" without requiring per-test instrumentation.
        started = time.monotonic()
        try:
            result = run_claude(
                prompt=prompt,
                model=model,
                base_url=base_url,
                api_key=api_key,
                extra_env=extra_env,
                extra_args=extra_args,
                stdin_input=stdin_input,
                cli_path=cli_path,
                timeout=timeout,
                runner=runner,
            )
            elapsed = time.monotonic() - started
            # Stamp the duration onto the DriverResult so callers (tests,
            # diagnostics) can attribute slow cells without re-timing.
            result.duration_ms = int(elapsed * 1000)
            return model, result, elapsed
        except ClaudeCLIError as exc:
            elapsed = time.monotonic() - started
            return model, exc, elapsed
        except Exception as exc:
            # Honor the documented "errors as values" contract for any
            # exception type — not just ClaudeCLIError. The rate
            # limiter does file I/O (OSError), `infer_provider` can
            # raise ValueError on edge-case model strings, and a future
            # bug elsewhere in the call stack must not abort the entire
            # parallel batch and lose the other models' outcomes.
            elapsed = time.monotonic() - started
            wrapped = ClaudeCLIError(
                f"unexpected error running model {model!r}: "
                f"{type(exc).__name__}: {exc}"
            )
            wrapped.__cause__ = exc
            return model, wrapped, elapsed

    outcomes: Dict[str, ModelResult] = {}
    durations: Dict[str, float] = {}
    overall_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = [pool.submit(_one, model) for model in models]
        for future in as_completed(futures):
            model, outcome, elapsed = future.result()
            outcomes[model] = outcome
            durations[model] = elapsed
    overall_elapsed = time.monotonic() - overall_started

    _log_parallel_breakdown(models, durations, outcomes, overall_elapsed)
    return outcomes


def _log_parallel_breakdown(
    models: Sequence[str],
    durations: Mapping[str, float],
    outcomes: Mapping[str, ModelResult],
    overall_elapsed: float,
) -> None:
    """Emit a one-block timing breakdown to stderr.

    Pytest only shows captured output for failing tests by default, but
    `-s` surfaces it for passing tests too — which is exactly when you
    care about "did parallelization actually help?". The block reports:

      - per-model wall time and outcome (ok / cli-error / non-zero exit)
      - the slowest model (the parallel run's wall-time floor)
      - the sum of sequential model times (what the old serial path
        would have paid)
      - the overall parallel wall time and the speedup ratio

    If one model dominates, `slowest ≈ overall ≈ sequential / 1`, and
    the speedup will be near 1× — exactly the diagnostic that explains
    "why didn't this get faster?".
    """
    sequential_total = sum(durations.values())
    slowest_model = max(durations, key=durations.get) if durations else None
    slowest = durations[slowest_model] if slowest_model else 0.0
    speedup = sequential_total / overall_elapsed if overall_elapsed > 0 else 0.0

    lines: List[str] = []
    lines.append("[parallel] per-model wall time:")
    for model in models:
        elapsed = durations.get(model, 0.0)
        outcome = outcomes.get(model)
        if isinstance(outcome, ClaudeCLIError):
            status = "cli-error"
        elif isinstance(outcome, DriverResult):
            status = f"exit={outcome.exit_code}"
        else:
            status = "missing"
        lines.append(f"  {model:<40s} {elapsed:6.2f}s  ({status})")
    if slowest_model is not None:
        lines.append(
            f"[parallel] slowest={slowest_model} ({slowest:.2f}s); "
            f"sequential_sum={sequential_total:.2f}s; "
            f"parallel_wall={overall_elapsed:.2f}s; "
            f"speedup={speedup:.2f}x"
        )
    print("\n".join(lines), file=sys.stderr, flush=True)


def _parse_stream_json(stdout: str) -> List[Dict[str, Any]]:
    """Parse newline-delimited JSON emitted by `claude --output-format stream-json`.

    Lines that don't parse as JSON are silently skipped — the CLI occasionally
    emits debug output we don't care about, and a single malformed line should
    not abort the whole run. Real failure modes surface via exit code.
    """
    events: List[Dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _extract_assistant_text(events: Sequence[Mapping[str, Any]]) -> str:
    """Concatenate the text content of every `assistant` event in order.

    The non-streaming `--print` path emits a single `assistant` event whose
    `message.content` is a list of content blocks. We walk the blocks and
    join every `text` block — the CLI prints other block types (e.g.
    `tool_use`) which we ignore for the basic-messaging case.
    """
    chunks: List[str] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            chunks.append(content)
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                chunks.append(block["text"])
    return "".join(chunks)


def failure_diagnostic(result: "DriverResult", *, max_len: int = 800) -> str:
    """Build a human-readable error string from a non-zero `claude` CLI run.

    The CLI is annoying to debug because the most useful failure signal
    rarely lands on stderr. When the proxy returns an HTTP error, the CLI
    swallows it into an `assistant`/`result` event on **stdout** with
    `is_error: true` and a JSON-shaped `text` block — and exits non-zero.
    Tests that only print `stderr.strip()` see an empty string, which is
    exactly the situation that masked a misconfigured proxy in early
    bring-up of the compat matrix.

    This helper concatenates the most useful diagnostic we can find, in
    priority order:

      1. `result.text` (the assistant's user-visible reply, which is where
         API errors land in stream-json mode), trimmed
      2. `api_error_status` from any `result` event, if present
      3. `result.stderr`, trimmed
      4. `<no diagnostic output>` as a last resort

    The output is truncated to `max_len` characters so a giant HTML 502
    page from a misbehaving load balancer doesn't blow up the matrix
    JSON.
    """
    pieces: List[str] = [f"exit={result.exit_code}"]

    # api_error_status only appears on the final `result` event when the
    # CLI received an HTTP error from the upstream API. Surfacing it
    # explicitly makes "is this a proxy/auth problem or a CLI problem?"
    # answerable without re-reading the events list.
    api_status = _extract_api_error_status(result.events)
    if api_status is not None:
        pieces.append(f"api_status={api_status}")

    text = (result.text or "").strip()
    if text:
        pieces.append(f"text={_truncate(text, max_len)}")

    stderr = (result.stderr or "").strip()
    if stderr:
        pieces.append(f"stderr={_truncate(stderr, max_len)}")

    if len(pieces) == 1:
        pieces.append("(no diagnostic output)")

    return "; ".join(pieces)


def _extract_api_error_status(
    events: Sequence[Mapping[str, Any]],
) -> Optional[int]:
    """Return the `api_error_status` from the last `result` event, if any."""
    for event in reversed(list(events)):
        if event.get("type") != "result":
            continue
        status = event.get("api_error_status")
        if isinstance(status, int):
            return status
    return None


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "...(truncated)"


def _extract_usage(events: Sequence[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the most recent `usage` block seen on any event, if any.

    The CLI surfaces token + cache usage on the final `result` event for
    non-streaming runs, but earlier events also carry partial usage in some
    versions; taking the last non-empty one is the safe default.
    """
    last: Optional[Dict[str, Any]] = None
    for event in events:
        usage = event.get("usage")
        if isinstance(usage, dict) and usage:
            last = usage
            continue
        message = event.get("message")
        if isinstance(message, dict):
            inner = message.get("usage")
            if isinstance(inner, dict) and inner:
                last = inner
    return last
