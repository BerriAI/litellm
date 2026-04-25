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
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

CLAUDE_CLI_DEFAULT = "claude"
DEFAULT_TIMEOUT_SECONDS = 120


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
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    extra_env: Optional[Mapping[str, str]] = None,
    extra_args: Optional[Sequence[str]] = None,
    cli_path: str = CLAUDE_CLI_DEFAULT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    runner: Optional[Any] = None,
) -> DriverResult:
    """Invoke `claude` once in headless stream-JSON mode and return the result.

    The CLI is pointed at a LiteLLM proxy via `ANTHROPIC_BASE_URL` /
    `ANTHROPIC_AUTH_TOKEN`, so the same code path exercises every provider
    column — only the model id and the proxy's routing differ between
    invocations.

    `runner` is an injection seam used by the unit tests: by default we call
    `subprocess.run`, but the test suite swaps in a fake that yields canned
    stream-JSON. Production callers should never set it.
    """
    if not prompt:
        raise ValueError("prompt must be a non-empty string")
    if not model:
        raise ValueError("model must be a non-empty string")
    if not base_url:
        raise ValueError("base_url must be a non-empty string")
    if not api_key:
        raise ValueError("api_key must be a non-empty string")

    cmd: List[str] = [
        cli_path,
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--model",
        model,
        prompt,
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = {**os.environ, **(extra_env or {})}
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_AUTH_TOKEN"] = api_key

    run_fn = runner or subprocess.run
    try:
        completed = run_fn(
            cmd,
            env=env,
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
