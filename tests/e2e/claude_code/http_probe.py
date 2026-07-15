"""Direct HTTP probe helpers for the Claude Code compatibility matrix.

Most matrix cells drive the `claude` CLI in headless mode and observe
the stream-json wire (see `cli_driver.py`). A handful of features the
proxy must support don't have any CLI surface area -- `count_tokens` is
the canonical example: Claude Code calls it internally for budget
display, but the result never appears in stream-json events, so a CLI
test cannot observe whether the endpoint round-tripped correctly
through the proxy for any given provider.

This module is the second test pattern the matrix supports: a plain
HTTP POST against a LiteLLM proxy endpoint, parsed and shape-checked
in the test, with the same `compat_result` recording convention as the
CLI-driven cells. The goal is to keep this pattern *narrow* -- if a
feature can be tested via the CLI, it should be, because the CLI path
is closer to what real Claude Code users hit. HTTP probes are only for
features the CLI can't reach.

The probe deliberately uses a short timeout (30s) and small payloads:
this is a "did the request shape survive the proxy's
provider-specific transformations" test, not a load test, and a real
endpoint regression typically surfaces in well under a second of wall
time (400 / 500 from the upstream, or LiteLLM 500 on a transformation
bug).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import httpx
from pydantic import ValidationError

from claude_code.json_types import JSON_VALUE_ADAPTER, JSONValue
from claude_code.rate_limiter import (
    RateLimiter,
    get_default_limiter,
    infer_provider,
)


DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass
class ProbeResult:
    """Structured outcome of a single HTTP probe.

    `status_code` and `body` are the wire response; `payload` is the
    parsed JSON body if the response was JSON, else None. Tests assert
    on `status_code` + `payload` shape; `body` is preserved so failure
    diagnostics can echo the raw error string (which is the only thing
    a maintainer needs to triage a red cell).
    """

    status_code: int
    body: str
    payload: Optional[JSONValue] = None
    error: Optional[str] = None


def probe_count_tokens(
    *,
    base_url: str,
    api_key: str,
    model: str,
    message: str = "hello world",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    rate_limiter: Optional[RateLimiter] = None,
) -> ProbeResult:
    """POST to `{base_url}/v1/messages/count_tokens` for `model` and return the parsed result.

    The Anthropic / LiteLLM `count_tokens` endpoint accepts a request
    body whose shape mirrors `/v1/messages` (model + messages), and
    returns `{"input_tokens": N}` for a successful response. Anything
    else -- non-200 status, non-JSON body, missing/non-int
    `input_tokens` -- is a regression we want the cell to flip red on.

    The same cross-process token-bucket limiter `cli_driver.run_claude`
    uses is acquired here too, so probe rows count against the
    aggregate per-provider budget. Without this, an HTTP-probe row
    would fire unthrottled requests in parallel with throttled CLI
    rows and silently violate the limiter's aggregate-rate guarantee.
    `rate_limiter` is an injection seam for unit tests; production
    callers should leave it unset to use the process-wide default.
    """
    limiter = rate_limiter if rate_limiter is not None else get_default_limiter()
    limiter.acquire(infer_provider(model))

    url = base_url.rstrip("/") + "/v1/messages/count_tokens"
    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # `anthropic-version` is required by Anthropic's native
                # API and harmless on every other provider the proxy
                # routes to. Matches what the Claude Code CLI sends
                # for its own internal `count_tokens` calls.
                "anthropic-version": "2023-06-01",
            },
            json={"model": model, "messages": [{"role": "user", "content": message}]},
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return ProbeResult(status_code=0, body="", error=f"transport: {exc}")

    body = response.text or ""
    try:
        payload = JSON_VALUE_ADAPTER.validate_json(body) if body else None
    except ValidationError:
        payload = None

    return ProbeResult(
        status_code=response.status_code,
        body=body,
        payload=payload,
    )


def probe_tool_search(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    rate_limiter: Optional[RateLimiter] = None,
) -> ProbeResult:
    """POST to `{base_url}/v1/messages` with a `tool_search_tool_regex_20251119`
    tool definition and return the result.

    The shape of the tools array is the one Claude Code emits when its
    MCP-tool-search beta is active: a `tool_search_tool_regex_20251119`
    discovery tool (name `tool_search_tool_regex`) plus at least one
    regular user tool to be searched. LiteLLM's
    `is_tool_search_used` helper keys on the `_20251119`-suffixed type
    string to decide whether to attach the provider-specific tool-search
    beta header (`advanced-tool-use-2025-11-20` for Anthropic/Azure,
    `tool-search-tool-2025-10-19` for Vertex/Bedrock). A proxy
    regression in that translation will surface here as a 400 from
    the upstream complaining about the tool type or beta header.

    The prompt deliberately does not force a tool call -- the goal is
    to verify the *request* round-trips without 400 and produces some
    response, not to test whether the model decided to invoke
    tool_search. That kind of behavior test would couple this row to
    Claude Code's model behavior heuristics, which change weekly.

    Like `probe_count_tokens`, this acquires one token from the
    process-wide rate limiter so probe traffic counts against the
    same aggregate per-provider budget as the CLI rows. `rate_limiter`
    is a test seam; production callers should leave it unset.
    """
    limiter = rate_limiter if rate_limiter is not None else get_default_limiter()
    limiter.acquire(infer_provider(model))

    url = base_url.rstrip("/") + "/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 64,
        "messages": [
            {
                "role": "user",
                "content": (
                    "If you have a tool to discover other tools, use it to "
                    "find one. Otherwise reply with the word 'done'."
                ),
            }
        ],
        "tools": [
            # The tool_search discovery tool itself. Type is the SDK-
            # version-pinned `_20251119` suffix; name is the canonical
            # `tool_search_tool_regex` (no suffix) Anthropic accepts.
            # LiteLLM keys its beta-header translation on the type.
            {
                "type": "tool_search_tool_regex_20251119",
                "name": "tool_search_tool_regex",
            },
            # A trivial user tool for the discovery tool to potentially
            # surface. Without at least one non-search tool the request
            # is shape-valid but semantically empty; we include one so
            # the wire shape mirrors what real Claude Code sends.
            {
                "name": "add_numbers",
                "description": "Add two integers",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        ],
    }
    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        return ProbeResult(status_code=0, body="", error=f"transport: {exc}")

    body = response.text or ""
    try:
        payload_out = JSON_VALUE_ADAPTER.validate_json(body) if body else None
    except ValidationError:
        payload_out = None

    return ProbeResult(
        status_code=response.status_code,
        body=body,
        payload=payload_out,
    )


def assert_tool_search_shape(result: ProbeResult) -> Optional[str]:
    """Return None on success, else describe the first violation.

    Acceptance criteria:

      1. HTTP status is 200 (no 400 from the upstream rejecting the
         tool_search tool type or a missing beta header).
      2. Body is valid JSON.
      3. Body has either `content` (Anthropic-shape passthrough) or
         `choices` (LiteLLM normalized openai-shape, used by Bedrock
         Converse). Either is acceptable -- the matrix cares that the
         proxy *accepts and forwards* tool_search, not that the model
         actually chose to invoke it. Tool-invocation behavior is a
         model decision the matrix has no business asserting on.

    The cell goes red when the upstream rejects the tool type, the
    proxy drops the beta header, or the response shape is unusable.
    Anything else (model decided to call or not call tool_search) is
    irrelevant for this row.
    """
    if result.error is not None:
        return f"transport error: {result.error}"
    if result.status_code != 200:
        return f"status {result.status_code}: {result.body[:400]}"
    if result.payload is None:
        return f"non-JSON body: {result.body[:400]}"
    if not isinstance(result.payload, Mapping):
        return f"body is not a JSON object: {type(result.payload).__name__}"
    # LiteLLM normalizes some provider responses to OpenAI shape
    # (`choices`) and passes others through Anthropic-shape (`content`).
    # Accept either; both prove the proxy round-tripped the request.
    if "content" not in result.payload and "choices" not in result.payload:
        return (
            f"response has neither `content` nor `choices`: "
            f"keys={sorted(result.payload.keys())}"
        )
    return None


def assert_count_tokens_shape(result: ProbeResult) -> Optional[str]:
    """Return None on success, or an error string describing the first violation.

    Acceptance criteria are intentionally minimal:

      1. HTTP status is 200.
      2. Body is valid JSON.
      3. Body has an `input_tokens` key whose value is a positive int.

    Anything beyond that (cache token fields, server metadata) is
    optional and varies by provider/transport. Asserting on extras
    would create a brittle test that flips red on neutral protocol
    drift; matrix cells should only go red on functional regressions
    a Claude Code user would feel.
    """
    if result.error is not None:
        return f"transport error: {result.error}"
    if result.status_code != 200:
        return f"status {result.status_code}: {result.body[:400]}"
    if result.payload is None:
        return f"non-JSON body: {result.body[:400]}"
    if not isinstance(result.payload, Mapping):
        return f"body is not a JSON object: {type(result.payload).__name__}"
    tokens = result.payload.get("input_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool):
        return f"input_tokens missing or not an int: got {tokens!r}"
    if tokens <= 0:
        return f"input_tokens must be positive; got {tokens}"
    return None
