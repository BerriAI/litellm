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

The probes ride the shared transport: each takes an injected `ProxyClient`
and issues its request through the shared `count_tokens` / `messages`
methods, so they reuse the split control/data-plane routing, timeout,
and typed `Result` handling the rest of `tests/e2e/` uses. This is a
"did the request shape survive the proxy's provider-specific
transformations" test, not a load test, and a real endpoint regression
typically surfaces in well under a second of wall time (400 / 500 from
the upstream, or LiteLLM 500 on a transformation bug).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

from e2e_http import (
    NetworkError,
    RateLimitedError,
    Result,
    Success,
    UnauthorizedError,
    UnknownApiError,
    ValidationError,
)
from models import (
    AnthropicAssistantTurn,
    AnthropicCustomTool,
    AnthropicMessage,
    AnthropicMessagesBody,
    AnthropicMessagesResponse,
    AnthropicTool,
    AnthropicToolResultBlock,
    AnthropicToolResultTurn,
    AnthropicToolSearchTool,
    ChatMessage,
    CountTokensBody,
    CountTokensResponse,
    JsonSchemaProperty,
    ToolInputSchema,
)

from claude_code.rate_limiter import (
    RateLimiter,
    get_default_limiter,
    infer_provider,
)

if TYPE_CHECKING:
    from proxy_client import ProxyClient


# The tool_search discovery tool plus one trivial user tool, matching the
# `tools` array real Claude Code emits when its MCP-tool-search beta is active.
# The discovery tool's `_20251119`-suffixed type is what LiteLLM keys its
# per-provider beta-header translation on; the user tool is included so the wire
# shape mirrors what Claude Code sends rather than a semantically empty request.
_TOOL_SEARCH_TOOLS: tuple[AnthropicTool, ...] = (
    AnthropicToolSearchTool(
        type="tool_search_tool_regex_20251119",
        name="tool_search_tool_regex",
    ),
    AnthropicCustomTool(
        name="add_numbers",
        description="Add two integers",
        input_schema=ToolInputSchema(
            properties={
                "a": JsonSchemaProperty(type="integer"),
                "b": JsonSchemaProperty(type="integer"),
            },
            required=["a", "b"],
        ),
    ),
)

_TOOL_SEARCH_PROMPT = (
    "If you have a tool to discover other tools, use it to "
    "find one. Otherwise reply with the word 'done'."
)


def _acquire(model: str, rate_limiter: RateLimiter | None) -> None:
    """Take one token from the cross-process per-provider limiter so probe
    traffic counts against the same aggregate budget as the CLI rows. Without
    this, an HTTP-probe row would fire unthrottled requests in parallel with
    throttled CLI rows and silently violate the limiter's aggregate-rate
    guarantee. `rate_limiter` is an injection seam for unit tests; production
    callers leave it unset to use the process-wide default."""
    limiter = rate_limiter if rate_limiter is not None else get_default_limiter()
    limiter.acquire(infer_provider(model))


def probe_count_tokens(
    *,
    client: ProxyClient,
    api_key: str,
    model: str,
    message: str = "hello world",
    rate_limiter: RateLimiter | None = None,
) -> Result[CountTokensResponse]:
    """POST to `/v1/messages/count_tokens` for `model` and return the typed result.

    The Anthropic / LiteLLM `count_tokens` endpoint accepts a request body whose
    shape mirrors `/v1/messages` (model + messages) and returns
    `{"input_tokens": N}` for a successful response. Anything else -- non-200
    status, non-JSON body, missing/non-int `input_tokens` -- is a regression the
    cell flips red on (see `assert_count_tokens_shape`).
    """
    _acquire(model, rate_limiter)
    return client.count_tokens(
        api_key,
        CountTokensBody(model=model, messages=[ChatMessage(role="user", content=message)]),
    )


def probe_tool_search(
    *,
    client: ProxyClient,
    api_key: str,
    model: str,
    max_tokens: int = 64,
    rate_limiter: RateLimiter | None = None,
) -> Result[AnthropicMessagesResponse]:
    """POST to `/v1/messages` with a `tool_search_tool_regex_20251119` tool
    definition and return the typed result.

    LiteLLM's `is_tool_search_used` helper keys on the `_20251119`-suffixed type
    string to decide whether to attach the provider-specific tool-search beta
    header (`advanced-tool-use-2025-11-20` for Anthropic/Azure,
    `tool-search-tool-2025-10-19` for Vertex/Bedrock). A proxy regression in that
    translation surfaces here as a 400 from the upstream complaining about the
    tool type or beta header.

    The prompt deliberately does not force a tool call -- the goal is to verify
    the *request* round-trips without 400 and produces some response, not to test
    whether the model decided to invoke tool_search. That kind of behavior test
    would couple this row to Claude Code's model behavior heuristics, which change
    weekly.
    """
    _acquire(model, rate_limiter)
    return client.messages(
        api_key,
        AnthropicMessagesBody(
            model=model,
            max_tokens=max_tokens,
            messages=[ChatMessage(role="user", content=_TOOL_SEARCH_PROMPT)],
            tools=list(_TOOL_SEARCH_TOOLS),
        ),
    )


_TOOL_SEARCH_FOLLOW_UP = "Thanks. Now reply with the word 'done'."
_TOOL_RESULT_STUB = "3"
# A `server_tool_use` block and the `tool_search_tool_result` answering it are
# one indivisible pair: replaying the request without its result is malformed
# Anthropic and 400s on any provider. 64 output tokens is not enough room for
# both, so the turn we replay is generated with a budget that fits the whole
# discovery round trip.
_REPLAY_SOURCE_MAX_TOKENS = 1024
_REPLAYED_SERVER_BLOCKS = frozenset({"server_tool_use", "tool_search_tool_result"})


@dataclass(frozen=True, slots=True)
class ToolSearchReplay:
    """Both turns of the multi-turn probe plus the history the second turn
    carried, so a failing cell can report which turn broke and what was on the
    wire when it did."""

    first_turn: Result[AnthropicMessagesResponse]
    history: tuple[AnthropicMessage, ...]
    second_turn: Result[AnthropicMessagesResponse] | None


def _replayed_server_block_types(history: tuple[AnthropicMessage, ...]) -> frozenset[str]:
    return frozenset(
        block.type
        for turn in history
        if isinstance(turn, AnthropicAssistantTurn)
        for block in turn.content
        if block.type in _REPLAYED_SERVER_BLOCKS
    )


def _replay_history(answer: AnthropicMessagesResponse) -> tuple[AnthropicMessage, ...]:
    """Turn a real first-turn answer into a well-formed two-turn history.

    Every client-side `tool_use` the model emitted gets a `tool_result` keyed on
    the id the model actually returned; a turn with none gets a plain follow-up
    instead. An unanswered `tool_use`, or a `tool_result` pointing at an invented
    id, is malformed Anthropic and 400s on any provider, which would make this
    probe measure our own request rather than the provider's handling of the
    replayed server-tool blocks."""
    blocks = tuple(answer.content or ())
    pending = tuple(block.id for block in blocks if block.type == "tool_use" and block.id is not None)
    reply: AnthropicMessage = (
        AnthropicToolResultTurn(
            content=[
                AnthropicToolResultBlock(tool_use_id=tool_use_id, content=_TOOL_RESULT_STUB)
                for tool_use_id in pending
            ]
        )
        if pending
        else ChatMessage(role="user", content=_TOOL_SEARCH_FOLLOW_UP)
    )
    return (
        ChatMessage(role="user", content=_TOOL_SEARCH_PROMPT),
        AnthropicAssistantTurn(content=list(blocks)),
        reply,
    )


def probe_tool_search_multiturn(
    *,
    client: ProxyClient,
    api_key: str,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> ToolSearchReplay:
    """Run `probe_tool_search`, then send the real assistant turn back as
    history with the same tools still declared.

    The first turn only proves the proxy attaches the tool-search beta header on
    the way out. Nothing proves the provider accepts the `server_tool_use` and
    `tool_search_tool_result` blocks it produced when they come back in
    `messages`, which is every turn of a real Claude Code session after the
    first."""
    first_turn = probe_tool_search(
        client=client,
        api_key=api_key,
        model=model,
        max_tokens=_REPLAY_SOURCE_MAX_TOKENS,
        rate_limiter=rate_limiter,
    )
    if not isinstance(first_turn, Success):
        return ToolSearchReplay(first_turn=first_turn, history=(), second_turn=None)

    history = _replay_history(first_turn.data)
    _acquire(model, rate_limiter)
    return ToolSearchReplay(
        first_turn=first_turn,
        history=history,
        second_turn=client.messages(
            api_key,
            AnthropicMessagesBody(
                model=model,
                max_tokens=64,
                messages=list(history),
                tools=list(_TOOL_SEARCH_TOOLS),
            ),
        ),
    )


def _failure_diagnostic[R: BaseModel](result: Result[R], route: str) -> str:
    """Map a non-success `Result` to a one-line diagnostic. The `status 429`
    wording is load-bearing: the compat conftest classifies a rate-limited cell
    by matching the failure text against `RATE_LIMIT_SHAPED_RE`, so the literal
    `429` must survive into the reported error."""
    match result:
        case Success():
            return ""
        case UnauthorizedError():
            return "status 401 (unauthorized)"
        case RateLimitedError(body=body):
            return f"status 429: {body[:400]}"
        case UnknownApiError(status_code=status_code, body=body):
            return f"status {status_code}: {body[:400]}"
        case ValidationError(message=message):
            return f"unexpected {route} response body: {message}"
        case NetworkError(message=message):
            return f"transport error: {message}"
        case _:
            return f"unexpected result: {result!r}"


def assert_tool_search_shape(result: Result[AnthropicMessagesResponse]) -> str | None:
    """Return None on success, else describe the first violation.

    Acceptance criteria:

      1. The call succeeded (HTTP 200, no 400 from the upstream rejecting the
         tool_search tool type or a missing beta header, no 429/401/transport
         error).
      2. The body has either `content` (Anthropic-shape passthrough) or `choices`
         (LiteLLM normalized OpenAI-shape, used by Bedrock Converse). Either is
         acceptable -- the matrix cares that the proxy *accepts and forwards*
         tool_search, not that the model actually chose to invoke it.
    """
    match result:
        case Success(data=data):
            if data.content is None and data.choices is None:
                keys = sorted(data.model_dump(exclude_none=True).keys())
                return f"response has neither `content` nor `choices`: keys={keys}"
            return None
        case _:
            return _failure_diagnostic(result, "/v1/messages")


def assert_tool_search_replay_shape(replay: ToolSearchReplay) -> str | None:
    """Return None on success, else describe the first violation.

    Acceptance criteria:

      1. The first turn succeeded, on the same terms as `assert_tool_search_shape`.
      2. That turn produced a complete `server_tool_use` / `tool_search_tool_result`
         pair to replay. Without both the second turn carries either an ordinary
         text history or a half-finished tool call, and the cell would report on
         our own request rather than on the provider's handling of server-tool
         blocks in history.
      3. The provider accepted the history containing those blocks.
    """
    first_error = assert_tool_search_shape(replay.first_turn)
    if first_error is not None:
        return f"first turn: {first_error}"

    replayed = _replayed_server_block_types(replay.history)
    missing = _REPLAYED_SERVER_BLOCKS - replayed
    if missing:
        return (
            f"first turn returned no {' or '.join(sorted(missing))} block to replay, so the history "
            "proves nothing about server-tool handling; a turn truncated at max_tokens looks like this"
        )

    if replay.second_turn is None:
        return "second turn was never sent"

    second_error = assert_tool_search_shape(replay.second_turn)
    if second_error is not None:
        return f"history replaying {sorted(replayed)} rejected: {second_error}"
    return None


def assert_count_tokens_shape(result: Result[CountTokensResponse]) -> str | None:
    """Return None on success, or an error string describing the first violation.

    Acceptance criteria are intentionally minimal:

      1. The call succeeded (HTTP 200, valid JSON parsing into `input_tokens`).
      2. `input_tokens` is a positive int.

    Anything beyond that (cache token fields, server metadata) is optional and
    varies by provider/transport; asserting on extras would create a brittle test
    that flips red on neutral protocol drift.
    """
    match result:
        case Success(data=data):
            if data.input_tokens <= 0:
                return f"input_tokens must be positive; got {data.input_tokens}"
            return None
        case _:
            return _failure_diagnostic(result, "/v1/messages/count_tokens")
