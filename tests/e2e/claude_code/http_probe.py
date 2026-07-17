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
in the test. It rides the shared e2e transport rather than a bespoke
HTTP client: each probe takes an injected `Gateway` and issues the
request through `gateway.transport.send`, so the split control/data
plane routing, timeouts, and error normalization are the same ones the
rest of `tests/e2e/` uses. The claude-specific parts that stay are the
request bodies (the `tool_search_tool_regex_20251119` discovery tool
shape) and the per-provider rate-limit accounting the CLI rows share.

The goal is to keep this pattern *narrow* -- if a feature can be tested
via the CLI, it should be, because the CLI path is closer to what real
Claude Code users hit. HTTP probes are only for features the CLI can't
reach.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from e2e_gateway import Gateway
from e2e_http import Headers, StreamingResponse
from models import ChatMessage

from claude_code.rate_limiter import (
    RateLimiter,
    get_default_limiter,
    infer_provider,
)


ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProbeHeaders(Headers):
    """Bearer auth plus the `anthropic-version` header Claude Code sends on its
    own internal `/v1/messages` and `count_tokens` calls. It is required by
    Anthropic's native API and harmless on every other provider the proxy routes
    to, so the probe mirrors it for wire fidelity."""

    authorization: str | None = None
    anthropic_version: str = Field(default=ANTHROPIC_VERSION, alias="anthropic-version")


class CountTokensRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


class ToolSearchDiscoveryTool(BaseModel):
    """The tool_search discovery tool itself. `type` is the SDK-version-pinned
    `_20251119` suffix LiteLLM keys its beta-header translation on; `name` is the
    canonical `tool_search_tool_regex` (no suffix) Anthropic accepts."""

    type: str = "tool_search_tool_regex_20251119"
    name: str = "tool_search_tool_regex"


class ToolInputProperty(BaseModel):
    type: str


class ToolInputSchema(BaseModel):
    type: str = "object"
    properties: dict[str, ToolInputProperty]
    required: list[str]


class UserTool(BaseModel):
    """A trivial user tool for the discovery tool to potentially surface, so the
    wire shape mirrors what real Claude Code sends (>=1 non-search tool)."""

    name: str
    description: str
    input_schema: ToolInputSchema


ProbeTool = ToolSearchDiscoveryTool | UserTool


class ToolSearchRequest(BaseModel):
    model: str
    max_tokens: int
    messages: list[ChatMessage]
    tools: list[ProbeTool]


class CountTokensResult(BaseModel):
    input_tokens: int


class ContentBlock(BaseModel):
    type: str | None = None


class ResponseChoice(BaseModel):
    index: int | None = None


class ToolSearchResponse(BaseModel):
    """LiteLLM normalizes some provider responses to OpenAI shape (`choices`) and
    passes others through Anthropic-shape (`content`). Either proves the proxy
    round-tripped the request; the matrix does not assert the model actually
    invoked tool_search."""

    content: list[ContentBlock] | None = None
    choices: list[ResponseChoice] | None = None


def _probe_headers(gateway: Gateway) -> AnthropicProbeHeaders:
    return AnthropicProbeHeaders(authorization=gateway.transport.master.authorization)


def _acquire(model: str, rate_limiter: RateLimiter | None) -> None:
    limiter = rate_limiter if rate_limiter is not None else get_default_limiter()
    _ = limiter.acquire(infer_provider(model))


def probe_count_tokens(
    *,
    gateway: Gateway,
    model: str,
    message: str = "hello world",
    rate_limiter: RateLimiter | None = None,
) -> StreamingResponse:
    """POST to `/v1/messages/count_tokens` for `model` and return the raw outcome.

    The Anthropic / LiteLLM `count_tokens` endpoint accepts a request body whose
    shape mirrors `/v1/messages` (model + messages) and returns
    `{"input_tokens": N}` on success; `assert_count_tokens_shape` checks that.

    The same cross-process token-bucket limiter `cli_driver.run_claude` uses is
    acquired here too, so probe rows count against the aggregate per-provider
    budget. `rate_limiter` is an injection seam for unit tests; production callers
    should leave it unset to use the process-wide default."""
    _acquire(model, rate_limiter)
    return gateway.transport.send(
        "/v1/messages/count_tokens",
        headers=_probe_headers(gateway),
        json=CountTokensRequest(
            model=model, messages=[ChatMessage(role="user", content=message)]
        ),
    )


def probe_tool_search(
    *,
    gateway: Gateway,
    model: str,
    rate_limiter: RateLimiter | None = None,
) -> StreamingResponse:
    """POST to `/v1/messages` with a `tool_search_tool_regex_20251119` tool and
    return the raw outcome.

    The tools array is the one Claude Code emits when its MCP-tool-search beta is
    active: a discovery tool plus at least one regular user tool. LiteLLM keys on
    the `_20251119` type string to attach the provider-specific beta header
    (`advanced-tool-use-2025-11-20` for Anthropic/Azure,
    `tool-search-tool-2025-10-19` for Vertex/Bedrock); a regression in that
    translation surfaces as a 400 from the upstream.

    The prompt deliberately does not force a tool call -- the goal is to verify
    the request round-trips without 400, not that the model chose to invoke
    tool_search. Like `probe_count_tokens`, this acquires one rate-limit token."""
    _acquire(model, rate_limiter)
    return gateway.transport.send(
        "/v1/messages",
        headers=_probe_headers(gateway),
        json=ToolSearchRequest(
            model=model,
            max_tokens=64,
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "If you have a tool to discover other tools, use it to "
                        "find one. Otherwise reply with the word 'done'."
                    ),
                )
            ],
            tools=[
                ToolSearchDiscoveryTool(),
                UserTool(
                    name="add_numbers",
                    description="Add two integers",
                    input_schema=ToolInputSchema(
                        properties={
                            "a": ToolInputProperty(type="integer"),
                            "b": ToolInputProperty(type="integer"),
                        },
                        required=["a", "b"],
                    ),
                ),
            ],
        ),
    )


def assert_count_tokens_shape(result: StreamingResponse) -> str | None:
    """Return None on success, or an error string describing the first violation.

    Acceptance criteria are intentionally minimal: HTTP status 200, a valid JSON
    body, and an `input_tokens` key whose value is a positive int. Anything beyond
    that (cache token fields, server metadata) varies by provider/transport and
    would make the cell flip red on neutral protocol drift."""
    if not result.ok:
        return _status_error(result)
    try:
        parsed = CountTokensResult.model_validate_json(result.body)
    except ValidationError as exc:
        return f"invalid count_tokens body ({result.body[:200]}): {exc}"
    if parsed.input_tokens <= 0:
        return f"input_tokens must be positive; got {parsed.input_tokens}"
    return None


def assert_tool_search_shape(result: StreamingResponse) -> str | None:
    """Return None on success, else describe the first violation.

    Acceptance: HTTP status 200 (no 400 rejecting the tool_search type or a
    missing beta header), a valid JSON body, and either `content` (Anthropic
    passthrough) or `choices` (LiteLLM normalized OpenAI shape). The cell goes red
    when the upstream rejects the tool type, the proxy drops the beta header, or
    the response shape is unusable; a model deciding not to call tool_search is
    irrelevant."""
    if not result.ok:
        return _status_error(result)
    try:
        parsed = ToolSearchResponse.model_validate_json(result.body)
    except ValidationError as exc:
        return f"non-JSON or unparseable body ({result.body[:200]}): {exc}"
    if parsed.content is None and parsed.choices is None:
        return f"response has neither `content` nor `choices`: {result.body[:400]}"
    return None


def _status_error(result: StreamingResponse) -> str:
    if result.status_code < 0:
        return f"transport error: {result.body}"
    return f"status {result.status_code}: {result.body[:400]}"
