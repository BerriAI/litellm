"""Per-server outcomes for the aggregate MCP tools/list fan-out.

The aggregate listing deliberately keeps serving the healthy subset when one server fails, but a
failed server must contribute a classified outcome instead of silently shrinking the list: an empty
contribution with no signal makes a broken upstream indistinguishable from a healthy server with no
tools. Outcomes carry only machine fields (category and status code) so nothing from an upstream
body crosses the trust boundary; classification is total, so any exception out of a server fetch
becomes an outcome, never a second failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, NamedTuple, NoReturn, TypeAlias

import httpx
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, ConfigDict
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.exceptions import (
    MCPServerListError,
    MCPUpstreamAuthError,
)

ListFaultCategory: TypeAlias = Literal[
    "auth_required",
    "forbidden",
    "timeout",
    "unreachable",
    "upstream_error",
    "internal",
]


class ServerListOk(BaseModel):
    model_config = ConfigDict(frozen=True)
    tag: Literal["ok"] = "ok"
    tool_count: int


class ServerListFault(BaseModel):
    """Why a server contributed nothing to a listing: the caller must authenticate upstream
    (``auth_required``/``forbidden``), the upstream did not answer (``timeout``/``unreachable``),
    the upstream answered outside its contract (``upstream_error``), or the gateway itself failed
    (``internal``). ``status_code`` is the upstream HTTP status when one exists."""

    model_config = ConfigDict(frozen=True)
    tag: ListFaultCategory
    status_code: int | None = None


ServerOutcome: TypeAlias = ServerListOk | ServerListFault

SERVER_OUTCOMES_META_KEY = "litellm.ai/server_outcomes"
"""The tools/list result ``_meta`` key carrying per-server outcomes. Prefixed with the litellm.ai
domain per the MCP spec's ``_meta`` key format so it cannot collide with spec-reserved names."""


class AggregateToolListing(NamedTuple):
    tools: list[MCPTool]
    outcomes: dict[str, ServerOutcome]


def _iter_upstream_responses(exc: BaseException) -> Iterator[httpx.Response]:
    """Yield every ``httpx.Response`` in the exception tree (``__cause__``/``__context__``/
    ExceptionGroup members) in deliberate order, mirroring how upstream failures surface through the
    MCP SDK's task groups. Explicit links come first: each node's ``raise ... from`` cause, then
    group members in raise order, then the incidental ``__context__`` chain, so a response raised
    while handling the real failure can never shadow one on the explicit causal chain. Consumers
    apply their own predicate over the stream: selecting the first response and THEN testing it
    would miss a causal auth response sitting behind an unrelated earlier one."""
    seen: set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        response = getattr(current, "response", None)
        if isinstance(response, httpx.Response):
            yield response
        if current.__context__ is not None:
            stack.append(current.__context__)
        exceptions = getattr(current, "exceptions", None)
        if isinstance(exceptions, tuple):
            stack.extend(reversed(exceptions))
        if current.__cause__ is not None:
            stack.append(current.__cause__)


def _find_upstream_response(exc: BaseException) -> httpx.Response | None:
    return next(_iter_upstream_responses(exc), None)


def upstream_auth_challenge(exc: BaseException) -> tuple[int, str | None] | None:
    """The first upstream 401/403 in deliberate order and its ``WWW-Authenticate`` challenge, both
    read from the SAME response, so the status that picks the carrier channel and the challenge that
    rides with it can never come from two different responses in the tree. Non-auth responses do not
    end the scan: a causal 401 behind an unrelated 5xx must still be found, or the client never
    receives the challenge it needs to re-authenticate."""
    for response in _iter_upstream_responses(exc):
        if response.status_code in (401, 403):
            return response.status_code, response.headers.get("www-authenticate")
    return None


def raise_classified_list_failure(
    exc: BaseException,
    server_name: str,
    suppress_challenge: bool = False,
) -> NoReturn:
    """The one place a failed server fetch chooses its carrier: an upstream 401/403 travels as
    ``MCPUpstreamAuthError`` with the upstream's own challenge preserved (a challenge is only ever
    fabricated at the HTTP edge, and only for a 401), everything else as ``MCPServerListError`` with
    a classified fault. Every fetch site delegates here so the two channels cannot drift apart per
    call site. ``suppress_challenge`` is for dcr_bridge servers, whose upstream challenge points
    clients at the wrong protected-resource metadata and must never relay."""
    auth = upstream_auth_challenge(exc)
    if auth is not None:
        status_code, challenge = auth
        raise MCPUpstreamAuthError(
            status_code=status_code,
            www_authenticate=None if suppress_challenge else challenge,
            server_name=server_name,
        ) from exc
    raise MCPServerListError(classify_list_exception(exc), server_name) from exc


def classify_list_exception(exc: BaseException) -> ServerListFault:
    """Classify a per-server listing failure into exactly one outcome. Total: an exception this
    function cannot recognize is the gateway's own fault (``internal``), never a re-raise."""
    if isinstance(exc, MCPServerListError) and isinstance(exc.fault, ServerListFault):
        return exc.fault
    if isinstance(exc, MCPUpstreamAuthError):
        tag = "forbidden" if exc.status_code == 403 else "auth_required"
        return ServerListFault(tag=tag, status_code=exc.status_code)
    if isinstance(exc, TimeoutError):
        return ServerListFault(tag="timeout")
    if isinstance(exc, ConnectionError):
        return ServerListFault(tag="unreachable")
    auth = upstream_auth_challenge(exc)
    if auth is not None:
        status_code, _ = auth
        return ServerListFault(
            tag="forbidden" if status_code == 403 else "auth_required",
            status_code=status_code,
        )
    response = _find_upstream_response(exc)
    if response is not None:
        return ServerListFault(tag="upstream_error", status_code=response.status_code)
    if isinstance(exc, (httpx.TimeoutException,)):
        return ServerListFault(tag="timeout")
    if isinstance(exc, httpx.TransportError):
        return ServerListFault(tag="unreachable")
    return ServerListFault(tag="internal")


def outcome_wire_value(outcome: ServerOutcome) -> dict[str, object]:
    """The client-visible form of one outcome, for the tools/list result ``_meta`` and the REST
    response: category plus status code only, never upstream prose or URLs."""
    match outcome.tag:
        case "ok":
            return {"status": "ok", "tool_count": outcome.tool_count}
        case "auth_required" | "forbidden" | "timeout" | "unreachable" | "upstream_error" | "internal":
            return {
                "status": outcome.tag,
                **({"http_status": outcome.status_code} if outcome.status_code is not None else {}),
            }
        case _:
            assert_never(outcome.tag)


def list_fault_http_status(fault: ServerListFault) -> int:
    """The truthful HTTP status for a single-upstream listing fault per RFC 9110: the upstream's own
    401/403 for auth, 504 for a timeout, 502 for an unreachable or misbehaving upstream, and 500 only
    for the gateway's own failure."""
    match fault.tag:
        case "auth_required":
            return fault.status_code or 401
        case "forbidden":
            return 403
        case "timeout":
            return 504
        case "unreachable" | "upstream_error":
            return 502
        case "internal":
            return 500
        case _:
            assert_never(fault.tag)
