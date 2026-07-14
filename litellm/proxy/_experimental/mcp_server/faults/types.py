"""Fault taxonomy for upstream OAuth token and DCR registration failures.

Each fault is a frozen model on a ``tag`` literal. The tag alone decides the HTTP status, the wire
error code, and whose prose the caller sees, so those three facts can never disagree the way they can
when an upstream's status and error code are relayed independently.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

MAX_WIRE_FIELD_CHARS = 500
"""Bound on every upstream-derived string that crosses to a caller or into a log line."""

CredentialSource: TypeAlias = Literal["gateway_stored", "caller_supplied"]
"""Whose client credentials the gateway presented upstream: the MCP server's stored configuration or
credentials the caller supplied on the request. Decides whether a credential rejection is the
caller's problem to fix or the gateway operator's."""

GATEWAY_CREDENTIAL_CODES: frozenset[str] = frozenset({"invalid_client", "unauthorized_client"})
"""RFC 6749 error codes that indict the OAuth client's credentials or grant authorization. When the
gateway presented its own stored credentials, these are gateway-side faults the caller cannot act on;
when the caller supplied the credentials, they are the caller's to fix."""

GATEWAY_CAPABILITY_CODES: frozenset[str] = frozenset({"invalid_target"})
"""Codes that indict a gateway capability regardless of whose credentials were presented:
``invalid_target`` means the upstream wants RFC 8707 resource indicators, which the gateway does not
send yet (LIT-4339). Never the caller's fault."""

UPSTREAM_FAULT_CODES: frozenset[str] = frozenset({"server_error", "temporarily_unavailable"})
"""Codes by which the upstream blames itself. Relaying them as caller faults would invert blame, so
they classify as upstream-reported faults and render on the 5xx their meaning implies."""


class CallerRejected(BaseModel):
    """The upstream spoke the OAuth error contract and the failure is actionable by our caller
    (e.g. ``invalid_grant``: re-run authorization). The code and its bounded prose relay on the
    4xx status the code itself implies."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["caller_rejected"] = "caller_rejected"
    code: str
    description: str | None = None
    error_uri: str | None = None


class GatewayRejected(BaseModel):
    """The upstream rejected the request for a cause only the gateway operator can address: the
    server's stored client credentials or a gateway capability gap. Not actionable by the caller:
    rendered as 502 with gateway-authored prose naming the code; the upstream's prose goes to
    server logs only."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["gateway_rejected"] = "gateway_rejected"
    code: str


class UpstreamReportedFault(BaseModel):
    """The upstream blamed itself in the OAuth vocabulary. Rendered on the 5xx the code implies
    (``server_error`` 502, ``temporarily_unavailable`` 503) so blame and status agree."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["upstream_reported_fault"] = "upstream_reported_fault"
    code: Literal["server_error", "temporarily_unavailable"]


class UpstreamProtocolFault(BaseModel):
    """The upstream broke the error contract: no JSON ``error`` field, an undecodable body, or a
    success response without a usable token. Rendered as 502 with a gateway-authored note; the
    upstream body never crosses to the caller."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["upstream_protocol_fault"] = "upstream_protocol_fault"
    note: str


UpstreamOAuthFault: TypeAlias = CallerRejected | GatewayRejected | UpstreamReportedFault | UpstreamProtocolFault
