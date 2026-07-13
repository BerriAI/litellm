"""Fault taxonomy for upstream OAuth token and DCR registration failures.

Each fault is a frozen model on a ``tag`` literal. The tag alone decides the HTTP status, the wire
error code, and whose prose the caller sees, so those three facts can never disagree the way they can
when an upstream's status and error code are relayed independently.
"""

from __future__ import annotations

from typing import Literal, Optional, TypeAlias

from pydantic import BaseModel, ConfigDict

MAX_WIRE_FIELD_CHARS = 500
"""Bound on every upstream-derived string that crosses to a caller or into a log line."""

CredentialSource: TypeAlias = Literal["gateway_stored", "caller_supplied"]
"""Whose client credentials the gateway presented upstream: the MCP server's stored configuration or
credentials the caller supplied on the request. Decides whether a credential rejection is the
caller's problem to fix or the gateway operator's."""

GATEWAY_RESPONSIBILITY_CODES: frozenset[str] = frozenset({"invalid_client", "unauthorized_client", "invalid_target"})
"""RFC 6749 error codes that indict the OAuth client itself (its credentials, its grant
authorization, or a gateway capability such as RFC 8707 resource indicators). When the gateway
presented its own stored credentials, these are gateway-side faults the caller cannot act on."""


class CallerRejected(BaseModel):
    """The upstream spoke the OAuth error contract and the failure is actionable by our caller
    (e.g. ``invalid_grant``: re-run authorization). The code and its bounded prose relay on the
    4xx status the code itself implies."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["caller_rejected"] = "caller_rejected"
    code: str
    description: Optional[str] = None
    error_uri: Optional[str] = None


class GatewayCredentialsRejected(BaseModel):
    """The upstream rejected the gateway's own stored client credentials or a gateway capability.
    Not actionable by the caller: rendered as 502 with gateway-authored prose naming the code;
    the upstream's prose goes to server logs only."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["gateway_credentials_rejected"] = "gateway_credentials_rejected"
    code: str


class UpstreamProtocolFault(BaseModel):
    """The upstream broke the error contract: no JSON ``error`` field, an undecodable body, or a
    success response without a usable token. Rendered as 502 with a gateway-authored note; the
    upstream body never crosses to the caller."""

    model_config = ConfigDict(frozen=True)
    tag: Literal["upstream_protocol_fault"] = "upstream_protocol_fault"
    note: str


UpstreamOAuthFault: TypeAlias = CallerRejected | GatewayCredentialsRejected | UpstreamProtocolFault
