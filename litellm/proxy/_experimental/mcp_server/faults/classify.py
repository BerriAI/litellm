"""The single place that reads upstream OAuth/DCR failure responses.

Every accessor here is total: an upstream that lies about its content encoding, sends an undecodable
body, or omits the spec fields yields a classified fault, never an exception. Nothing outside this
module should touch a failed upstream response's body.
"""

from __future__ import annotations

import httpx

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.faults.types import (
    GATEWAY_RESPONSIBILITY_CODES,
    MAX_WIRE_FIELD_CHARS,
    CallerRejected,
    CredentialSource,
    GatewayCredentialsRejected,
    UpstreamOAuthFault,
    UpstreamProtocolFault,
)


def _safe_text(response: httpx.Response) -> str:
    try:
        return response.text
    except Exception:
        return ""


def _safe_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except Exception:
        return None


def _bounded_field(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:MAX_WIRE_FIELD_CHARS]


def _log_out_of_contract(endpoint_kind: str, response: httpx.Response, log_context: str) -> None:
    verbose_logger.warning(
        "MCP upstream %s endpoint (%s) returned HTTP %s outside the OAuth error contract (first %s chars): %s",
        endpoint_kind,
        log_context,
        response.status_code,
        MAX_WIRE_FIELD_CHARS,
        _safe_text(response)[:MAX_WIRE_FIELD_CHARS],
    )


def classify_upstream_token_rejection(
    response: httpx.Response,
    credential_source: CredentialSource,
    log_context: str,
) -> UpstreamOAuthFault:
    """Classify a token-endpoint rejection into exactly one fault.

    A body with an RFC 6749 §5.2 ``error`` field is a contract-conformant rejection: it indicts the
    gateway when the code blames the client credentials the gateway itself presented
    (``GATEWAY_RESPONSIBILITY_CODES`` with ``gateway_stored`` credentials), and the caller otherwise.
    The upstream's HTTP status is deliberately not consulted for blame: status is derived from the
    classification at render time, which is what keeps status and code from contradicting each other.
    Anything without a usable ``error`` field is an upstream protocol fault."""
    parsed = _safe_json(response)
    fields = parsed if isinstance(parsed, dict) else {}
    code = _bounded_field(fields.get("error"))
    if code is None:
        _log_out_of_contract("token", response, log_context)
        return UpstreamProtocolFault(note=f"upstream token endpoint returned HTTP {response.status_code}")
    if code == "invalid_target":
        verbose_logger.warning(
            "MCP server %s: the upstream authorization server rejected the token request with "
            "invalid_target; it may require RFC 8707 resource indicators, which the gateway "
            "does not send yet (tracked as LIT-4339)",
            log_context,
        )
    if credential_source == "gateway_stored" and code in GATEWAY_RESPONSIBILITY_CODES:
        verbose_logger.warning(
            "MCP server %s: upstream authorization server rejected the gateway's configured client "
            "credentials (%s): %s",
            log_context,
            code,
            _bounded_field(fields.get("error_description")) or "<no description>",
        )
        return GatewayCredentialsRejected(code=code)
    return CallerRejected(
        code=code,
        description=_bounded_field(fields.get("error_description")),
        error_uri=_bounded_field(fields.get("error_uri")),
    )


def classify_upstream_dcr_rejection(response: httpx.Response, log_context: str) -> UpstreamOAuthFault:
    """Classify a dynamic-client-registration rejection. RFC 7591 §3.2.2 errors carry
    ``error`` / ``error_description`` and are caller-actionable (the registration metadata was
    rejected); anything else is an upstream protocol fault."""
    parsed = _safe_json(response)
    fields = parsed if isinstance(parsed, dict) else {}
    code = _bounded_field(fields.get("error"))
    if code is None:
        _log_out_of_contract("registration", response, log_context)
        return UpstreamProtocolFault(note=f"upstream registration failed with HTTP {response.status_code}")
    return CallerRejected(code=code, description=_bounded_field(fields.get("error_description")))
