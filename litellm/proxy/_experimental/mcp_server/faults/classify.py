"""The single place that reads upstream OAuth/DCR failure responses.

Every accessor here is total: an upstream that lies about its content encoding, sends an undecodable
body, or omits the spec fields yields a classified fault, never an exception. Nothing outside this
module should touch a failed upstream response's body.
"""

from __future__ import annotations

import httpx

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.faults.types import (
    GATEWAY_CAPABILITY_CODES,
    GATEWAY_CREDENTIAL_CODES,
    MAX_WIRE_FIELD_CHARS,
    CallerRejected,
    CredentialSource,
    GatewayRejected,
    UpstreamOAuthFault,
    UpstreamProtocolFault,
    UpstreamReportedFault,
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


def _classify_oauth_error_code(
    code: str,
    description: str | None,
    error_uri: str | None,
    credential_source: CredentialSource,
    log_context: str,
) -> UpstreamOAuthFault:
    """Blame assignment for a contract-conformant OAuth error code, shared by the token and DCR
    classifiers. Codes by which the upstream blames itself keep that blame; ``invalid_target`` is a
    gateway capability gap (RFC 8707 resource indicators, LIT-4339) no matter whose credentials were
    presented; credential-indicting codes follow the credential source; everything else, including
    codes we do not recognize, is the caller's to act on. The upstream's HTTP status is deliberately
    never consulted: status derives from this classification at render time, which is what keeps
    status and code from contradicting each other."""
    if code == "server_error" or code == "temporarily_unavailable":
        return UpstreamReportedFault(code=code)
    if code in GATEWAY_CAPABILITY_CODES:
        verbose_logger.warning(
            "MCP server %s: the upstream authorization server rejected the request with "
            "invalid_target; it may require RFC 8707 resource indicators, which the gateway "
            "does not send yet (tracked as LIT-4339)",
            log_context,
        )
        return GatewayRejected(code=code)
    if credential_source == "gateway_stored" and code in GATEWAY_CREDENTIAL_CODES:
        verbose_logger.warning(
            "MCP server %s: upstream authorization server rejected the gateway's configured client "
            "credentials (%s): %s",
            log_context,
            code,
            description or "<no description>",
        )
        return GatewayRejected(code=code)
    return CallerRejected(code=code, description=description, error_uri=error_uri)


def classify_upstream_token_rejection(
    response: httpx.Response,
    credential_source: CredentialSource,
    log_context: str,
) -> UpstreamOAuthFault:
    """Classify a token-endpoint rejection into exactly one fault: a body with an RFC 6749 §5.2
    ``error`` field goes through blame assignment (:func:`_classify_oauth_error_code`); anything
    without a usable ``error`` field is an upstream protocol fault."""
    parsed = _safe_json(response)
    fields = parsed if isinstance(parsed, dict) else {}
    code = _bounded_field(fields.get("error"))
    if code is None:
        _log_out_of_contract("token", response, log_context)
        return UpstreamProtocolFault(note=f"upstream token endpoint returned HTTP {response.status_code}")
    return _classify_oauth_error_code(
        code,
        description=_bounded_field(fields.get("error_description")),
        error_uri=_bounded_field(fields.get("error_uri")),
        credential_source=credential_source,
        log_context=log_context,
    )


def classify_upstream_dcr_rejection(response: httpx.Response, log_context: str) -> UpstreamOAuthFault:
    """Classify a dynamic-client-registration rejection. RFC 7591 §3.2.2 errors carry
    ``error`` / ``error_description`` and go through the same blame assignment as token errors
    (registration sends no client credentials, so credential codes stay caller-actionable); anything
    without a usable ``error`` field is an upstream protocol fault."""
    parsed = _safe_json(response)
    fields = parsed if isinstance(parsed, dict) else {}
    code = _bounded_field(fields.get("error"))
    if code is None:
        _log_out_of_contract("registration", response, log_context)
        return UpstreamProtocolFault(note=f"upstream registration failed with HTTP {response.status_code}")
    return _classify_oauth_error_code(
        code,
        description=_bounded_field(fields.get("error_description")),
        error_uri=None,
        credential_source="caller_supplied",
        log_context=log_context,
    )
