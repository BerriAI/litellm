"""Render upstream OAuth/DCR faults onto the wire. The only place that chooses statuses and bodies
for these faults, so every consumer emits the same contract: RFC 6749 §5.2-shaped JSON with the §5.1
no-store headers on token endpoints, HTTPException details on registration. Status, code, and prose
all derive from the fault tag; exhaustive matches keep a new fault arm from shipping unrendered.
"""

from __future__ import annotations

from fastapi.responses import JSONResponse
from typing_extensions import assert_never

from litellm.proxy._experimental.mcp_server.faults.types import UpstreamOAuthFault
from litellm.proxy._experimental.mcp_server.oauth_utils import TOKEN_NO_CACHE_HEADERS


def _gateway_credentials_description(code: str) -> str:
    if code == "invalid_target":
        return (
            "the upstream authorization server rejected the token request (invalid_target); "
            "it may require RFC 8707 resource indicators, which the gateway does not send yet"
        )
    return (
        f"the upstream authorization server rejected the gateway's configured client credentials "
        f"({code}); verify the MCP server's client_id and client_secret"
    )


def render_token_fault(fault: UpstreamOAuthFault) -> JSONResponse:
    """RFC 6749 §5.2 response for a token-endpoint fault. Caller-actionable rejections relay the
    upstream's code on the status that code implies (401 for invalid_client per §5.2, else 400);
    gateway-side faults are 502 ``server_error`` with gateway-authored prose so a caller is never
    blamed for, or shown the internals of, a failure only the operator can fix."""
    match fault.tag:
        case "caller_rejected":
            content = {
                "error": fault.code,
                **({"error_description": fault.description} if fault.description else {}),
                **({"error_uri": fault.error_uri} if fault.error_uri else {}),
            }
            status_code = 401 if fault.code == "invalid_client" else 400
            return JSONResponse(status_code=status_code, content=content, headers=TOKEN_NO_CACHE_HEADERS)
        case "gateway_credentials_rejected":
            return JSONResponse(
                status_code=502,
                content={
                    "error": "server_error",
                    "error_description": _gateway_credentials_description(fault.code),
                },
                headers=TOKEN_NO_CACHE_HEADERS,
            )
        case "upstream_protocol_fault":
            return JSONResponse(
                status_code=502,
                content={"error": "server_error", "error_description": fault.note},
                headers=TOKEN_NO_CACHE_HEADERS,
            )
        case _:
            assert_never(fault.tag)


def dcr_fault_detail(fault: UpstreamOAuthFault) -> "tuple[int, str]":
    """Status and detail string for a registration fault, raised as HTTPException by the caller.
    RFC 7591 §3.2.2 defines registration errors as 400, so a contract-conformant rejection is 400
    regardless of the status the upstream chose; everything else is a 502 upstream fault."""
    match fault.tag:
        case "caller_rejected":
            detail = f"{fault.code}: {fault.description}" if fault.description else fault.code
            return 400, detail
        case "gateway_credentials_rejected":
            return 502, _gateway_credentials_description(fault.code)
        case "upstream_protocol_fault":
            return 502, fault.note
        case _:
            assert_never(fault.tag)
