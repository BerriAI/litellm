"""Rendering contract: status, wire code, and prose all derive from the fault tag, so a caller-fault
code can never ship on a server-fault status and gateway-side faults never carry provider prose."""

import json

from litellm.proxy._experimental.mcp_server.faults.render_oauth import (
    dcr_fault_detail,
    render_token_fault,
)
from litellm.proxy._experimental.mcp_server.faults.types import (
    CallerRejected,
    GatewayRejected,
    UpstreamProtocolFault,
    UpstreamReportedFault,
)


def test_caller_rejected_renders_code_derived_status():
    response = render_token_fault(CallerRejected(code="invalid_grant", description="Code expired."))
    assert response.status_code == 400
    assert json.loads(response.body) == {"error": "invalid_grant", "error_description": "Code expired."}
    assert response.headers["cache-control"] == "no-store"


def test_caller_rejected_invalid_client_renders_401():
    response = render_token_fault(CallerRejected(code="invalid_client"))
    assert response.status_code == 401
    assert json.loads(response.body) == {"error": "invalid_client"}


def test_caller_rejected_includes_error_uri_only_when_present():
    response = render_token_fault(
        CallerRejected(code="invalid_scope", description="bad scope", error_uri="https://idp.example.com/e")
    )
    assert json.loads(response.body) == {
        "error": "invalid_scope",
        "error_description": "bad scope",
        "error_uri": "https://idp.example.com/e",
    }


def test_gateway_rejected_renders_502_with_gateway_prose():
    response = render_token_fault(GatewayRejected(code="invalid_client"))
    assert response.status_code == 502
    body = json.loads(response.body)
    assert body["error"] == "server_error"
    assert "invalid_client" in body["error_description"]
    assert "client_id and client_secret" in body["error_description"]


def test_gateway_invalid_target_prose_names_resource_indicators():
    response = render_token_fault(GatewayRejected(code="invalid_target"))
    body = json.loads(response.body)
    assert response.status_code == 502
    assert "RFC 8707" in body["error_description"]


def test_protocol_fault_renders_502_note():
    response = render_token_fault(UpstreamProtocolFault(note="upstream token endpoint returned HTTP 503"))
    assert response.status_code == 502
    assert json.loads(response.body) == {
        "error": "server_error",
        "error_description": "upstream token endpoint returned HTTP 503",
    }


def test_dcr_caller_rejection_is_400_per_rfc7591_regardless_of_upstream_status():
    status_code, detail = dcr_fault_detail(CallerRejected(code="invalid_client_metadata", description="bad grant types"))
    assert status_code == 400
    assert detail == "invalid_client_metadata: bad grant types"


def test_dcr_protocol_fault_is_502():
    status_code, detail = dcr_fault_detail(UpstreamProtocolFault(note="upstream registration failed with HTTP 500"))
    assert status_code == 502
    assert detail == "upstream registration failed with HTTP 500"


def test_upstream_reported_server_error_renders_502_with_matching_code():
    response = render_token_fault(UpstreamReportedFault(code="server_error"))
    assert response.status_code == 502
    assert json.loads(response.body)["error"] == "server_error"


def test_upstream_reported_temporarily_unavailable_renders_503_with_matching_code():
    response = render_token_fault(UpstreamReportedFault(code="temporarily_unavailable"))
    assert response.status_code == 503
    body = json.loads(response.body)
    assert body["error"] == "temporarily_unavailable"
    assert "retry" in body["error_description"]


def test_dcr_upstream_reported_fault_maps_to_5xx():
    status_code, detail = dcr_fault_detail(UpstreamReportedFault(code="server_error"))
    assert status_code == 502
    assert "internal error" in detail
