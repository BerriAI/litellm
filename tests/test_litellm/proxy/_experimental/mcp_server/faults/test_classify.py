"""Classification matrix for upstream OAuth/DCR rejections: who is blamed depends only on the §5.2
code and whose credentials the gateway presented, never on the upstream's HTTP status."""

import httpx

from litellm.proxy._experimental.mcp_server.faults.classify import (
    classify_upstream_dcr_rejection,
    classify_upstream_token_rejection,
)
from litellm.proxy._experimental.mcp_server.faults.types import (
    CallerRejected,
    GatewayCredentialsRejected,
    UpstreamProtocolFault,
)


def _response(status_code: int, *, json_body: object = None, text_body: str = "", headers: dict = None) -> httpx.Response:
    request = httpx.Request("POST", "https://idp.example.com/token")
    if json_body is not None:
        return httpx.Response(status_code, json=json_body, request=request)
    return httpx.Response(status_code, text=text_body, headers=headers or {}, request=request)


def test_caller_fault_code_classifies_as_caller_rejected_regardless_of_status():
    fault = classify_upstream_token_rejection(
        _response(500, json_body={"error": "invalid_grant", "error_description": "Code expired."}),
        credential_source="gateway_stored",
        log_context="srv",
    )
    assert isinstance(fault, CallerRejected)
    assert fault.code == "invalid_grant"
    assert fault.description == "Code expired."


def test_credential_code_with_gateway_stored_credentials_indicts_gateway():
    fault = classify_upstream_token_rejection(
        _response(401, json_body={"error": "invalid_client", "error_description": "not found"}),
        credential_source="gateway_stored",
        log_context="srv",
    )
    assert isinstance(fault, GatewayCredentialsRejected)
    assert fault.code == "invalid_client"


def test_credential_code_with_caller_supplied_credentials_stays_caller_fault():
    fault = classify_upstream_token_rejection(
        _response(401, json_body={"error": "invalid_client"}),
        credential_source="caller_supplied",
        log_context="srv",
    )
    assert isinstance(fault, CallerRejected)
    assert fault.code == "invalid_client"


def test_unknown_code_relays_as_caller_rejected():
    fault = classify_upstream_token_rejection(
        _response(400, json_body={"error": "slow_down", "error_description": "Polling too fast."}),
        credential_source="gateway_stored",
        log_context="srv",
    )
    assert isinstance(fault, CallerRejected)
    assert fault.code == "slow_down"


def test_body_without_error_field_is_protocol_fault():
    fault = classify_upstream_token_rejection(
        _response(404, text_body="<html>not here</html>"),
        credential_source="gateway_stored",
        log_context="srv",
    )
    assert isinstance(fault, UpstreamProtocolFault)
    assert fault.note == "upstream token endpoint returned HTTP 404"


def test_unreadable_body_is_protocol_fault_not_exception():
    unreadable = httpx.Response(
        400,
        stream=httpx.ByteStream(b"\x1f\x8bnot-gzip"),
        headers={"content-encoding": "gzip"},
        request=httpx.Request("POST", "https://idp.example.com/token"),
    )
    fault = classify_upstream_token_rejection(unreadable, credential_source="gateway_stored", log_context="srv")
    assert isinstance(fault, UpstreamProtocolFault)


def test_wire_fields_are_bounded():
    fault = classify_upstream_token_rejection(
        _response(400, json_body={"error": "invalid_request", "error_description": "x" * 5000}),
        credential_source="gateway_stored",
        log_context="srv",
    )
    assert isinstance(fault, CallerRejected)
    assert len(fault.description) == 500


def test_dcr_rejection_with_rfc7591_code_is_caller_rejected():
    fault = classify_upstream_dcr_rejection(
        _response(400, json_body={"error": "invalid_redirect_uri", "error_description": "not allowed"}),
        log_context="srv",
    )
    assert isinstance(fault, CallerRejected)
    assert fault.code == "invalid_redirect_uri"


def test_dcr_rejection_without_code_is_protocol_fault():
    fault = classify_upstream_dcr_rejection(_response(500, text_body="<html>trace</html>"), log_context="srv")
    assert isinstance(fault, UpstreamProtocolFault)
    assert fault.note == "upstream registration failed with HTTP 500"
