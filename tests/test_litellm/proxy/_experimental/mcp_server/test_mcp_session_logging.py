"""The MCP ``mcp-session-id`` is captured for tool-call logging so the otel span
can carry ``mcp.session.id``. Guards the header read against casing and absence."""

from litellm.proxy._experimental.mcp_server.server import _mcp_session_id_from_headers


def test_reads_session_id_case_insensitively():
    # Clients send varied casing (``Mcp-Session-Id``, ``mcp-session-id``); all resolve.
    assert _mcp_session_id_from_headers({"mcp-session-id": "s1"}) == "s1"
    assert _mcp_session_id_from_headers({"Mcp-Session-Id": "s2"}) == "s2"
    assert _mcp_session_id_from_headers({"MCP-SESSION-ID": "s3"}) == "s3"


def test_stateless_call_has_no_session_id():
    # No header (stateless request) and an empty value both yield None, not "".
    assert _mcp_session_id_from_headers({"authorization": "Bearer x"}) is None
    assert _mcp_session_id_from_headers({"mcp-session-id": ""}) is None
    assert _mcp_session_id_from_headers(None) is None
    assert _mcp_session_id_from_headers({}) is None
