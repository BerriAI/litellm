import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.identity.extractors.client import extract_client_info


def _fake_request(headers, client_host=None):
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=client_host) if client_host else None,
    )


def test_falls_back_to_direct_peer_when_not_trusted():
    req = _fake_request({"x-forwarded-for": "9.9.9.9"}, client_host="10.0.0.1")
    info = extract_client_info(req, general_settings={})
    assert info.ip == "10.0.0.1"
    assert info.forwarded_chain == ["9.9.9.9"]


def test_uses_xff_first_hop_when_proxy_trusted():
    req = _fake_request(
        {"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, client_host="10.0.0.1"
    )
    settings = {
        "use_x_forwarded_for": True,
        "mcp_trusted_proxy_ranges": ["10.0.0.0/8"],
    }
    info = extract_client_info(req, general_settings=settings)
    assert info.ip == "1.2.3.4"
    assert info.forwarded_chain == ["1.2.3.4", "10.0.0.1"]


def test_no_xff_returns_direct_peer():
    req = _fake_request({}, client_host="127.0.0.1")
    info = extract_client_info(req, general_settings={})
    assert info.ip == "127.0.0.1"
    assert info.forwarded_chain == []


def test_user_agent_passthrough():
    req = _fake_request({"user-agent": "curl/8"}, client_host="127.0.0.1")
    info = extract_client_info(req, general_settings={})
    assert info.user_agent == "curl/8"
