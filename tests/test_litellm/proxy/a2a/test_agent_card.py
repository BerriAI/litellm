"""Unit tests for the pure merge logic in litellm/proxy/a2a/agent_card.py."""

from litellm.proxy.a2a.agent_card import (
    LITELLM_A2A_PROTOCOL_VERSION,
    LITELLM_SECURITY_REQUIREMENTS,
    LITELLM_SECURITY_SCHEMES,
    merge_agent_card,
)

PROXY_URL = "https://proxy.example/a2a/agent-xyz"
PROXY_BASE = "https://proxy.example"


def _full_upstream_card() -> dict:
    return {
        "protocolVersion": "0.9",
        "name": "Upstream Name",
        "description": "Upstream description",
        "url": "http://internal:9999/",
        "version": "1.2.3",
        "capabilities": {
            "streaming": True,
            "pushNotifications": True,
            "stateTransitionHistory": True,
            "extensions": [{"uri": "x"}],
        },
        "skills": [
            {"id": "s1", "name": "skill one", "description": "d", "tags": ["t"]}
        ],
        "defaultInputModes": ["text", "audio"],
        "defaultOutputModes": ["text"],
        "securitySchemes": {"upstreamKey": {"type": "apiKey"}},
        "security": [{"upstreamKey": []}],
        "provider": {"organization": "UpstreamCo", "url": "https://upstream.example"},
        "iconUrl": "https://upstream.example/icon.png",
        "documentationUrl": "https://upstream.example/docs",
        "somethingNotInSchema": "should be stripped",
    }


def test_preserves_top_level_url_for_runtime_invocation():
    # The runtime A2A invocation path reads ``agent_card_params['url']`` to
    # know where to proxy requests, so the merge must keep the upstream URL
    # on the stored card. The public well-known endpoint rewrites this field
    # to the proxy URL before exposing it to clients.
    merged = merge_agent_card(
        _full_upstream_card(), proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE
    )
    assert merged["url"] == "http://internal:9999/"


def test_overrides_protocol_version():
    merged = merge_agent_card(
        _full_upstream_card(), proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE
    )
    assert merged["protocolVersion"] == LITELLM_A2A_PROTOCOL_VERSION


def test_overrides_name_and_description_when_provided():
    merged = merge_agent_card(
        _full_upstream_card(),
        proxy_url=PROXY_URL,
        proxy_base_url=PROXY_BASE,
        name="UI Name",
        description="UI Description",
    )
    assert merged["name"] == "UI Name"
    assert merged["description"] == "UI Description"


def test_keeps_upstream_name_and_description_when_not_overridden():
    merged = merge_agent_card(
        _full_upstream_card(), proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE
    )
    assert merged["name"] == "Upstream Name"
    assert merged["description"] == "Upstream description"


def test_filters_capabilities_to_allowlist():
    merged = merge_agent_card(
        _full_upstream_card(), proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE
    )
    # Only ``streaming`` is allowlisted today.
    assert merged["capabilities"] == {"streaming": True}


def test_drops_streaming_when_upstream_disables_it():
    upstream = _full_upstream_card()
    upstream["capabilities"]["streaming"] = False
    merged = merge_agent_card(upstream, proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE)
    assert merged["capabilities"] == {}


def test_replaces_security_schemes_and_requirements():
    merged = merge_agent_card(
        _full_upstream_card(), proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE
    )
    assert merged["securitySchemes"] == LITELLM_SECURITY_SCHEMES
    assert merged["security"] == LITELLM_SECURITY_REQUIREMENTS
    assert "securityRequirements" not in merged


def test_emits_supported_interfaces_pointing_at_proxy():
    merged = merge_agent_card(
        _full_upstream_card(), proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE
    )
    assert merged["supportedInterfaces"] == [
        {
            "url": PROXY_URL,
            "protocolBinding": "JSONRPC",
            "protocolVersion": LITELLM_A2A_PROTOCOL_VERSION,
        }
    ]


def test_passes_through_skills_modes_provider_icon_docs():
    merged = merge_agent_card(
        _full_upstream_card(), proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE
    )
    assert merged["skills"] == _full_upstream_card()["skills"]
    assert merged["defaultInputModes"] == ["text", "audio"]
    assert merged["defaultOutputModes"] == ["text"]
    assert merged["provider"] == {
        "organization": "UpstreamCo",
        "url": "https://upstream.example",
    }
    assert merged["iconUrl"] == "https://upstream.example/icon.png"
    assert merged["documentationUrl"] == "https://upstream.example/docs"


def test_strips_fields_not_in_v1_schema():
    merged = merge_agent_card(
        _full_upstream_card(), proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE
    )
    assert "somethingNotInSchema" not in merged


def test_defaults_for_missing_skills_and_modes():
    sparse = {"name": "x", "description": "y", "version": "1"}
    merged = merge_agent_card(sparse, proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE)
    assert merged["skills"] and merged["skills"][0]["id"] == "chat"
    assert merged["defaultInputModes"] == ["text"]
    assert merged["defaultOutputModes"] == ["text"]


def test_defaults_version_when_upstream_omits_it():
    sparse = {"name": "x", "description": "y"}
    merged = merge_agent_card(sparse, proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE)
    assert merged["version"] == "1.0.0"


def test_preserves_upstream_version_when_present():
    merged = merge_agent_card(
        _full_upstream_card(), proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE
    )
    assert merged["version"] == "1.2.3"


def test_falls_back_to_litellm_provider_when_upstream_lacks_one():
    sparse = {"name": "x", "description": "y", "version": "1"}
    merged = merge_agent_card(sparse, proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE)
    assert merged["provider"] == {
        "organization": "LiteLLM Proxy",
        "url": PROXY_BASE,
    }


def test_handles_none_upstream_card():
    merged = merge_agent_card(None, proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE)
    assert merged["protocolVersion"] == LITELLM_A2A_PROTOCOL_VERSION
    assert merged["supportedInterfaces"][0]["url"] == PROXY_URL
    assert merged["securitySchemes"] == LITELLM_SECURITY_SCHEMES


def test_does_not_mutate_input():
    upstream = _full_upstream_card()
    snapshot = dict(upstream)
    merge_agent_card(upstream, proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE)
    assert upstream == snapshot


def test_strips_additional_interfaces_to_prevent_backend_url_leak():
    upstream = _full_upstream_card()
    upstream["additionalInterfaces"] = [
        {"url": "http://internal-backend:8080/", "transport": "JSONRPC"},
        {"url": "grpc://internal-backend:50051", "transport": "GRPC"},
    ]
    merged = merge_agent_card(upstream, proxy_url=PROXY_URL, proxy_base_url=PROXY_BASE)
    assert "additionalInterfaces" not in merged
