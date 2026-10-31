"""Tests for litellm/llms/a2a/common_utils.py."""

from collections.abc import Mapping
from types import MappingProxyType

import pytest

from litellm.llms.a2a.common_utils import resolve_a2a_hop_auth_header


class _RecordingEntraResolver:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, object]] = []

    async def __call__(self, litellm_params: Mapping[str, object]) -> Mapping[str, str]:
        self.calls.append(litellm_params)
        return MappingProxyType({"Authorization": "Bearer minted-entra-token"})


_SERVICE_PRINCIPAL = MappingProxyType({"tenant_id": "tenant", "client_id": "client", "client_secret": "sp-secret"})


@pytest.mark.asyncio
async def test_entra_agent_gets_a_minted_bearer_for_the_a2a_hop():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(_SERVICE_PRINCIPAL, None, resolver)

    assert header == {"Authorization": "Bearer minted-entra-token"}
    assert resolver.calls == [_SERVICE_PRINCIPAL]


@pytest.mark.asyncio
async def test_completion_bridge_agent_keeps_its_entra_credentials_for_the_model_provider():
    """A bridged agent's tenant_id/client_id/client_secret authenticate the model it bridges to, so the A2A hop
    must not spend them on a bearer of its own."""
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(_SERVICE_PRINCIPAL, "azure_ai", resolver)

    assert header is None
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_agent_without_entra_credentials_gets_no_bearer():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header({"api_base": "https://agent.example.com"}, None, resolver)

    assert header is None
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_a2a_agent_with_api_key_gets_bearer_auth_header():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(
        {"api_base": "https://agent.example.com", "api_key": "secret-123"},
        None,
        resolver,
    )

    assert header == {"Authorization": "Bearer secret-123"}
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_a2a_agent_with_bearer_prefixed_api_key():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(
        {"api_base": "https://agent.example.com", "api_key": "Bearer already-bearer"},
        None,
        resolver,
    )

    assert header == {"Authorization": "Bearer already-bearer"}
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_a2a_agent_with_lowercase_bearer_prefixed_api_key():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(
        {"api_base": "https://agent.example.com", "api_key": "bearer lowercase-bearer"},
        None,
        resolver,
    )

    assert header == {"Authorization": "bearer lowercase-bearer"}
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_a2a_agent_with_os_environ_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MY_A2A_KEY", "env-resolved-key")
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(
        {"api_base": "https://agent.example.com", "api_key": "os.environ/MY_A2A_KEY"},
        None,
        resolver,
    )

    assert header == {"Authorization": "Bearer env-resolved-key"}
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_a2a_agent_with_missing_os_environ_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISSING_A2A_KEY", raising=False)
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(
        {"api_base": "https://agent.example.com", "api_key": "os.environ/MISSING_A2A_KEY"},
        None,
        resolver,
    )

    assert header is None
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_completion_bridge_agent_keeps_its_api_key_for_the_model_provider():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(
        {"api_base": "https://agent.example.com", "api_key": "openai-key"},
        "openai",
        resolver,
    )

    assert header is None
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_a2a_agent_with_static_auth_header_does_not_override_with_api_key():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(
        {"api_base": "https://agent.example.com", "api_key": "agent-key"},
        None,
        resolver,
        has_static_auth=True,
    )

    assert header is None
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_a2a_agent_with_stored_authorization_header_does_not_override_with_api_key():
    resolver = _RecordingEntraResolver()

    header = await resolve_a2a_hop_auth_header(
        {
            "api_base": "https://agent.example.com",
            "api_key": "agent-key",
            "headers": {"Authorization": "Basic dXNlcjpwYXNz"},
        },
        None,
        resolver,
    )

    assert header is None
    assert resolver.calls == []


@pytest.mark.asyncio
async def test_a2a_agent_with_empty_or_whitespace_api_key():
    resolver = _RecordingEntraResolver()

    header_empty = await resolve_a2a_hop_auth_header(
        {"api_base": "https://agent.example.com", "api_key": ""},
        None,
        resolver,
    )
    header_ws = await resolve_a2a_hop_auth_header(
        {"api_base": "https://agent.example.com", "api_key": "   "},
        None,
        resolver,
    )

    assert header_empty is None
    assert header_ws is None
    assert resolver.calls == []
