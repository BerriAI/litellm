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
