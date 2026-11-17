"""Tests for litellm/llms/a2a/common_utils.py."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

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


@pytest.mark.parametrize(
    ("api_key", "expected"),
    (
        ("secret", "Bearer secret"),
        ("Bearer secret", "Bearer Bearer secret"),
        ("os.environ/SYNTHETIC_KEY", "Bearer os.environ/SYNTHETIC_KEY"),
        ("", None),
        (None, None),
        (42, None),
    ),
)
async def test_plain_agent_key_is_used_verbatim_only_for_nonempty_strings(
    api_key: object, expected: str | None
) -> None:
    header: Final = await resolve_a2a_hop_auth_header({"api_key": api_key}, None)
    assert header == ({"Authorization": expected} if expected is not None else None)


@pytest.mark.parametrize("provider", ("azure_ai", "databricks_agent"))
async def test_completion_bridge_api_key_is_not_used_for_a2a_hop(provider: str) -> None:
    assert await resolve_a2a_hop_auth_header({"api_key": "secret"}, provider) is None


async def test_entra_bearer_overrides_agent_key_and_passthrough() -> None:
    resolver: Final = _RecordingEntraResolver()
    params: Final = {**_SERVICE_PRINCIPAL, "api_key": "secret"}
    header: Final = await resolve_a2a_hop_auth_header(params, None, resolver, {"authorization": "Bearer forwarded"})
    assert header == {"Authorization": "Bearer minted-entra-token"}
    assert resolver.calls == [params]


@pytest.mark.parametrize("name", ("Authorization", "aUtHoRiZaTiOn"))
async def test_passthrough_authorization_takes_precedence_over_plain_agent_key(name: str) -> None:
    assert (
        await resolve_a2a_hop_auth_header({"api_key": "secret"}, None, agent_extra_headers={name: "Bearer forwarded"})
        is None
    )
