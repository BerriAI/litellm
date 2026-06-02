"""Mocked tests for the `NoopProvider`."""

from __future__ import annotations

import pytest

from litellm.proxy.agent_session_endpoints.vm_providers import (
    NoopProvider,
    ProvisionContext,
    VMState,
)


@pytest.mark.asyncio
async def test_noop_provision_returns_handle():
    provider = NoopProvider()
    ctx = ProvisionContext(session_id="sess-1", team_id="team-1")
    handle = await provider.provision(ctx)
    assert handle.vm_id.startswith("noop-")
    assert handle.provider == "noop"
    assert handle.metadata["session_id"] == "sess-1"
    assert handle.metadata["team_id"] == "team-1"


@pytest.mark.asyncio
async def test_noop_status_lifecycle():
    provider = NoopProvider()
    ctx = ProvisionContext(session_id="sess-1", team_id="team-1")
    handle = await provider.provision(ctx)

    status = await provider.status(handle)
    assert status.state == VMState.RUNNING
    assert status.public_ip == "127.0.0.1"

    await provider.terminate(handle)
    status_after = await provider.status(handle)
    assert status_after.state == VMState.TERMINATED


@pytest.mark.asyncio
async def test_noop_terminate_idempotent():
    provider = NoopProvider()
    handle = await provider.provision(ProvisionContext(session_id="s", team_id="t"))
    await provider.terminate(handle)
    # Second call must not raise.
    await provider.terminate(handle)
