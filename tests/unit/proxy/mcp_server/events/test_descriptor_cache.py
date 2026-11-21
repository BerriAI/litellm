from typing import Final

import pytest

from litellm.mcp_client.client import MCPClient
from litellm.proxy.mcp_server.events.runtime import DescriptorCache
from litellm.proxy.mcp_server.subscriptions import SubscriptionTarget
from litellm.types.mcp_server.events import EventType

from .fakes import BUILD_FAILED, server


def _target(server_id: str = "srv-ci", credential: str = "credential") -> SubscriptionTarget:
    return SubscriptionTarget(
        server(server_id=server_id), MCPClient(server_url="https://upstream.example/mcp"), credential
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Upstream:
    def __init__(self) -> None:
        self.loads: list[tuple[str, str]] = []  # mutable-ok: records each upstream listing

    async def load(self, target: SubscriptionTarget) -> tuple[EventType, ...]:
        self.loads.append((target.server.server_id, target.credential))
        return (BUILD_FAILED,)


@pytest.mark.asyncio
async def test_a_listing_is_reused_until_it_expires() -> None:
    clock: Final = _Clock()
    upstream: Final = _Upstream()
    cache: Final = DescriptorCache(clock=clock)
    await cache.get(_target(), upstream.load)
    clock.now = 59.0
    await cache.get(_target(), upstream.load)
    clock.now = 61.0
    await cache.get(_target(), upstream.load)
    assert upstream.loads == [("srv-ci", "credential"), ("srv-ci", "credential")]


@pytest.mark.asyncio
async def test_each_credential_gets_its_own_listing() -> None:
    upstream: Final = _Upstream()
    cache: Final = DescriptorCache(clock=_Clock())
    await cache.get(_target(credential="alice"), upstream.load)
    await cache.get(_target(credential="bob"), upstream.load)
    await cache.get(_target(credential="alice"), upstream.load)
    assert upstream.loads == [("srv-ci", "alice"), ("srv-ci", "bob")]


@pytest.mark.asyncio
async def test_the_least_recently_stored_listing_is_evicted_past_the_size_bound() -> None:
    upstream: Final = _Upstream()
    cache: Final = DescriptorCache(clock=_Clock())
    for index in range(1025):
        await cache.get(_target(server_id=f"srv-{index}"), upstream.load)
    await cache.get(_target(server_id="srv-1024"), upstream.load)
    await cache.get(_target(server_id="srv-0"), upstream.load)
    assert upstream.loads[-1] == ("srv-0", "credential") and len(upstream.loads) == 1026


@pytest.mark.asyncio
async def test_a_failed_listing_is_not_cached() -> None:
    calls: Final[list[int]] = []  # mutable-ok: counts attempts

    async def flaky(target: SubscriptionTarget) -> tuple[EventType, ...]:
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("upstream down")
        return (BUILD_FAILED,)

    cache: Final = DescriptorCache(clock=_Clock())
    with pytest.raises(ConnectionError):
        await cache.get(_target(), flaky)
    assert await cache.get(_target(), flaky) == (BUILD_FAILED,)
