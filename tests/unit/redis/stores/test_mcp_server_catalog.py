import json
from datetime import timedelta
from typing import Final

import fakeredis
import pytest

from litellm.redis.stores.mcp_server_catalog import TOOLSET_NOT_FOUND, TransportMcpServerCatalogStore
from litellm.redis.transport.contract import RedisFailure
from litellm.types.mcp_server.mcp_toolset import MCPToolset
from tests._support.fake_redis import fake_redis_cache


def _store(server: fakeredis.FakeServer) -> TransportMcpServerCatalogStore:
    return TransportMcpServerCatalogStore(fake_redis_cache(server, namespace="ns").transport())


@pytest.mark.asyncio
async def test_entries_keep_the_legacy_keys_json_bytes_and_whole_second_ttls(server: fakeredis.FakeServer) -> None:
    store: Final = _store(server)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    toolset: Final = MCPToolset(toolset_id="ts-1", toolset_name="ops", tools=[{"server_id": "a", "tool_name": "t"}])

    assert await store.put_submitted_server_ids("u1", ("a", "b"), ttl=timedelta(seconds=60)) is True
    assert await store.put_toolset_permissions(("t2", "t1"), {"a": ("x",)}, ttl=timedelta(seconds=90.7)) is True
    assert await store.put_toolset_by_name("ops", toolset, ttl=timedelta(seconds=30)) is True
    assert await store.put_toolset_by_name("gone", None, ttl=timedelta(seconds=30)) is True

    assert await raw.get("ns:byom_submitted_servers:u1") == b'["a", "b"]'
    assert await raw.ttl("ns:byom_submitted_servers:u1") == 60
    assert await raw.get("ns:toolset_perms:t1,t2") == b'{"a": ["x"]}'
    assert await raw.ttl("ns:toolset_perms:t1,t2") == 90
    assert await raw.get("ns:toolset_name:gone") == b'"__not_found__"'
    assert await store.get_submitted_server_ids("u1") == ("a", "b")
    assert await store.get_toolset_permissions(("t1", "t2")) == {"a": ("x",)}
    assert await store.get_toolset_by_name("ops") == toolset
    assert await store.get_toolset_by_name("gone") == TOOLSET_NOT_FOUND
    assert await store.invalidate_submitted_server_ids("u1") == 1
    assert await store.get_submitted_server_ids("u1") is None


@pytest.mark.asyncio
async def test_old_pod_entries_decode_and_wrong_shapes_are_decode_failures(server: fakeredis.FakeServer) -> None:
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set("ns:byom_submitted_servers:u1", json.dumps(["a"]))
    await raw.set("ns:toolset_perms:t1", "{'a': ['x']}")
    await raw.set("ns:toolset_name:ops", json.dumps({"toolset_id": "ts-1", "toolset_name": "ops"}))
    await raw.set("ns:byom_submitted_servers:bad", b'{"not": "a list"}')
    await raw.set("ns:toolset_name:bad", b"{not json")
    store: Final = _store(server)

    assert await store.get_submitted_server_ids("u1") == ("a",)
    assert await store.get_toolset_permissions(("t1",)) == {"a": ("x",)}
    assert await store.get_toolset_by_name("ops") == MCPToolset(toolset_id="ts-1", toolset_name="ops")
    wrong_shape: Final = await store.get_submitted_server_ids("bad")
    malformed: Final = await store.get_toolset_by_name("bad")
    assert isinstance(wrong_shape, RedisFailure) and wrong_shape.kind == "decode"
    assert isinstance(malformed, RedisFailure) and malformed.kind == "decode"


@pytest.mark.asyncio
async def test_an_outage_is_a_transport_failure(server: fakeredis.FakeServer) -> None:
    store: Final = _store(server)
    server.connected = False
    outage: Final = await store.get_toolset_permissions(("t1",))
    assert isinstance(outage, RedisFailure) and outage.kind == "transport"
