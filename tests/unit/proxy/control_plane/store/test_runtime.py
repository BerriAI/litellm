import asyncio
from typing import Final

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm.proxy.control_plane.auth.rbac import Principal, Scope
from litellm.proxy.control_plane.store.models import Change, Entry
from litellm.proxy.control_plane.store.runtime import Written
from tests.unit.proxy.control_plane.store.support import MemoryStore, MemoryStream, container, role_entry
from tests.unit.proxy.control_plane.support import MASTER_KEY


@pytest.mark.asyncio
async def test_background_consumer_loads_snapshot_and_observes_writes_and_stops_when_blocked() -> None:
    stored: Final = role_entry("9-0", "existing", "fleet:read")
    store: Final = MemoryStore((stored,))
    stream: Final = MemoryStream()
    rig: Final = container(store, stream)
    runtime: Final = rig.runtime_config
    await runtime.start()
    assert runtime.cursor == "9-0"
    assert rig.policy.scopes_for("existing") == frozenset({Scope.FLEET_READ})
    actor: Final = rig.token_verifier.verify(MASTER_KEY)
    assert isinstance(actor, Principal)
    result: Final = await runtime.write(role_entry("0-0", "created", "usage:read").change, actor)
    assert isinstance(result, Written)
    await stream.consumed(result.entry.revision)
    assert rig.policy.scopes_for("created") == frozenset({Scope.USAGE_READ})
    assert store.rows[-1].revision == stream.entries[-1].revision == result.entry.revision
    assert store.rows[-1].change.actor == actor.subject
    await runtime.close()
    assert runtime.task is not None and runtime.task.done()
    assert stream.closed and store.closed


@pytest.mark.asyncio
async def test_uncommitted_or_failed_events_do_not_grant_permissions_and_resync_restores_database() -> None:
    store: Final = MemoryStore((role_entry("1-0", "saved", "fleet:read"),))
    rig: Final = container(store, MemoryStream())
    runtime: Final = rig.runtime_config
    await runtime.reload()
    await runtime.process(role_entry("2-0", "uncommitted", "roles:write"))
    assert "uncommitted" not in rig.policy.names
    runtime.state.apply(role_entry("3-0", "stale", "usage:read"))
    await runtime.process(Entry(revision="4-0", change=Change(op="resync", actor="operator")))
    assert rig.policy.scopes_for("saved") == frozenset({Scope.FLEET_READ})
    assert "stale" not in rig.policy.names


@pytest.mark.asyncio
async def test_redis_error_backs_off_and_reloads_before_consuming_again() -> None:
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def pause() -> None:
        entered.set()
        await release.wait()

    store: Final = MemoryStore()
    stream: Final = MemoryStream()
    stream.failure = RedisConnectionError("disconnected")
    rig: Final = container(store, stream)
    runtime: Final = rig.runtime_config
    runtime.pause = pause
    await runtime.start()
    await asyncio.wait_for(entered.wait(), 30)
    assert not runtime.ready
    store.rows = (role_entry("12-0", "recovered", "usage:read"),)
    release.set()
    await stream.consumed("12-0")
    assert rig.policy.scopes_for("recovered") == frozenset({Scope.USAGE_READ})
    assert runtime.ready
    await runtime.close()


@pytest.mark.asyncio
async def test_failed_database_write_appends_resync_and_returns_unavailable() -> None:
    store: Final = MemoryStore()
    store.fail_write = True
    stream: Final = MemoryStream()
    rig: Final = container(store, stream)
    runtime: Final = rig.runtime_config
    runtime.ready = True
    actor: Final = rig.token_verifier.verify(MASTER_KEY)
    assert isinstance(actor, Principal)
    result: Final = await runtime.write(role_entry("0-0", "failed", "roles:write").change, actor)
    assert not isinstance(result, Written)
    assert result.status == 503
    assert tuple(entry.change.op for entry in stream.entries) == ("upsert", "resync")
    assert store.rows == ()
    assert "failed" not in rig.policy.names


@pytest.mark.asyncio
async def test_behind_schema_keeps_reads_config_only_and_does_not_start_consumer(caplog) -> None:
    store: Final = MemoryStore((role_entry("1-0", "unmigrated", "roles:write"),), migrated=False)
    stream: Final = MemoryStream()
    rig: Final = container(store, stream)
    await rig.runtime_config.start()
    assert not rig.runtime_config.ready
    assert rig.runtime_config.task is None
    assert "unmigrated" not in rig.policy.names
    assert store.loads == 0
    assert "litellm --control-plane --skip_server_startup" in caplog.text
    await rig.runtime_config.close()
