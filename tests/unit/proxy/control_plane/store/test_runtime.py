import asyncio
from typing import Final

import pytest
from pydantic import SecretStr
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm.proxy.control_plane.auth.rbac import Principal, Scope, build_principal
from litellm.proxy.control_plane.settings import DataplaneSpec
from litellm.proxy.control_plane.store.crypto import SecretCodec
from litellm.proxy.control_plane.store.models import Change, Entry
from litellm.proxy.control_plane.store.runtime import Written
from tests.unit.proxy.control_plane.store.support import MemoryStore, MemoryStream, container, role_entry
from tests.unit.proxy.control_plane.support import MASTER_KEY


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_kind", ["undecryptable", "missing-secret", "unknown-scope", "unknown-parent"])
async def test_quarantined_rows_revoke_stale_grants_without_blocking_other_writes_or_their_own_deletion(
    bad_kind: str,
) -> None:
    encrypted: Final = SecretCodec(SecretStr("wrong-key")).encode(
        DataplaneSpec.model_validate({"url": "https://runtime.test", "service_token": {"token": "private"}})
    )
    bad: Final = Entry(
        revision="2-0",
        change=Change(
            op="upsert",
            kind="dataplane" if bad_kind in ("undecryptable", "missing-secret") else "role",
            name="bad",
            actor="operator",
            entity={
                "undecryptable": encrypted,
                "missing-secret": {"url": "https://runtime.test", "service_token": {}},
                "unknown-scope": {"scopes": ["invalid:scope"]},
                "unknown-parent": {"scopes": ["roles:write"], "inherits": ["absent"]},
            }[bad_kind],
        ),
    )
    dependent: Final = Entry(
        revision="3-0",
        change=Change(op="upsert", kind="role", name="dependent", entity={"inherits": ["bad"]}, actor="operator"),
    )
    store: Final = MemoryStore((bad, dependent, role_entry("4-0", "safe", "usage:read")))
    stream: Final = MemoryStream()
    rig: Final = container(store, stream)
    runtime: Final = rig.runtime_config
    assert runtime.state.apply(role_entry("1-0", "bad", "roles:write"))
    assert await runtime.reload()
    runtime.ready = True
    assert "bad" not in rig.policy.names and "dependent" not in rig.policy.names
    assert rig.registry.get("bad") is None
    assert rig.policy.scopes_for("safe") == frozenset({Scope.USAGE_READ})
    assert runtime.state.view.quarantined == {(bad.change.kind, "bad"), ("role", "dependent")}
    actor: Final = rig.token_verifier.verify(MASTER_KEY)
    assert isinstance(actor, Principal)
    created: Final = await runtime.write(role_entry("0-0", "created", "usage:read").change, actor)
    assert isinstance(created, Written)
    deleted: Final = await runtime.write(Change(op="delete", kind=bad.change.kind, name="bad", actor="operator"), actor)
    assert isinstance(deleted, Written)
    assert tuple(row.change.name for row in store.rows) == ("dependent", "safe", "created")
    assert tuple(entry.change.op for entry in stream.entries) == ("upsert", "delete")


@pytest.mark.asyncio
async def test_committed_writes_update_local_live_views_without_a_consumer() -> None:
    store: Final = MemoryStore()
    rig: Final = container(store, MemoryStream())
    runtime: Final = rig.runtime_config
    runtime.ready = True
    actor: Final = rig.token_verifier.verify(MASTER_KEY)
    assert isinstance(actor, Principal)
    created: Final = await runtime.write(role_entry("0-0", "local", "proxy:read").change, actor)
    assert isinstance(created, Written)
    assert rig.policy.scopes_for("local") == frozenset({Scope.PROXY_READ})
    revoked: Final = await runtime.write(Change(op="delete", kind="role", name="local", actor="operator"), actor)
    assert isinstance(revoked, Written)
    assert "local" not in rig.policy.names
    spec: Final = DataplaneSpec.model_validate({"url": "https://runtime.test", "service_token": {"token": "private"}})
    codec: Final = runtime.state.codec
    assert codec is not None
    written: Final = await runtime.write(
        Change(op="upsert", kind="dataplane", name="local", entity=codec.encode(spec), actor="operator"), actor
    )
    assert isinstance(written, Written)
    assert rig.registry.get("local") is not None
    removed: Final = await runtime.write(Change(op="delete", kind="dataplane", name="local", actor="operator"), actor)
    assert isinstance(removed, Written)
    assert rig.registry.get("local") is None and store.rows == ()
    assert runtime.task is None


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
async def test_behind_schema_keeps_reads_config_only_without_loading_unmigrated_tables(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store: Final = MemoryStore((role_entry("1-0", "unmigrated", "roles:write"),), migrated=False)
    stream: Final = MemoryStream()
    rig: Final = container(store, stream)
    await rig.runtime_config.start()
    assert not rig.runtime_config.ready
    assert rig.runtime_config.task is not None
    assert "unmigrated" not in rig.policy.names
    assert store.loads == 0
    assert "litellm --control-plane --skip_server_startup" in caplog.text
    await rig.runtime_config.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_current", [False, True])
async def test_startup_database_failure_recovers_without_restarting(fail_current: bool) -> None:
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()

    async def pause() -> None:
        entered.set()
        await release.wait()

    store: Final = MemoryStore((role_entry("12-0", "recovered", "usage:read"),))
    store.fail_current = fail_current
    store.fail_load = not fail_current
    stream: Final = MemoryStream()
    rig: Final = container(store, stream)
    runtime: Final = rig.runtime_config
    runtime.pause = pause
    try:
        await runtime.start()
        await asyncio.wait_for(entered.wait(), 1)
        assert not runtime.ready and "recovered" not in rig.policy.names
        store.fail_current = store.fail_load = False
        release.set()
        await stream.consumed("12-0")
        assert runtime.ready and rig.policy.scopes_for("recovered") == frozenset({Scope.USAGE_READ})
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_empty_stream_poll_reconciles_a_committed_role_revocation_without_an_event() -> None:
    store: Final = MemoryStore((role_entry("12-0", "revoked", "proxy:read"),))
    stream: Final = MemoryStream()
    rig: Final = container(store, stream)
    runtime: Final = rig.runtime_config
    try:
        await runtime.start()
        await stream.consumed("12-0")
        assert rig.policy.scopes_for("revoked") == frozenset({Scope.PROXY_READ})
        store.rows = ()
        stream.changed.set()
        await stream.consumed("12-0")
        assert runtime.ready and "revoked" not in rig.policy.names
        assert rig.policy.scopes_for("revoked") == frozenset()
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remaining", "change"),
    [
        ((), role_entry("0-0", "delegated", "roles:write", "usage:read").change),
        (("roles:write",), role_entry("0-0", "delegated", "roles:write", "usage:read").change),
        ((), Change(op="delete", kind="role", name="other", actor="operator")),
        (
            ("roles:write",),
            Change(
                op="upsert",
                kind="dataplane",
                name="new",
                entity={"url": "https://runtime.test", "service_token": {"token": "unused"}},
                actor="operator",
            ),
        ),
    ],
)
async def test_committed_revocation_rejects_a_stale_replica_writer_before_publication(
    remaining: tuple[str, ...], change: Change
) -> None:
    store: Final = MemoryStore(
        (role_entry("1-0", "delegated", "roles:write", "dataplanes:write", "usage:read"), role_entry("2-0", "other"))
    )
    stream: Final = MemoryStream()
    first: Final = container(store, stream).runtime_config
    stale: Final = container(store, stream).runtime_config
    await first.reload()
    await stale.reload()
    first.ready = stale.ready = True
    actor: Final = build_principal(
        stale.state.view.policy, "operator", None, frozenset({"delegated"}), None, frozenset(), "oidc"
    )
    assert actor.has_scope(Scope.ROLES_WRITE) and actor.has_scope(Scope.DATAPLANES_WRITE)
    master: Final = container().token_verifier.verify(MASTER_KEY)
    assert isinstance(master, Principal)
    revoked: Final = await first.write(role_entry("0-0", "delegated", *remaining).change, master)
    assert isinstance(revoked, Written)
    result: Final = await stale.write(change, actor)
    assert not isinstance(result, Written)
    assert result.status == 403
    assert stream.entries == store.writes == (revoked.entry,)
    assert store.rows == (role_entry("2-0", "other"), revoked.entry)


@pytest.mark.asyncio
async def test_locked_authorization_keeps_claim_narrowing_and_master_key_without_an_admin_role() -> None:
    store: Final = MemoryStore((role_entry("1-0", "delegated", "roles:write", "usage:read"),))
    stream: Final = MemoryStream()
    rig: Final = container(store, stream, include_default_roles=False)
    runtime: Final = rig.runtime_config
    await runtime.reload()
    runtime.ready = True
    actor: Final = build_principal(
        rig.policy,
        "operator",
        None,
        frozenset({"delegated"}),
        frozenset({Scope.ROLES_WRITE}),
        frozenset(),
        "oidc",
    )
    desired: Final = role_entry("0-0", "new", "usage:read").change
    denied: Final = await runtime.write(desired, actor)
    assert not isinstance(denied, Written) and denied.status == 403
    assert stream.entries == store.writes == ()
    master: Final = rig.token_verifier.verify(MASTER_KEY)
    assert isinstance(master, Principal) and "admin" not in rig.policy.names
    allowed: Final = await runtime.write(desired, master)
    assert isinstance(allowed, Written)
    assert stream.entries == store.writes == (allowed.entry,)
