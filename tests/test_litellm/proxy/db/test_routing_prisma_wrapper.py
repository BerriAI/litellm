import asyncio
import logging
import os
import sys
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))


# NOTE: do NOT patch sys.modules["prisma"] file-wide via an autouse fixture.
# Doing so leaks across pytest-xdist test scheduling: when a worker runs a
# routing test, then later runs test_exception_handler.py, the cached MagicMock
# attribute references break `isinstance(e, prisma.errors.X)` in
# `is_database_transport_error`. The two tests below that actually need to
# stub the prisma SDK do so per-test via monkeypatch, which is properly scoped.


def _make_wrappers():
    from litellm.proxy.db.prisma_client import PrismaWrapper

    writer_inner = MagicMock(name="writer_prisma")
    reader_inner = MagicMock(name="reader_prisma")
    writer = PrismaWrapper(original_prisma=writer_inner, iam_token_db_auth=False)
    reader = PrismaWrapper(original_prisma=reader_inner, iam_token_db_auth=False)
    return writer, writer_inner, reader, reader_inner


class _FakeActions:
    """Stand-in for a Prisma per-model Actions instance (non-callable, has find_many/create)."""

    def __init__(self, name: str):
        self._name = name
        for method in (
            "find_many",
            "find_unique",
            "find_first",
            "count",
            "group_by",
            "create",
            "update",
            "upsert",
            "delete",
            "delete_many",
            "update_many",
        ):
            setattr(self, method, MagicMock(name=f"{name}.{method}"))


def _model_actions_mock(name: str) -> _FakeActions:
    return _FakeActions(name)


def test_top_level_query_raw_routes_to_reader():
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, writer_inner, reader, reader_inner = _make_wrappers()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    # query_raw should resolve to the reader's underlying client.
    assert routing.query_raw is reader_inner.query_raw
    assert routing.query_first is reader_inner.query_first


def test_top_level_execute_raw_routes_to_writer():
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, writer_inner, reader, reader_inner = _make_wrappers()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    # execute_raw, batch_, tx are write-side and must hit the writer.
    assert routing.execute_raw is writer_inner.execute_raw
    assert routing.batch_ is writer_inner.batch_
    assert routing.tx is writer_inner.tx


def test_per_model_reads_route_to_reader_writes_to_writer():
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, writer_inner, reader, reader_inner = _make_wrappers()
    writer_inner.litellm_usertable = _model_actions_mock("writer_users")
    reader_inner.litellm_usertable = _model_actions_mock("reader_users")
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    actions = routing.litellm_usertable

    # Reads → reader actions.
    assert actions.find_many is reader_inner.litellm_usertable.find_many
    assert actions.find_unique is reader_inner.litellm_usertable.find_unique
    assert actions.find_first is reader_inner.litellm_usertable.find_first
    assert actions.count is reader_inner.litellm_usertable.count
    assert actions.group_by is reader_inner.litellm_usertable.group_by

    # Writes → writer actions.
    assert actions.create is writer_inner.litellm_usertable.create
    assert actions.update is writer_inner.litellm_usertable.update
    assert actions.upsert is writer_inner.litellm_usertable.upsert
    assert actions.delete is writer_inner.litellm_usertable.delete
    assert actions.update_many is writer_inner.litellm_usertable.update_many
    assert actions.delete_many is writer_inner.litellm_usertable.delete_many


@pytest.mark.asyncio
async def test_connect_invokes_both_clients():
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, writer_inner, reader, reader_inner = _make_wrappers()
    writer_inner.connect = AsyncMock()
    reader_inner.connect = AsyncMock()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    await routing.connect()

    writer_inner.connect.assert_awaited_once()
    reader_inner.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_connect_logs_writer_and_reader_success(caplog):
    """Successful startup emits a positive INFO confirmation for both writer
    and reader so operators can verify connectivity without inspecting the URL
    in logs."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, writer_inner, reader, reader_inner = _make_wrappers()
    writer_inner.connect = AsyncMock()
    reader_inner.connect = AsyncMock()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
        await routing.connect()

    messages = [r.getMessage() for r in caplog.records]
    assert "[writer] DB connected" in messages
    assert "[reader] DB connected" in messages


@pytest.mark.asyncio
async def test_disconnect_continues_when_one_side_fails():
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, writer_inner, reader, reader_inner = _make_wrappers()
    writer_inner.disconnect = AsyncMock(side_effect=RuntimeError("writer down"))
    reader_inner.disconnect = AsyncMock()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    with pytest.raises(RuntimeError, match="writer down"):
        await routing.disconnect()

    # Reader still attempted even though writer raised.
    reader_inner.disconnect.assert_awaited_once()


def test_is_connected_reflects_writer_only():
    """is_connected() must NOT depend on reader health — a healthy writer with
    a degraded reader should report True so that PrismaClient.connect()'s
    health check does not re-trigger a writer reconnect (which only fixes
    writer-side problems and would loop indefinitely)."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, writer_inner, reader, reader_inner = _make_wrappers()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    writer_inner.is_connected = MagicMock(return_value=True)
    reader_inner.is_connected = MagicMock(return_value=True)
    assert routing.is_connected() is True

    # Reader down → still True (reader degradation is tracked separately).
    reader_inner.is_connected = MagicMock(return_value=False)
    assert routing.is_connected() is True

    # Writer down → False.
    writer_inner.is_connected = MagicMock(return_value=False)
    assert routing.is_connected() is False


def test_token_refresh_delegates_to_both_writer_and_reader():
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer = MagicMock()
    writer.start_token_refresh_task = AsyncMock()
    writer.stop_token_refresh_task = AsyncMock()
    reader = MagicMock()
    reader.start_token_refresh_task = AsyncMock()
    reader.stop_token_refresh_task = AsyncMock()

    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    asyncio.run(routing.start_token_refresh_task())
    asyncio.run(routing.stop_token_refresh_task())

    # Both wrappers get start/stop — each manages its own IAM token. When
    # IAM is disabled on a wrapper its task body is a no-op.
    writer.start_token_refresh_task.assert_awaited_once()
    writer.stop_token_refresh_task.assert_awaited_once()
    reader.start_token_refresh_task.assert_awaited_once()
    reader.stop_token_refresh_task.assert_awaited_once()


def test_routed_actions_falls_back_to_writer_for_unknown_methods():
    from litellm.proxy.db.routing_prisma_wrapper import _RoutedActions

    writer_actions = _model_actions_mock("writer")
    writer_actions.some_custom_method = "writer-custom"
    reader_actions = _model_actions_mock("reader")
    reader_actions.some_custom_method = "reader-custom"

    routed = _RoutedActions(writer_actions, reader_actions, lambda: True)
    # Unknown method → defaults to writer (safe fallback for write-like ops).
    assert routed.some_custom_method == "writer-custom"


def test_routed_actions_respects_should_use_reader_flag():
    """When the routing wrapper marks the reader unavailable, _RoutedActions
    must redirect reads to the writer instead — without needing to re-fetch
    the actions accessor."""
    from litellm.proxy.db.routing_prisma_wrapper import _RoutedActions

    writer_actions = _model_actions_mock("writer")
    reader_actions = _model_actions_mock("reader")

    use_reader = {"value": True}
    routed = _RoutedActions(writer_actions, reader_actions, lambda: use_reader["value"])

    # Reader healthy → reads to reader.
    assert routed.find_many is reader_actions.find_many

    # Reader degrades mid-flight → next read goes to writer.
    use_reader["value"] = False
    assert routed.find_many is writer_actions.find_many


# ---------------------------------------------------------------------------
# Reader graceful degradation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_swallows_reader_failure_and_falls_back_to_writer():
    """A reader connect failure must NOT abort proxy startup. The wrapper
    flips into degraded mode so subsequent reads route to the writer."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, writer_inner, reader, reader_inner = _make_wrappers()
    writer_inner.connect = AsyncMock()
    reader_inner.connect = AsyncMock(side_effect=RuntimeError("reader unreachable"))
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    # Must not raise — reader failure is non-fatal.
    await routing.connect()

    assert routing.reader_unavailable is True
    writer_inner.connect.assert_awaited_once()
    reader_inner.connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_reads_route_to_writer_when_reader_unavailable():
    """Top-level read methods and per-model reads must fall through to the
    writer while the reader is degraded."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, writer_inner, reader, reader_inner = _make_wrappers()
    writer_inner.litellm_usertable = _model_actions_mock("writer_users")
    reader_inner.litellm_usertable = _model_actions_mock("reader_users")
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    routing._reader_unavailable = True

    # Top-level reads → writer.
    assert routing.query_raw is writer_inner.query_raw
    assert routing.query_first is writer_inner.query_first

    # Per-model reads → writer actions.
    actions = routing.litellm_usertable
    assert actions.find_many is writer_inner.litellm_usertable.find_many
    assert actions.find_unique is writer_inner.litellm_usertable.find_unique


@pytest.mark.asyncio
async def test_recreate_prisma_client_recreates_both_writer_and_reader():
    """Writer reconnect path calls recreate_prisma_client. The routing wrapper
    must recreate BOTH clients so a DB-wide event doesn't leave a stale reader."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer = MagicMock()
    writer.recreate_prisma_client = AsyncMock()
    reader = MagicMock()
    reader.iam_token_db_auth = False
    reader.recreate_prisma_client = AsyncMock()

    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    with patch.dict(os.environ, {"DATABASE_URL_READ_REPLICA": "reader-url"}):
        await routing.recreate_prisma_client("writer-url", http_client=None)

    writer.recreate_prisma_client.assert_awaited_once_with(
        "writer-url", http_client=None
    )
    reader.recreate_prisma_client.assert_awaited_once_with(
        "reader-url", http_client=None
    )
    assert routing.reader_unavailable is False


@pytest.mark.asyncio
async def test_recreate_recovers_reader_after_prior_degradation():
    """If a previous connect/recreate degraded the reader, a successful
    recreate must clear the flag so reads start hitting the reader again."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer = MagicMock()
    writer.recreate_prisma_client = AsyncMock()
    reader = MagicMock()
    reader.iam_token_db_auth = False
    reader.recreate_prisma_client = AsyncMock()

    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    routing._reader_unavailable = True

    with patch.dict(os.environ, {"DATABASE_URL_READ_REPLICA": "reader-url"}):
        await routing.recreate_prisma_client("writer-url")

    assert routing.reader_unavailable is False


@pytest.mark.asyncio
async def test_recreate_degrades_reader_if_reader_recreate_fails():
    """If the reader recreate fails, writer recreate still succeeds and the
    routing wrapper degrades (does not raise)."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer = MagicMock()
    writer.recreate_prisma_client = AsyncMock()
    reader = MagicMock()
    reader.iam_token_db_auth = False
    reader.recreate_prisma_client = AsyncMock(
        side_effect=RuntimeError("reader still down")
    )

    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    with patch.dict(os.environ, {"DATABASE_URL_READ_REPLICA": "reader-url"}):
        # Must not raise — writer was recreated, reader is best-effort.
        await routing.recreate_prisma_client("writer-url")

    writer.recreate_prisma_client.assert_awaited_once()
    assert routing.reader_unavailable is True


@pytest.mark.asyncio
async def test_recreate_degrades_reader_when_replica_url_missing():
    """Non-IAM reader needs DATABASE_URL_READ_REPLICA. If it's missing
    (configuration drift), the wrapper degrades instead of raising."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer = MagicMock()
    writer.recreate_prisma_client = AsyncMock()
    reader = MagicMock()
    reader.iam_token_db_auth = False
    reader.recreate_prisma_client = AsyncMock()

    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    # Ensure env var is absent.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DATABASE_URL_READ_REPLICA", None)
        await routing.recreate_prisma_client("writer-url")

    writer.recreate_prisma_client.assert_awaited_once()
    reader.recreate_prisma_client.assert_not_awaited()
    assert routing.reader_unavailable is True


@pytest.mark.asyncio
async def test_recreate_iam_reader_refreshes_token():
    """IAM-enabled readers must refresh their token (reader has its own parsed
    endpoint) and pass the fresh URL to recreate_prisma_client."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer = MagicMock()
    writer.recreate_prisma_client = AsyncMock()
    reader = MagicMock()
    reader.iam_token_db_auth = True
    reader.get_rds_iam_token = MagicMock(return_value="postgresql://u:fresh@h:5432/db")
    reader.recreate_prisma_client = AsyncMock()

    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    await routing.recreate_prisma_client("writer-url")

    reader.get_rds_iam_token.assert_called_once()
    reader.recreate_prisma_client.assert_awaited_once_with(
        "postgresql://u:fresh@h:5432/db", http_client=None
    )
    assert routing.reader_unavailable is False


@pytest.mark.asyncio
async def test_recreate_degrades_when_iam_token_generation_returns_none():
    """If `get_rds_iam_token` returns None (e.g. AWS-side failure), the wrapper
    must degrade rather than crash — this exercises the explicit `raise
    RuntimeError` inside `_recreate_reader`'s IAM branch."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer = MagicMock()
    writer.recreate_prisma_client = AsyncMock()
    reader = MagicMock()
    reader.iam_token_db_auth = True
    reader.get_rds_iam_token = MagicMock(return_value=None)
    reader.recreate_prisma_client = AsyncMock()

    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    await routing.recreate_prisma_client("writer-url")

    writer.recreate_prisma_client.assert_awaited_once()
    reader.recreate_prisma_client.assert_not_awaited()
    assert routing.reader_unavailable is True


def test_writer_and_reader_properties_expose_underlying_wrappers():
    """The `writer` and `reader` properties are used by PrismaClient.writer_db
    to smoke-test the writer specifically during reconnect — they must return
    the exact wrappers passed in."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    writer, _, reader, _ = _make_wrappers()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    assert routing.writer is writer
    assert routing.reader is reader


def test_per_model_accessor_falls_back_when_reader_lacks_attr():
    """If the reader Prisma client somehow lacks a model accessor that the
    writer has (older client / partial mock), the wrapper must fall back to
    the writer accessor instead of raising AttributeError to the caller."""
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    # Plain class with only the accessor set on the writer side. Using a real
    # class instead of MagicMock so attribute access raises AttributeError
    # naturally instead of auto-creating mock attributes.
    class _PartialPrisma:
        pass

    writer_inner = _PartialPrisma()
    writer_inner.litellm_usertable = _model_actions_mock("writer_users")
    reader_inner = _PartialPrisma()  # deliberately missing litellm_usertable

    writer = PrismaWrapper(original_prisma=writer_inner, iam_token_db_auth=False)
    reader = PrismaWrapper(original_prisma=reader_inner, iam_token_db_auth=False)
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    actions = routing.litellm_usertable
    # Falls back to the writer's accessor verbatim — not a _RoutedActions wrapper.
    assert actions is writer_inner.litellm_usertable


@pytest.mark.asyncio
async def test_writer_recreate_passes_http_client_through(monkeypatch):
    """When PrismaClient is constructed with an http_client, recreate must
    forward it to the new Prisma() so connection settings persist across
    reconnects."""
    from litellm.proxy.db.prisma_client import PrismaWrapper

    captured_kwargs: Dict[str, Any] = {}

    class FakePrisma:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        async def connect(self):
            return None

    fake_module = MagicMock()
    fake_module.Prisma = FakePrisma
    monkeypatch.setitem(sys.modules, "prisma", fake_module)

    writer = PrismaWrapper(original_prisma=MagicMock(), iam_token_db_auth=False)
    sentinel_http = object()
    await writer.recreate_prisma_client(
        "postgresql://u:p@h:5432/db", http_client=sentinel_http
    )

    assert captured_kwargs == {"http": sentinel_http}


# ---------------------------------------------------------------------------
# IAM endpoint parsing + reader IAM refresh
# ---------------------------------------------------------------------------


def test_parse_iam_endpoint_from_url_extracts_all_fields():
    from litellm.proxy.db.prisma_client import parse_iam_endpoint_from_url

    ep = parse_iam_endpoint_from_url(
        "postgresql://litellm_user:initial-token@aurora-reader.example.com:6543/litellm?schema=public"
    )
    assert ep.host == "aurora-reader.example.com"
    assert ep.port == "6543"
    assert ep.user == "litellm_user"
    assert ep.name == "litellm"
    assert ep.schema == "public"


def test_parse_iam_endpoint_defaults_port_to_5432_and_skips_schema():
    from litellm.proxy.db.prisma_client import parse_iam_endpoint_from_url

    ep = parse_iam_endpoint_from_url("postgresql://u@host/dbname")
    assert ep.host == "host"
    assert ep.port == "5432"
    assert ep.user == "u"
    assert ep.name == "dbname"
    assert ep.schema is None


def test_parse_iam_endpoint_rejects_url_without_user_or_dbname():
    from litellm.proxy.db.prisma_client import parse_iam_endpoint_from_url

    with pytest.raises(ValueError, match="missing host or username"):
        parse_iam_endpoint_from_url("postgresql://host:5432/db")
    with pytest.raises(ValueError, match="missing database name"):
        parse_iam_endpoint_from_url("postgresql://u@host:5432/")


def test_iam_endpoint_build_url_inserts_token_verbatim():
    from litellm.proxy.db.prisma_client import IAMEndpoint

    # `generate_iam_auth_token` already URL-encodes the presigned token, so
    # `build_url` must NOT encode again — double-encoding turned `%3D` into
    # `%253D` and broke RDS auth on the reader path.
    ep = IAMEndpoint(host="h", port="5432", user="u", name="db", schema="public")
    pre_encoded_token = "token%2Fwith%3Fweird%26chars%3Dyes"
    url = ep.build_url(pre_encoded_token)
    assert url == f"postgresql://u:{pre_encoded_token}@h:5432/db?schema=public"
    # Sanity check: no `%25` (the encoding of `%`), confirming we didn't re-encode.
    assert "%25" not in url


@pytest.mark.asyncio
async def test_iam_refresh_logs_carry_log_prefix(caplog):
    """When `log_prefix` is set on a PrismaWrapper, every IAM-related log
    line emitted by that wrapper must start with the prefix so writer and
    reader can be told apart in interleaved output."""
    from litellm.proxy.db.prisma_client import PrismaWrapper

    wrapper = PrismaWrapper(
        original_prisma=MagicMock(),
        iam_token_db_auth=True,
        log_prefix="[reader]",
    )

    with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
        await wrapper.start_token_refresh_task()
        # Loop emits "RDS IAM token refresh loop started..." on first tick.
        # Cancel immediately so the loop body runs once and we can assert.
        await wrapper.stop_token_refresh_task()

    messages = [r.getMessage() for r in caplog.records]
    # Both start and stop notifications carry the prefix.
    assert any(
        m.startswith("[reader] Started RDS IAM token proactive refresh")
        for m in messages
    )
    assert any(
        m.startswith("[reader] Stopped RDS IAM token refresh background task")
        for m in messages
    )


def test_get_rds_iam_token_returns_none_when_iam_disabled():
    """`get_rds_iam_token` short-circuits to None when iam_token_db_auth is
    False — covers the early-return guard at the top of the method."""
    from litellm.proxy.db.prisma_client import PrismaWrapper

    wrapper = PrismaWrapper(original_prisma=MagicMock(), iam_token_db_auth=False)
    assert wrapper.get_rds_iam_token() is None


@pytest.mark.asyncio
async def test_getattr_does_not_block_inside_running_loop_on_expired_token(monkeypatch):
    """When `__getattr__` runs inside a running event loop and the IAM token
    is expired, it MUST schedule the refresh as a background task and return
    immediately. The previous `run_coroutine_threadsafe` + `future.result()`
    pattern deadlocks the loop (loop thread blocks waiting for a coroutine
    that needs the loop to run) and times out at 30s — exactly what was
    breaking the reader on first query."""
    from litellm.proxy.db.prisma_client import PrismaWrapper

    # Stale URL — `is_token_expired` returns True because the password isn't
    # a parseable IAM token, so we exercise the expired branch.
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA",
        "postgresql://reader:placeholder@reader.aurora.local:5432/litellm",
    )

    inner = MagicMock()
    inner.query_raw = MagicMock(name="query_raw_attr")

    wrapper = PrismaWrapper(
        original_prisma=inner,
        iam_token_db_auth=True,
        db_url_env_var="DATABASE_URL_READ_REPLICA",
    )

    # Replace the heavy refresh coroutine with a no-op AsyncMock so we can
    # observe whether it was scheduled without actually doing the recreate.
    refresh_calls = {"count": 0}

    async def fake_refresh():
        refresh_calls["count"] += 1

    monkeypatch.setattr(wrapper, "_safe_refresh_token", fake_refresh)

    # Direct attribute access from inside this async test runs __getattr__
    # on the loop thread, exercising the in-loop branch. If the previous
    # `run_coroutine_threadsafe` + `future.result()` pattern were back, this
    # line would deadlock the loop and the test would hang (and pytest's
    # per-test timeout would catch it).
    attr = wrapper.query_raw
    # Yield once so the scheduled refresh task gets a chance to run.
    await asyncio.sleep(0)

    assert attr is inner.query_raw
    assert refresh_calls["count"] == 1


def test_writer_get_rds_iam_token_defaults_port_when_unset(monkeypatch):
    """When DATABASE_PORT is unset, the writer must default to the Postgres
    standard port instead of passing `None` through. Passing None to
    `generate_iam_auth_token` makes botocore embed the literal string
    \"None\" in the presigned URL during signing and crashes with
    `ValueError: Port could not be cast to integer value as 'None'`."""
    from litellm.proxy.db.prisma_client import PrismaWrapper

    monkeypatch.setenv("DATABASE_HOST", "writer.aurora.local")
    monkeypatch.delenv("DATABASE_PORT", raising=False)
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    monkeypatch.delenv("DATABASE_SCHEMA", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    captured: Dict[str, Any] = {}

    def fake_generate(db_host=None, db_port=None, db_user=None):
        captured["port"] = db_port
        return "TOKEN"

    fake_module = MagicMock()
    fake_module.generate_iam_auth_token = fake_generate
    monkeypatch.setitem(sys.modules, "litellm.proxy.auth.rds_iam_token", fake_module)

    writer = PrismaWrapper(
        original_prisma=MagicMock(),
        iam_token_db_auth=True,
    )
    new_url = writer.get_rds_iam_token()

    assert captured["port"] == "5432"  # default applied, NOT None
    assert ":5432/litellm" in (new_url or "")


def test_writer_get_rds_iam_token_uses_database_host_env_vars(monkeypatch):
    """Writer's IAM path (no iam_endpoint configured) reads host/port/user/db
    from the legacy DATABASE_HOST/PORT/USER/NAME env vars and writes the URL
    back to DATABASE_URL — this is the pre-read-replica behavior the patch
    must preserve."""
    from litellm.proxy.db.prisma_client import PrismaWrapper

    monkeypatch.setenv("DATABASE_HOST", "writer.aurora.local")
    monkeypatch.setenv("DATABASE_PORT", "5432")
    monkeypatch.setenv("DATABASE_USER", "litellm")
    monkeypatch.setenv("DATABASE_NAME", "litellm")
    monkeypatch.setenv("DATABASE_SCHEMA", "public")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    captured: Dict[str, Any] = {}

    def fake_generate(db_host=None, db_port=None, db_user=None):
        captured["host"] = db_host
        captured["port"] = db_port
        captured["user"] = db_user
        return "WRITER-TOKEN"

    fake_module = MagicMock()
    fake_module.generate_iam_auth_token = fake_generate
    monkeypatch.setitem(sys.modules, "litellm.proxy.auth.rds_iam_token", fake_module)

    writer = PrismaWrapper(
        original_prisma=MagicMock(),
        iam_token_db_auth=True,
        # No iam_endpoint → legacy DATABASE_HOST/etc. path.
    )
    new_url = writer.get_rds_iam_token()

    assert captured == {
        "host": "writer.aurora.local",
        "port": "5432",
        "user": "litellm",
    }
    assert new_url == (
        "postgresql://litellm:WRITER-TOKEN@writer.aurora.local:5432/litellm?schema=public"
    )
    # Writer updates its own env var (DATABASE_URL by default), not the reader's.
    assert os.environ["DATABASE_URL"] == new_url


def test_reader_iam_refresh_uses_parsed_endpoint(monkeypatch):
    """The reader generates fresh tokens against its parsed endpoint and
    writes the new URL to DATABASE_URL_READ_REPLICA — not DATABASE_URL."""
    from litellm.proxy.db.prisma_client import IAMEndpoint, PrismaWrapper

    # Pre-seed env vars so we can prove the reader does NOT touch DATABASE_URL.
    monkeypatch.setenv("DATABASE_URL", "writer-url-untouched")
    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "stale-reader-url")

    captured: Dict[str, Any] = {}

    def fake_generate(db_host=None, db_port=None, db_user=None):
        captured["host"] = db_host
        captured["port"] = db_port
        captured["user"] = db_user
        return "FRESH-TOKEN"

    fake_module = MagicMock()
    fake_module.generate_iam_auth_token = fake_generate
    monkeypatch.setitem(sys.modules, "litellm.proxy.auth.rds_iam_token", fake_module)

    endpoint = IAMEndpoint(
        host="reader.aurora.local",
        port="5432",
        user="lit",
        name="litellm",
        schema=None,
    )
    reader = PrismaWrapper(
        original_prisma=MagicMock(),
        iam_token_db_auth=True,
        db_url_env_var="DATABASE_URL_READ_REPLICA",
        iam_endpoint=endpoint,
        recreate_uses_datasource=True,
    )

    new_url = reader.get_rds_iam_token()

    # IAM token generator was called with the reader's parsed endpoint, not
    # the writer's DATABASE_HOST/PORT/USER env vars.
    assert captured == {
        "host": "reader.aurora.local",
        "port": "5432",
        "user": "lit",
    }
    assert new_url is not None
    assert new_url.startswith(
        "postgresql://lit:FRESH-TOKEN@reader.aurora.local:5432/litellm"
    )
    # The reader updates its OWN env var; writer's DATABASE_URL is left alone.
    assert os.environ["DATABASE_URL_READ_REPLICA"] == new_url
    assert os.environ["DATABASE_URL"] == "writer-url-untouched"


@pytest.mark.asyncio
async def test_reader_recreate_uses_datasource_override(monkeypatch):
    """Reader recreate must pass `datasource={"url": ...}` to Prisma() — Prisma
    only auto-reads DATABASE_URL, so without the override the new reader URL
    would be silently ignored."""
    from litellm.proxy.db.prisma_client import IAMEndpoint, PrismaWrapper

    captured_kwargs: Dict[str, Any] = {}

    class FakePrisma:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        async def connect(self):
            return None

    fake_module = MagicMock()
    fake_module.Prisma = FakePrisma
    monkeypatch.setitem(sys.modules, "prisma", fake_module)

    reader = PrismaWrapper(
        original_prisma=MagicMock(),
        iam_token_db_auth=True,
        db_url_env_var="DATABASE_URL_READ_REPLICA",
        iam_endpoint=IAMEndpoint(host="h", port="5432", user="u", name="db"),
        recreate_uses_datasource=True,
    )

    await reader.recreate_prisma_client(
        "postgresql://u:newtoken@h:5432/db", http_client=None
    )

    assert captured_kwargs == {
        "datasource": {"url": "postgresql://u:newtoken@h:5432/db"}
    }


@pytest.mark.asyncio
async def test_writer_recreate_does_not_use_datasource(monkeypatch):
    """Writer keeps relying on Prisma reading DATABASE_URL from env — datasource
    override must NOT leak into the writer path (would override the freshly
    rotated env var)."""
    from litellm.proxy.db.prisma_client import PrismaWrapper

    captured_kwargs: Dict[str, Any] = {}

    class FakePrisma:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        async def connect(self):
            return None

    fake_module = MagicMock()
    fake_module.Prisma = FakePrisma
    monkeypatch.setitem(sys.modules, "prisma", fake_module)

    writer = PrismaWrapper(
        original_prisma=MagicMock(),
        iam_token_db_auth=True,
    )

    await writer.recreate_prisma_client(
        "postgresql://u:newtoken@h:5432/db", http_client=None
    )

    assert "datasource" not in captured_kwargs


def test_prisma_client_init_falls_back_to_writer_when_reader_iam_token_fails(
    monkeypatch, caplog
):
    """A transient AWS STS error (or any other failure) during the reader
    IAM token mint must NOT abort proxy startup. The reader is opt-in, so
    `PrismaClient.__init__` should log a warning and fall back to the
    writer-only `PrismaWrapper`. The runtime contract in
    `RoutingPrismaWrapper.connect` already says reader-side failures are
    non-fatal — but that code never runs if construction throws first."""
    from litellm.proxy.db.prisma_client import PrismaWrapper
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    monkeypatch.setenv("IAM_TOKEN_DB_AUTH", "true")
    monkeypatch.setenv(
        "DATABASE_URL_READ_REPLICA",
        "postgresql://reader_user@reader.aurora.local:5432/litellm",
    )

    class FakePrisma:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def connect(self):
            return None

    fake_prisma_module = MagicMock()
    fake_prisma_module.Prisma = FakePrisma
    monkeypatch.setitem(sys.modules, "prisma", fake_prisma_module)

    fake_iam_module = MagicMock()

    def boom(**_kwargs):
        raise RuntimeError("simulated AWS STS hiccup")

    fake_iam_module.generate_iam_auth_token = boom
    monkeypatch.setitem(
        sys.modules, "litellm.proxy.auth.rds_iam_token", fake_iam_module
    )

    from litellm.proxy.utils import PrismaClient

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        client = PrismaClient(
            database_url="postgresql://writer@writer.aurora.local:5432/litellm",
            proxy_logging_obj=MagicMock(),
        )

    # Construction did not raise, and the proxy is in writer-only mode —
    # NOT a RoutingPrismaWrapper, so reads will go to the writer.
    assert isinstance(client.db, PrismaWrapper)
    assert not isinstance(client.db, RoutingPrismaWrapper)
    # And the operator gets a clear warning.
    assert any(
        "Failed to initialize read replica Prisma client" in r.getMessage()
        for r in caplog.records
    )
