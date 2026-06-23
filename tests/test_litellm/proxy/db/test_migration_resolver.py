"""Unit tests for the Prisma migration resolver (v2 default + v1 legacy).

These cover ``ProxyExtrasDBManager.setup_database`` for both resolvers:

* **v2** (the CLI default) runs ``prisma migrate deploy`` and fails fast on
  unrecoverable errors. It never calls ``_resolve_all_migrations`` — the
  diff-and-force recovery that caused schema thrashing during rolling deploys.
* **v1** (legacy, opt back in via ``--use_legacy_migration_resolver``) keeps the
  old diff-and-force recovery and retries rather than raising on most failures.

At the library level the resolver is selected via the ``use_v2_resolver`` kwarg,
which still defaults to False; the proxy CLI passes ``use_v2_resolver=True`` by
default. Migrated here from ``litellm-proxy-extras/tests`` so the resolver is
covered by the standard unit suite (test-unit-proxy-infra) on every PR.
"""

import subprocess
from unittest.mock import patch

import pytest

from litellm_proxy_extras.utils import (
    ProxyExtrasDBManager,
    _max_migration_timestamp,
    _migration_timestamp,
)


def _fake_migrate_deploy_failure(returncode: int, stderr: str):
    def _run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=returncode,
            cmd=args[0],
            stderr=stderr,
            output="",
        )

    return _run


# --------------------------------------------------------------------------- #
# v2 resolver (the default)
# --------------------------------------------------------------------------- #
def test_v2_p3018_permission_error_raises_runtime_error(monkeypatch, tmp_path):
    """v2: a permission failure during migrate deploy raises RuntimeError."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
    monkeypatch.setattr(
        ProxyExtrasDBManager, "_warn_if_db_ahead_of_head", lambda _: None
    )
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")

    stderr = (
        "Error: P3018\nMigration name: 20250326162113_baseline\n"
        "Database error code: 42501\npermission denied for schema public"
    )
    with patch("subprocess.run", side_effect=_fake_migrate_deploy_failure(1, stderr)):
        with pytest.raises(RuntimeError, match="permission"):
            ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)


def test_v2_non_idempotent_p3009_raises_runtime_error(monkeypatch, tmp_path):
    """v2: a non-idempotent migration failure raises (no silent recovery)."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
    monkeypatch.setattr(
        ProxyExtrasDBManager, "_warn_if_db_ahead_of_head", lambda _: None
    )
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")

    stderr = (
        "Error: P3009\nMigration `20260101000000_genuinely_broken` failed\n"
        'Reason: syntax error at or near "BRKN" LINE 42'
    )
    with patch("subprocess.run", side_effect=_fake_migrate_deploy_failure(1, stderr)):
        with pytest.raises(RuntimeError, match="cannot be auto-recovered"):
            ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)


def test_v2_db_push_wraps_subprocess_error_as_runtime_error(monkeypatch, tmp_path):
    """v2: a failing `prisma db push` must raise RuntimeError, not leak
    CalledProcessError past proxy_cli.py's `except RuntimeError`."""
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")

    stderr = "db push error"
    with patch("subprocess.run", side_effect=_fake_migrate_deploy_failure(1, stderr)):
        with pytest.raises(RuntimeError, match="prisma db push failed"):
            ProxyExtrasDBManager.setup_database(use_migrate=False, use_v2_resolver=True)


def test_v2_warn_ahead_of_head_swallows_db_errors(monkeypatch, tmp_path):
    """_warn_if_db_ahead_of_head must never raise — it's informational.

    Non-connection DB errors (e.g. InsufficientPrivilege from a user
    without SELECT on _prisma_migrations) must be caught, not propagated.
    """
    import psycopg

    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, *a, **kw):
            # Simulate an InsufficientPrivilege (subclass of DatabaseError).
            raise psycopg.errors.InsufficientPrivilege("permission denied")

    def _fake_connect(*a, **kw):
        return _FakeConn()

    monkeypatch.setattr("psycopg.connect", _fake_connect)

    # Must not raise.
    ProxyExtrasDBManager._warn_if_db_ahead_of_head(str(tmp_path))


def test_v2_resolve_specific_migration_failure_raises_runtime_error(
    monkeypatch, tmp_path
):
    """If marking a migration as applied fails inside P3009 idempotent
    recovery, the subprocess error must be re-raised as RuntimeError so
    proxy_cli.py catches it cleanly (instead of leaking CalledProcessError)."""
    monkeypatch.setattr(
        ProxyExtrasDBManager, "_warn_if_db_ahead_of_head", lambda _: None
    )
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")
    monkeypatch.setattr(
        ProxyExtrasDBManager, "_roll_back_migration", lambda *a, **kw: None
    )

    # First call: migrate deploy -> P3009 idempotent error.
    # Recovery path tries _resolve_specific_migration; that also raises.
    def _failing_resolve(*a, **kw):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd="prisma migrate resolve --applied",
            stderr="resolve failed",
            output="",
        )

    monkeypatch.setattr(
        ProxyExtrasDBManager, "_resolve_specific_migration", _failing_resolve
    )

    stderr = (
        "Error: P3009\nMigration `20260101000000_some_migration` failed\n"
        "relation already exists"
    )
    with patch("subprocess.run", side_effect=_fake_migrate_deploy_failure(1, stderr)):
        with pytest.raises(
            RuntimeError, match="Failed to mark migration .* as applied"
        ):
            ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)


def test_v2_does_not_call_resolve_all_migrations(monkeypatch, tmp_path):
    """v2 must never call _resolve_all_migrations — that's the bug it fixes."""
    monkeypatch.setattr(
        ProxyExtrasDBManager, "_warn_if_db_ahead_of_head", lambda _: None
    )
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")

    class FakeResult:
        stdout = "Applied migration.\n"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeResult())

    resolve_called = {"n": 0}
    monkeypatch.setattr(
        ProxyExtrasDBManager,
        "_resolve_all_migrations",
        lambda *a, **kw: resolve_called.__setitem__("n", resolve_called["n"] + 1),
    )

    ok = ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)
    assert ok is True
    assert resolve_called["n"] == 0, "v2 must not invoke the diff-and-force recovery"


# --------------------------------------------------------------------------- #
# v1 resolver (legacy, opt-in via --use_legacy_migration_resolver)
# --------------------------------------------------------------------------- #
def test_v1_default_still_calls_resolve_all_migrations(monkeypatch, tmp_path):
    """v1 (legacy) continues to call _resolve_all_migrations on the happy path.

    This is the diff-and-force recovery v2 avoids — pinning it here so the
    legacy path's behavior is locked and a future inadvertent change is caught.
    """
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")

    # Stub `prisma migrate deploy` to claim success with pending migrations
    # applied, which is the code path that triggers the legacy post-migration
    # sanity check (a call to _resolve_all_migrations).
    class FakeResult:
        stdout = "Applied migration.\n"
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        return FakeResult()

    resolve_called = {"n": 0}

    def fake_resolve(*args, **kwargs):
        resolve_called["n"] += 1

    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(ProxyExtrasDBManager, "_resolve_all_migrations", fake_resolve)

    ok = ProxyExtrasDBManager.setup_database(use_migrate=True)  # v2 flag NOT set
    assert ok is True
    assert resolve_called["n"] == 1, "v1 default should still invoke the legacy path"


def test_v1_no_pending_migrations_returns_true_without_resolve_all(
    monkeypatch, tmp_path
):
    """v1: when migrate deploy reports no pending migrations, return True and
    skip the post-migration sanity check (no _resolve_all_migrations call)."""
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")

    class FakeResult:
        stdout = "No pending migrations to apply\n"
        stderr = ""

    monkeypatch.setattr("subprocess.run", lambda *a, **kw: FakeResult())

    resolve_called = {"n": 0}
    monkeypatch.setattr(
        ProxyExtrasDBManager,
        "_resolve_all_migrations",
        lambda *a, **kw: resolve_called.__setitem__("n", resolve_called["n"] + 1),
    )

    ok = ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=False)
    assert ok is True
    assert resolve_called["n"] == 0, "no pending migrations → skip the sanity check"


def test_v1_db_push_returns_true(monkeypatch, tmp_path):
    """v1: `use_migrate=False` runs `prisma db push` and returns True.

    Contrast with v2, which wraps a db-push failure as RuntimeError.
    """
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")

    pushes = {"n": 0}

    def fake_run(cmd, *args, **kwargs):
        if "push" in cmd:
            pushes["n"] += 1

        class R:
            stdout = ""
            stderr = ""

        return R()

    monkeypatch.setattr("subprocess.run", fake_run)

    ok = ProxyExtrasDBManager.setup_database(use_migrate=False, use_v2_resolver=False)
    assert ok is True
    assert pushes["n"] == 1, "v1 db push path should invoke `prisma db push` once"


def test_v1_permission_error_p3018_raises_runtime_error(monkeypatch, tmp_path):
    """v1 also fails fast on permission errors: a P3018 permission failure
    raises RuntimeError rather than silently marking the migration applied."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")

    stderr = (
        "Error: P3018\nMigration name: 20250326162113_baseline\n"
        "Database error code: 42501\npermission denied for schema public"
    )
    with patch("subprocess.run", side_effect=_fake_migrate_deploy_failure(1, stderr)):
        with pytest.raises(RuntimeError, match="permission"):
            ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=False)


def test_v1_non_idempotent_p3009_returns_false_without_raising(monkeypatch, tmp_path):
    """v1: a persistent non-idempotent migration failure is retried and
    ultimately returns False — it does NOT raise. (The v2 resolver raises
    'cannot be auto-recovered' for the same input.)"""
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")
    # Don't actually sleep between the 4 retry attempts.
    monkeypatch.setattr("time.sleep", lambda *a, **kw: None)

    stderr = (
        "Error: P3009\n"
        "The `20260101000000_genuinely_broken` migration failed to apply.\n"
        'Reason: syntax error at or near "BRKN" LINE 42'
    )
    with patch("subprocess.run", side_effect=_fake_migrate_deploy_failure(1, stderr)):
        ok = ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=False)
    assert ok is False, "v1 retries then returns False instead of raising"


# --------------------------------------------------------------------------- #
# resolver-independent helpers
# --------------------------------------------------------------------------- #
def test_strip_prisma_query_params_removes_connection_limit():
    """DATABASE_URLs with Prisma-specific params should be parseable by psycopg."""
    url = "postgresql://u:p@h:5432/db?connection_limit=100&pool_timeout=60&sslmode=require"
    stripped = ProxyExtrasDBManager._strip_prisma_query_params(url)
    assert "connection_limit" not in stripped
    assert "pool_timeout" not in stripped
    assert "sslmode=require" in stripped


def test_strip_prisma_query_params_passthrough_no_query():
    """URLs without query strings are returned unchanged."""
    url = "postgresql://u:p@h:5432/db"
    assert ProxyExtrasDBManager._strip_prisma_query_params(url) == url


def test_migration_timestamp_extracts_leading_digits():
    assert _migration_timestamp("20260101000000_add_foo") == 20260101000000
    assert _migration_timestamp("20250326162113_baseline") == 20250326162113


def test_migration_timestamp_returns_zero_on_malformed():
    assert _migration_timestamp("0_init") == 0
    assert _migration_timestamp("not_a_migration") == 0


def test_max_migration_timestamp():
    names = {"20250326000000_a", "20260415000000_b", "20251115000000_c"}
    assert _max_migration_timestamp(names) == 20260415000000


def test_max_migration_timestamp_empty_set():
    assert _max_migration_timestamp(set()) == 0
