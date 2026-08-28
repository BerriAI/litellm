"""Regression tests for ProxyExtrasDBManager's v2 migration resolver.

v2 is the proxy CLI default; v1 stays reachable via the `use_v2_resolver`
kwarg, which still defaults to False for direct callers.
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


def test_v1_default_still_calls_resolve_all_migrations(monkeypatch, tmp_path):
    """v1 (default) continues to call _resolve_all_migrations on the happy path.

    This is the existing buggy behavior — we're not fixing it in v1, only
    offering v2 as opt-in. This test pins the default so that a future
    inadvertent default flip is caught.
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

    connects = {"n": 0}

    def _fake_connect(*a, **kw):
        connects["n"] += 1
        return _FakeConn()

    monkeypatch.setattr("psycopg.connect", _fake_connect)

    assert ProxyExtrasDBManager._warn_if_db_ahead_of_head(str(tmp_path)) is None
    assert connects["n"] == 1, "the failing query must actually have been reached"


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
            RuntimeError, match=r"Failed to mark migration .* as applied"
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


_DEADLOCK_STDERR = (
    "Error: ERROR: deadlock detected\n"
    "DETAIL: Process 277 waits for ExclusiveLock on advisory lock "
    "[17556,0,72707369,1]; blocked by process 278.\n"
    "Process 278 waits for ShareLock on virtual transaction 3/1041; "
    "blocked by process 277."
)


class _DeployApplied:
    stdout = "All migrations have been successfully applied."
    stderr = ""
    returncode = 0


def _deploy_only(deploy_side_effect):
    """subprocess.run stand-in that only intercepts `prisma migrate deploy`.

    Scoped by argv so the Prisma toolchain check cannot consume the mock first.
    """
    deploys = {"n": 0}

    def _run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if list(cmd)[-2:] == ["migrate", "deploy"]:
            deploys["n"] += 1
            return deploy_side_effect(deploys["n"], cmd)
        return _DeployApplied()

    return _run, deploys


def _prepare_v2_resolver(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:9/x")
    monkeypatch.setattr(
        ProxyExtrasDBManager, "_warn_if_db_ahead_of_head", lambda _: None
    )
    monkeypatch.setattr(ProxyExtrasDBManager, "_get_prisma_dir", lambda: str(tmp_path))
    (tmp_path / "schema.prisma").write_text("// stub")
    monkeypatch.setattr("time.sleep", lambda *_a, **_k: None)


def test_v2_retries_transient_advisory_lock_deadlock(monkeypatch, tmp_path):
    """v2: replicas racing `migrate deploy` deadlock on Prisma's advisory
    lock, which is transient and must be retried rather than kill the boot."""
    _prepare_v2_resolver(monkeypatch, tmp_path)

    def _side_effect(n, cmd):
        if n == 1:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, stderr=_DEADLOCK_STDERR, output=""
            )
        return _DeployApplied()

    run, deploys = _deploy_only(_side_effect)
    with patch("subprocess.run", side_effect=run):
        ok = ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)

    assert ok is True
    assert deploys["n"] == 2, "the deadlocked deploy must be retried, not raised"


def test_v2_persistent_advisory_lock_deadlock_eventually_raises(monkeypatch, tmp_path):
    """v2: the deadlock retry is bounded, so a deadlock that never clears
    still raises instead of looping or reporting success."""
    _prepare_v2_resolver(monkeypatch, tmp_path)

    def _side_effect(n, cmd):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, stderr=_DEADLOCK_STDERR, output=""
        )

    run, deploys = _deploy_only(_side_effect)
    with patch("subprocess.run", side_effect=run):
        with pytest.raises(RuntimeError, match="after 4 attempts"):
            ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)

    assert deploys["n"] == 4


@pytest.mark.parametrize(
    "stderr",
    [
        "Error: P1001: Can't reach database server at `db`:`5432`",
        "Error: P1002: The database server was reached but timed out.",
    ],
)
def test_v2_retries_transient_database_connectivity_errors(monkeypatch, tmp_path, stderr):
    """v2: a database not accepting connections yet is retried, not fatal."""
    _prepare_v2_resolver(monkeypatch, tmp_path)

    def _side_effect(n, cmd):
        if n == 1:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, stderr=stderr, output=""
            )
        return _DeployApplied()

    run, deploys = _deploy_only(_side_effect)
    with patch("subprocess.run", side_effect=run):
        ok = ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)

    assert ok is True
    assert deploys["n"] == 2, "an unreachable database must be retried, not raised"


def test_v2_unreachable_database_still_fails_after_the_retries(monkeypatch, tmp_path):
    """v2: a genuinely unreachable database still raises once the attempts
    are spent, rather than passing as a successful migration."""
    _prepare_v2_resolver(monkeypatch, tmp_path)

    def _side_effect(n, cmd):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            stderr="Error: P1001: Can't reach database server at `db`:`5432`",
            output="",
        )

    run, deploys = _deploy_only(_side_effect)
    with patch("subprocess.run", side_effect=run):
        with pytest.raises(RuntimeError, match="after 4 attempts"):
            ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)

    assert deploys["n"] == 4


def test_v2_exhausted_retries_report_the_prisma_error(monkeypatch, tmp_path, caplog):
    """v2: retrying must not swallow Prisma's stderr, which is captured and is
    the only place the cause appears for an operator or a boot-log grep."""
    _prepare_v2_resolver(monkeypatch, tmp_path)
    stderr = "Error: P1001: Can't reach database server at `wrong`:`5432`"

    def _side_effect(n, cmd):
        raise subprocess.CalledProcessError(
            returncode=1, cmd=cmd, stderr=stderr, output=""
        )

    run, _ = _deploy_only(_side_effect)
    with caplog.at_level("INFO", logger="litellm_proxy_extras"):
        with patch("subprocess.run", side_effect=run):
            with pytest.raises(RuntimeError) as exc_info:
                ProxyExtrasDBManager.setup_database(
                    use_migrate=True, use_v2_resolver=True
                )

    assert "P1001" in str(exc_info.value)
    assert "P1001" in caplog.text


def test_v2_db_push_retries_transient_failures(monkeypatch, tmp_path):
    """v2: `prisma db push` retries a transient failure like v1 did, so the
    default flip does not cost --use_prisma_db_push its retries."""
    _prepare_v2_resolver(monkeypatch, tmp_path)

    pushes = {"n": 0}

    def _run(*args, **kwargs):
        cmd = list(args[0] if args else kwargs.get("args", []))
        if cmd[-3:] != ["db", "push", "--accept-data-loss"]:
            return _DeployApplied()
        pushes["n"] += 1
        if pushes["n"] == 1:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=cmd,
                stderr="Error: P1001: Can't reach database server at `db`:`5432`",
                output="",
            )
        return _DeployApplied()

    monkeypatch.setattr(
        ProxyExtrasDBManager, "spend_logs_is_partitioned", lambda: False
    )
    with patch("subprocess.run", side_effect=_run):
        ok = ProxyExtrasDBManager.setup_database(use_migrate=False, use_v2_resolver=True)

    assert ok is True
    assert pushes["n"] == 2


def test_v2_unclassified_failure_is_not_treated_as_transient(monkeypatch, tmp_path):
    """v2: an unrecognised deploy failure still raises on the first attempt."""
    _prepare_v2_resolver(monkeypatch, tmp_path)

    def _side_effect(n, cmd):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            stderr="Error: relation \"LiteLLM_SpendLogs\" does not exist",
            output="",
        )

    run, deploys = _deploy_only(_side_effect)
    with patch("subprocess.run", side_effect=run):
        with pytest.raises(RuntimeError, match="cannot be auto-recovered"):
            ProxyExtrasDBManager.setup_database(use_migrate=True, use_v2_resolver=True)

    assert deploys["n"] == 1


