"""The opt-in REPLICA IDENTITY FULL step, without a database.

The behavior against real Postgres is covered by
tests/proxy_migration_tests/test_replica_identity_full.py; these pin the two
things that hold with no database at all: the statement handed to the Prisma
CLI, and the promise that no failure of this optional step escapes into a
migration run that already succeeded.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from litellm_proxy_extras.replica_identity import (
    REPLICA_IDENTITY_FULL_ENV_VAR,
    apply_replica_identity_full,
)
from litellm_proxy_extras.utils import ProxyExtrasDBManager


def test_hands_the_alter_statement_to_the_prisma_cli():
    captured = {}

    def capture(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["sql"] = Path(cmd[cmd.index("--file") + 1]).read_text()
        return subprocess.CompletedProcess(cmd, 0)

    with patch(
        "litellm_proxy_extras.replica_identity.subprocess.run", side_effect=capture
    ):
        applied = apply_replica_identity_full(
            schema_path="/somewhere/schema.prisma",
            prisma_command="prisma",
            prisma_env={"DATABASE_URL": "postgresql://x/y"},
        )

    assert applied is True
    assert captured["cmd"][:3] == ["prisma", "db", "execute"]
    assert captured["cmd"][-2:] == ["--schema", "/somewhere/schema.prisma"]

    sql = captured["sql"]
    assert "ALTER TABLE %s REPLICA IDENTITY FULL" in sql
    assert r"c.relname LIKE 'LiteLLM\_%'" in sql
    assert "c.relreplident <> 'f'" in sql
    assert "lock_timeout" in sql


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "prisma", stderr="must be owner of table"),
        subprocess.TimeoutExpired("prisma", 60),
        OSError(2, "No such file or directory"),
        PermissionError(13, "Read-only file system"),
    ],
    ids=["rejected", "timed-out", "cli-missing", "read-only-fs"],
)
def test_every_failure_is_reported_instead_of_raised(failure):
    with patch(
        "litellm_proxy_extras.replica_identity.subprocess.run", side_effect=failure
    ):
        assert (
            apply_replica_identity_full(
                schema_path="/somewhere/schema.prisma",
                prisma_command="prisma",
                prisma_env={},
            )
            is False
        )


def test_an_unusable_migrations_dir_skips_the_step_instead_of_killing_the_run(
    tmp_path, monkeypatch
):
    """LITELLM_MIGRATION_DIR makes the step copy the migrations tree before it
    can run, and that copy is filesystem work that can fail on its own."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv(REPLICA_IDENTITY_FULL_ENV_VAR, "true")
    monkeypatch.setenv("LITELLM_MIGRATION_DIR", str(blocker / "migrations"))

    assert ProxyExtrasDBManager.apply_replica_identity_full_if_requested() is False
