import json
import logging
import sys
from typing import Final

import pytest

import litellm
from litellm._logging import turn_on_json, verbose_logger, verbose_proxy_logger


def test_check_migration_out_of_sync(mocker):
    """
    Test that the check_prisma_schema_diff function
    - 🚨 [IMPORTANT] Does NOT Raise an Exception when the Prisma schema is out of sync with the database.
    - logs an error when the Prisma schema is out of sync with the database.
    """
    # Import the module first so check_migration is in sys.modules,
    # then patch the logger reference in that module directly (not the source
    # module) so the patch works regardless of import order or xdist worker
    # assignment.
    from litellm.proxy.db import check_migration

    # Mock the helper function to simulate out-of-sync state
    mock_logger = mocker.patch.object(
        check_migration,
        "verbose_logger",
        autospec=True,
    )
    mocker.patch.object(
        check_migration,
        "check_prisma_schema_diff_helper",
        return_value=(True, ["ALTER TABLE users ADD COLUMN new_field TEXT;"]),
    )

    # Run the function - it should not raise an error
    try:
        check_migration.check_prisma_schema_diff(db_url="mock_url")
    except Exception as e:
        pytest.fail(f"check_prisma_schema_diff raised an unexpected exception: {e}")

    # Verify the logger was called with the expected message
    check_migration.verbose_logger.exception.assert_called_once()
    actual_message = check_migration.verbose_logger.exception.call_args[0][0]
    assert "prisma schema out of sync with db" in actual_message


@pytest.mark.timeout(30)
def test_migrate_diff_stops_at_its_budget_and_takes_its_process_tree_with_it(fake_prisma_cli, monkeypatch):
    """
    `prisma migrate diff` ran unbounded, so a database that never answers hung boot
    before uvicorn ever started, and interrupting the proxy orphaned the schema engine.
    """
    from litellm.proxy.db.check_migration import check_prisma_schema_diff_helper

    monkeypatch.setenv("FAKE_PRISMA_HANG_FIRST", "1")

    assert check_prisma_schema_diff_helper("postgresql://u:p@localhost:9/x") == (False, [])
    assert fake_prisma_cli.calls == [
        [
            "migrate",
            "diff",
            "--from-url",
            "postgresql://u:p@localhost:9/x",
            "--to-schema-datamodel",
            "./schema.prisma",
            "--script",
        ]
    ]
    assert fake_prisma_cli.grandchild_is_gone(within_seconds=5)


def test_migrate_diff_without_the_prisma_runner_skips_instead_of_crashing_boot(monkeypatch):
    """
    Boot calls this helper directly, so an ImportError here takes the proxy down before
    uvicorn starts. An install without the runner must lose the diagnostic, not the proxy.
    """
    import sys

    from litellm.proxy.db.check_migration import check_prisma_schema_diff_helper

    monkeypatch.setitem(sys.modules, "litellm_proxy_extras.prisma_toolchain", None)

    assert check_prisma_schema_diff_helper("postgresql://u:p@localhost:9/x") == (False, [])


def test_a_skipped_migration_diff_is_one_json_record_in_json_mode(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """
    With json_logs on, the notice that the migration diff was skipped is one WARNING record on stdout,
    also while the loggers sit at WARNING, as they do at boot without LITELLM_LOG.
    """
    from litellm.proxy.db.check_migration import check_prisma_schema_diff_helper

    monkeypatch.setitem(sys.modules, "litellm_proxy_extras.prisma_toolchain", None)
    monkeypatch.setattr(litellm, "json_logs", True)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    turn_on_json()
    verbose_logger.setLevel(logging.WARNING)
    verbose_proxy_logger.setLevel(logging.WARNING)

    assert check_prisma_schema_diff_helper("postgresql://u:p@localhost:9/x") == (False, [])

    out, err = capfd.readouterr()
    assert err == ""
    lines: Final = [raw for raw in out.splitlines() if raw.strip()]
    assert len(lines) == 1, lines
    record: Final = json.loads(lines[0])
    assert (record["level"], record["message"]) == (
        "WARNING",
        "Skipping the migration diff: litellm-proxy-extras has no Prisma runner. "
        "Error: import of litellm_proxy_extras.prisma_toolchain halted; None in sys.modules",
    )


_BOOT_DRIFT_SQL: Final = """-- AlterTable
ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;

-- AlterTable
ALTER TABLE "LiteLLM_SpendLogs" DROP CONSTRAINT "LiteLLM_SpendLogs_pkey",
ADD CONSTRAINT "LiteLLM_SpendLogs_pkey" PRIMARY KEY ("request_id");

-- DropTable
DROP TABLE "LiteLLM_SpendLogs_legacy";

-- CreateIndex
CREATE INDEX "LiteLLM_SpendLogs_litellm_call_id_idx" ON "LiteLLM_SpendLogs"("litellm_call_id");
"""


_DB_URL: Final = "postgresql://u:p@db/litellm"


def _drift_commands_with(monkeypatch: pytest.MonkeyPatch, *, partitioned: bool, diff_output: str) -> tuple[str, ...]:
    from litellm_proxy_extras.utils import (
        ProxyExtrasDBManager,  # pyright: ignore[reportUnknownVariableType]  # litellm-proxy-extras is an optional install
    )

    from litellm.proxy.db.check_migration import _drift_commands

    def probe(database_url: str | None = None) -> bool:
        assert database_url == _DB_URL
        return partitioned

    monkeypatch.setattr(ProxyExtrasDBManager, "spend_logs_is_partitioned", staticmethod(probe))  # pyright: ignore[reportUnknownArgumentType]  # litellm-proxy-extras is an optional install
    return _drift_commands(diff_output, _DB_URL)


def test_boot_drift_check_ignores_the_partitioned_spend_logs_primary_key_and_runbook_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: Final = _drift_commands_with(monkeypatch, partitioned=True, diff_output=_BOOT_DRIFT_SQL)

    assert commands == ('ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;',)


def test_boot_drift_check_still_reports_the_primary_key_of_an_unpartitioned_spend_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: Final = _drift_commands_with(monkeypatch, partitioned=False, diff_output=_BOOT_DRIFT_SQL)

    assert commands == (
        'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN     "updated_by" TEXT;',
        'ALTER TABLE "LiteLLM_SpendLogs" DROP CONSTRAINT "LiteLLM_SpendLogs_pkey", '
        'ADD CONSTRAINT "LiteLLM_SpendLogs_pkey" PRIMARY KEY ("request_id");',
        'DROP TABLE "LiteLLM_SpendLogs_legacy";',
    )


def test_boot_drift_check_reports_nothing_for_an_empty_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    empty: Final = "-- This is an empty migration.\n"
    assert _drift_commands_with(monkeypatch, partitioned=True, diff_output=empty) == ()
