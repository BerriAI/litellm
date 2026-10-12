
import subprocess
import sys
from typing import Final

import pytest
from pytest_mock import MockerFixture



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
        ["migrate", "diff", "--from-url", "postgresql://u:p@localhost:9/x",
         "--to-schema-datamodel", "./schema.prisma", "--script"]
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


_PARTITIONED_SPEND_LOGS_PK_DIFF: Final = (
    "-- AlterTable\n"
    'ALTER TABLE "LiteLLM_SpendLogs" DROP CONSTRAINT "LiteLLM_SpendLogs_pkey",\n'
    'ADD CONSTRAINT "LiteLLM_SpendLogs_pkey" PRIMARY KEY ("request_id");\n'
)

_BUDGET_TABLE_DIFF: Final = (
    "-- AlterTable\n"
    'ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN "updated_at" TIMESTAMP(3);\n'
)


def _schema_diff_result(
    mocker: MockerFixture, diff_sql: str, *, partitioned: bool | None
) -> tuple[bool, list[str]]:
    mocker.patch(
        "litellm_proxy_extras.prisma_toolchain.run_prisma",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=diff_sql),
    )
    from litellm.proxy.db.check_migration import check_prisma_schema_diff_helper

    return check_prisma_schema_diff_helper("postgresql://u:p@localhost:9/x", partitioned=partitioned)


def test_partitioned_spend_logs_pk_rewrite_alone_reports_the_schema_in_sync(mocker: MockerFixture) -> None:
    """
    A partitioned LiteLLM_SpendLogs keeps "startTime" in its primary key, so the primary-key
    rewrite Prisma's diff suggests is not executable SQL and is not advice at all.
    """
    assert _schema_diff_result(mocker, _PARTITIONED_SPEND_LOGS_PK_DIFF, partitioned=True) == (False, [])


def test_partitioned_spend_logs_pk_filter_keeps_unrelated_tables_in_the_advice(mocker: MockerFixture) -> None:
    has_diff, sql_commands = _schema_diff_result(
        mocker, _PARTITIONED_SPEND_LOGS_PK_DIFF + _BUDGET_TABLE_DIFF, partitioned=True
    )
    assert has_diff is True
    assert sql_commands == ['ALTER TABLE "LiteLLM_BudgetTable" ADD COLUMN "updated_at" TIMESTAMP(3);']


def test_unpartitioned_spend_logs_still_reports_its_pk_drift(mocker: MockerFixture) -> None:
    has_diff, sql_commands = _schema_diff_result(mocker, _PARTITIONED_SPEND_LOGS_PK_DIFF, partitioned=False)
    assert has_diff is True
    assert len(sql_commands) == 1
    assert 'DROP CONSTRAINT "LiteLLM_SpendLogs_pkey"' in sql_commands[0]
    assert 'PRIMARY KEY ("request_id")' in sql_commands[0]


def test_a_failing_partition_probe_does_not_break_startup(mocker: MockerFixture) -> None:
    """
    The probe runs before uvicorn starts and psycopg's InterfaceError escapes its own
    handler, so boot would die on it. It must degrade to the unfiltered advice instead.
    """
    import psycopg

    from litellm_proxy_extras.utils import ProxyExtrasDBManager

    mocker.patch.object(
        ProxyExtrasDBManager,
        "spend_logs_is_partitioned",
        side_effect=psycopg.InterfaceError("another operation closed the connection"),
    )
    has_diff, sql_commands = _schema_diff_result(mocker, _PARTITIONED_SPEND_LOGS_PK_DIFF, partitioned=None)
    assert has_diff is True
    assert 'PRIMARY KEY ("request_id")' in sql_commands[0]


def test_missing_psycopg_skips_the_partition_check_with_a_note_instead_of_bad_advice_silently(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.proxy.db import check_migration

    monkeypatch.setitem(sys.modules, "psycopg", None)
    mock_logger = mocker.patch.object(check_migration, "verbose_logger", autospec=True)

    has_diff, sql_commands = _schema_diff_result(mocker, _PARTITIONED_SPEND_LOGS_PK_DIFF, partitioned=None)

    assert has_diff is True
    assert 'PRIMARY KEY ("request_id")' in sql_commands[0]
    mock_logger.warning.assert_called_once()
    assert "partitioned" in mock_logger.warning.call_args[0][0]


def test_missing_proxy_extras_skips_the_partition_check_instead_of_crashing_boot(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "litellm_proxy_extras.utils", None)
    has_diff, sql_commands = _schema_diff_result(mocker, _PARTITIONED_SPEND_LOGS_PK_DIFF, partitioned=None)
    assert has_diff is True
    assert 'PRIMARY KEY ("request_id")' in sql_commands[0]
