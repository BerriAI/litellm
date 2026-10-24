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


def test_migrate_diff_announces_the_schema_changes_as_one_json_record(fake_prisma_cli, monkeypatch, capfd):
    """
    The "Changes to DB Schema detected" notice and its SQL went to stdout as plaintext;
    with json_logs on they are one INFO record, so a log pipeline keeps parsing. The
    loggers sit at WARNING, as they do at boot without LITELLM_LOG, and the record
    still goes out: the print it replaces had no level.
    """
    from litellm.proxy.db.check_migration import check_prisma_schema_diff_helper

    monkeypatch.setenv(
        "FAKE_PRISMA_STDOUT", '-- AlterTable\nALTER TABLE "LiteLLM_UserTable" ADD COLUMN "nickname" TEXT;\n'
    )
    monkeypatch.setattr(litellm, "json_logs", True)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # turn_on_json swaps it; put it back at teardown
    turn_on_json()
    verbose_logger.setLevel(logging.WARNING)
    verbose_proxy_logger.setLevel(logging.WARNING)

    assert check_prisma_schema_diff_helper("postgresql://u:p@localhost:9/x") == (
        True,
        ['ALTER TABLE "LiteLLM_UserTable" ADD COLUMN "nickname" TEXT;'],
    )

    out, err = capfd.readouterr()
    assert err == ""
    lines: Final = [raw for raw in out.splitlines() if raw.strip()]
    assert len(lines) == 1, lines
    record: Final = json.loads(lines[0])
    assert record["level"] == "INFO"
    assert record["message"] == (
        'Changes to DB Schema detected\nRequired SQL commands:\nALTER TABLE "LiteLLM_UserTable" ADD COLUMN "nickname" TEXT;'
    )
