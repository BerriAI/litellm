
import logging
import subprocess
from collections.abc import Callable

import pytest



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


def test_migrate_diff_without_the_prisma_runner_skips_instead_of_crashing_boot(monkeypatch, caplog, capsys):
    """
    Boot calls this helper directly, so an ImportError here takes the proxy down before
    uvicorn starts. An install without the runner must lose the diagnostic, not the proxy.
    """
    import sys

    from litellm.proxy.db.check_migration import check_prisma_schema_diff_helper

    monkeypatch.setitem(sys.modules, "litellm_proxy_extras.prisma_toolchain", None)

    caplog.set_level(logging.ERROR, logger="LiteLLM")
    assert check_prisma_schema_diff_helper("postgresql://u:p@localhost:9/x") == (False, [])
    assert capsys.readouterr() == ("", "")

    caplog.set_level(logging.WARNING, logger="LiteLLM")
    check_prisma_schema_diff_helper("postgresql://u:p@localhost:9/x")
    assert "Skipping the migration diff" in _only_record(caplog, logging.WARNING).getMessage()


_DB_URL = "postgresql://u:p@localhost:9/x"
_DRIFT_SQL = 'ALTER TABLE "LiteLLM_UserTable" ADD COLUMN "nickname" TEXT;'
_DRIFT_DIFF_OUTPUT = f"-- AlterTable\n{_DRIFT_SQL}\n"


def _use_prisma_runner(monkeypatch: pytest.MonkeyPatch, run_prisma: Callable[..., object]) -> None:
    import litellm_proxy_extras.prisma_toolchain as prisma_toolchain

    monkeypatch.setattr(prisma_toolchain, "run_prisma", run_prisma)


def _only_record(caplog: pytest.LogCaptureFixture, level: int) -> logging.LogRecord:
    [record] = [r for r in caplog.records if r.name == "LiteLLM" and r.levelno == level]
    return record


def test_schema_drift_report_is_one_warning_record_not_stdout(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    from litellm.proxy.db.check_migration import check_prisma_schema_diff_helper

    def diff_with_drift(argv: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout=_DRIFT_DIFF_OUTPUT, stderr="")

    _use_prisma_runner(monkeypatch, diff_with_drift)

    caplog.set_level(logging.ERROR, logger="LiteLLM")
    assert check_prisma_schema_diff_helper(_DB_URL) == (True, [_DRIFT_SQL])
    assert capsys.readouterr() == ("", "")

    caplog.set_level(logging.WARNING, logger="LiteLLM")
    check_prisma_schema_diff_helper(_DB_URL)
    message = _only_record(caplog, logging.WARNING).getMessage()
    assert "Changes to DB Schema detected" in message
    assert "Required SQL commands:" in message
    assert _DRIFT_SQL in message


@pytest.mark.parametrize(
    ("failure", "expected_fragment"),
    [
        pytest.param(subprocess.TimeoutExpired(["prisma"], 1.0), "Timed out after", id="timeout"),
        pytest.param(
            subprocess.CalledProcessError(1, ["prisma"], stderr="P1001: Can't reach database server"),
            "Failed to generate migration diff. Error: P1001: Can't reach database server",
            id="prisma-error",
        ),
    ],
)
def test_migrate_diff_failures_are_one_error_record_not_stdout(
    failure: Exception,
    expected_fragment: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from litellm.proxy.db.check_migration import check_prisma_schema_diff_helper

    def raise_failure(*_: object, **__: object) -> subprocess.CompletedProcess[str]:
        raise failure

    _use_prisma_runner(monkeypatch, raise_failure)

    caplog.set_level(logging.CRITICAL, logger="LiteLLM")
    assert check_prisma_schema_diff_helper(_DB_URL) == (False, [])
    assert capsys.readouterr() == ("", "")

    caplog.set_level(logging.ERROR, logger="LiteLLM")
    check_prisma_schema_diff_helper(_DB_URL)
    assert expected_fragment in _only_record(caplog, logging.ERROR).getMessage()
