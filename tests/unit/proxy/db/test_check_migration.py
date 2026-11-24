import json
import logging
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
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
    [call] = fake_prisma_cli.calls
    assert call[:5] == ["migrate", "diff", "--from-url", "postgresql://u:p@localhost:9/x", "--to-schema-datamodel"]
    assert call[6:] == ["--script"]
    assert fake_prisma_cli.grandchild_is_gone(within_seconds=5)


def test_migrate_diff_reads_the_proxy_schema_from_any_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The diff named ./schema.prisma, which resolves against the working directory. Once the
    repo-root copy moved under litellm/, every boot outside a directory holding one failed to
    read the schema, so the out-of-sync warning could never fire.
    """
    from litellm_proxy_extras import prisma_toolchain

    from litellm.proxy.db.check_migration import check_prisma_schema_diff_helper

    calls: Final[list[Sequence[str]]] = []  # mutable-ok: records each argv the fake runner gets

    def record_run(argv: Sequence[str], *, timeout: float, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(prisma_toolchain, "run_prisma", record_run)
    unrelated_dir: Final = tmp_path / "unrelated"
    unrelated_dir.mkdir()
    monkeypatch.chdir(unrelated_dir)

    assert check_prisma_schema_diff_helper("postgresql://u:p@localhost:9/x") == (False, [])

    [call] = calls
    schema: Final = Path(call[call.index("--to-schema-datamodel") + 1])
    assert schema.is_absolute()
    assert "model LiteLLM_VerificationToken" in schema.read_text()
    assert call[-1] == "--script"


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
