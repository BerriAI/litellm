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


PKEY_REWRITE = (
    'ALTER TABLE "LiteLLM_SpendLogs" DROP CONSTRAINT "LiteLLM_SpendLogs_pkey", '
    'ADD CONSTRAINT "LiteLLM_SpendLogs_pkey" PRIMARY KEY ("request_id");'
)


def test_spend_logs_pkey_rewrite_filtered_when_partitioning_on():
    """Partitioned deployments (PK must include "startTime") must not be told
    to rewrite the primary key to request_id alone - the rewrite is invalid
    for the partitioned layout and schema.prisma's single-column @id always
    produces it."""
    from litellm.proxy.db.check_migration import filter_spend_logs_pkey_rewrite

    assert filter_spend_logs_pkey_rewrite([PKEY_REWRITE], use_spend_logs_partitioning=True) == []


def test_spend_logs_pkey_rewrite_kept_when_partitioning_off():
    """Without partitioning the rewrite is valid advice and must stay."""
    from litellm.proxy.db.check_migration import filter_spend_logs_pkey_rewrite

    assert filter_spend_logs_pkey_rewrite([PKEY_REWRITE], use_spend_logs_partitioning=False) == [PKEY_REWRITE]


def test_spend_logs_pkey_rewrite_filtered_in_two_statement_form():
    """The diff can also arrive as separate DROP / ADD statements."""
    from litellm.proxy.db.check_migration import filter_spend_logs_pkey_rewrite

    commands = [
        'ALTER TABLE "LiteLLM_SpendLogs" DROP CONSTRAINT "LiteLLM_SpendLogs_pkey";',
        'ALTER TABLE "LiteLLM_SpendLogs" ADD CONSTRAINT "LiteLLM_SpendLogs_pkey" PRIMARY KEY ("request_id");',
    ]

    assert filter_spend_logs_pkey_rewrite(commands, use_spend_logs_partitioning=True) == []


def test_real_drift_and_other_spend_logs_tables_survive_the_filter():
    """Unrelated drift, the legacy table, and partition child tables are not
    touched by the filter - only the exact table's pkey rewrite is."""
    from litellm.proxy.db.check_migration import filter_spend_logs_pkey_rewrite

    drift = 'ALTER TABLE "LiteLLM_VerificationToken" ADD COLUMN "new_field" TEXT;'
    spend_logs_drift = 'ALTER TABLE "LiteLLM_SpendLogs" ADD COLUMN "new_field" TEXT;'
    legacy = 'DROP TABLE "LiteLLM_SpendLogs_legacy";'
    child = 'ALTER TABLE "LiteLLM_SpendLogs_2026_10" ADD CONSTRAINT "LiteLLM_SpendLogs_2026_10_pkey" PRIMARY KEY ("request_id", "startTime");'
    commands = [PKEY_REWRITE, drift, spend_logs_drift, legacy, child]

    assert filter_spend_logs_pkey_rewrite(commands, use_spend_logs_partitioning=True) == [
        drift,
        spend_logs_drift,
        legacy,
        child,
    ]


def test_check_prisma_schema_diff_forwards_partitioning_flag(  # test-quality-ok: forwarding test; the helper behaviour is covered by the filtering tests above
    mocker,
):
    """The boot-time wrapper must forward the flag to the helper."""
    from litellm.proxy.db import check_migration

    mocker.patch.object(check_migration, "verbose_logger", autospec=True)
    helper = mocker.patch.object(
        check_migration,
        "check_prisma_schema_diff_helper",
        return_value=(False, []),
    )

    check_migration.check_prisma_schema_diff(db_url="mock_url", use_spend_logs_partitioning=True)

    helper.assert_called_once_with("mock_url", True)
