import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))


def test_check_migration_out_of_sync(mocker):
    """
    Test that the check_prisma_schema_diff function
    - 🚨 [IMPORTANT] Does NOT Raise an Exception when the Prisma schema is out of sync with the database.
    - logs an error when the Prisma schema is out of sync with the database.
    """
    from litellm.proxy.db import check_migration

    # Patch the module-local logger/helper so this test remains stable even if
    # check_migration was already imported earlier in the test session.
    mock_logger = mocker.patch.object(check_migration, "verbose_logger")
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
    mock_logger.exception.assert_called_once()
    actual_message = mock_logger.exception.call_args[0][0]
    assert "prisma schema out of sync with db" in actual_message
