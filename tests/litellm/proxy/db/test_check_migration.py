import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


import json
import os
import sys
import time

import pytest
from fastapi.testclient import TestClient

import litellm


def test_check_migration_out_of_sync(mocker):
    """
    Test that the check_prisma_schema_diff function
    - 🚨 [IMPORTANT] Does NOT Raise an Exception when the Prisma schema is out of sync with the database.
    - logs an error when the Prisma schema is out of sync with the database.
    """
    # Mock the logger BEFORE importing the function
    mock_logger = mocker.patch("litellm._logging.verbose_logger")

    # Import the function after mocking the logger
    from litellm.proxy.db.check_migration import check_prisma_schema_diff

    # Mock the helper function to simulate out-of-sync state
    mock_diff_helper = mocker.patch(
        "litellm.proxy.db.check_migration.check_prisma_schema_diff_helper",
        return_value=(True, ["ALTER TABLE users ADD COLUMN new_field TEXT;"]),
    )

    # Run the function - it should not raise an error
    try:
        check_prisma_schema_diff(db_url="mock_url")
    except Exception as e:
        pytest.fail(f"check_prisma_schema_diff raised an unexpected exception: {e}")

    # Verify the logger was called with the expected message
    mock_logger.exception.assert_called_once()
    actual_message = mock_logger.exception.call_args[0][0]
    assert "prisma schema out of sync with db" in actual_message
