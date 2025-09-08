import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time


def test_get_budget_reset_time():
    """
    Test that the budget reset time is set to the first of the next month
    """
    # Get the current date
    now = datetime.now(timezone.utc)

    # Calculate expected reset date (first of next month)
    if now.month == 12:
        expected_month = 1
        expected_year = now.year + 1
    else:
        expected_month = now.month + 1
        expected_year = now.year
    expected_reset_at = datetime(expected_year, expected_month, 1, tzinfo=timezone.utc)

    # Verify budget_reset_at is set to first of next month
    assert get_budget_reset_time(budget_duration="1mo") == expected_reset_at
