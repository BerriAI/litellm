import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.common_utils.timezone_utils import (
    get_budget_reset_time,
    get_budget_reset_timezone,
)


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


class TestGetBudgetResetTimezone:
    """
    Tests for get_budget_reset_timezone()
    """

    def test_returns_utc_when_timezone_not_set(self):
        """
        When litellm.timezone is not set, the function returns UTC. 
        """

        import litellm

        if hasattr(litellm, "timezone"):
            original = litellm.timezone
            delattr(litellm, "timezone")
            try:
                assert get_budget_reset_timezone() == "UTC"
            finally:
                litellm.timezone = original
        else:
            assert get_budget_reset_timezone() == "UTC"

    def test_returns_utc_when_timezone_is_none(self):
        """
        When litellm.timezone is None, the function returns UTC. 
        """

        import litellm

        original = getattr(litellm, "timezone", None)
        has_attr = hasattr(litellm, "timezone")

        try:
            litellm.timezone = None
            assert get_budget_reset_timezone() == "UTC"
        finally:
            if has_attr:
                litellm.timezone = original
            elif hasattr(litellm, "timezone"):
                delattr(litellm, "timezone")
    
    def test_returns_cofigured_timezone(self):
        """
        When litellm.timezone is any timezone ( Asia/Tokyo etc. ), the function returns connfigured value. 
        """

        import litellm

        original = getattr(litellm, "timezone", None)
        has_attr = hasattr(litellm, "timezone")

        try:
            litellm.timezone = "Asia/Tokyo"
            assert get_budget_reset_timezone() == "Asia/Tokyo"
        finally:
            if has_attr:
                litellm.timezone = original
            elif hasattr(litellm, "timezone"):
                delattr(litellm, "timezone")

    def test_returns_empty_string_fails_back_to_utc(self):
        """
        When litellm.timezone is empty, the function returns UTC. 
        """
        import litellm

        original = getattr(litellm, "timezone", None)
        has_attr = hasattr(litellm, "timezone")

        try:
            litellm.timezone = ""
            assert get_budget_reset_timezone() == "UTC"
        finally:
            if has_attr:
                litellm.timezone = original
            elif hasattr(litellm, "timezone"):
                delattr(litellm, "timezone")