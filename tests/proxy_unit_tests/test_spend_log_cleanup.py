"""
Test cases for spend log cleanup functionality
"""

import pytest
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup import _should_delete_spend_logs
from litellm.proxy.proxy_server import general_settings


@pytest.mark.asyncio
async def test_should_delete_spend_logs():
    """
    Test the _should_delete_spend_logs function with various scenarios
    """
    # Test case 1: No maximum_spend_logs_retention_period set
    general_settings.clear()
    assert _should_delete_spend_logs() is False

    # Test case 2: Valid integer maximum_spend_logs_retention_period (in seconds)
    general_settings["maximum_spend_logs_retention_period"] = 3600
    assert _should_delete_spend_logs() is False

    # Test case 3: Valid duration string - days
    general_settings["maximum_spend_logs_retention_period"] = "30d"
    assert _should_delete_spend_logs() is True

    # Test case 4: Valid duration string - hours
    general_settings["maximum_spend_logs_retention_period"] = "24h"
    assert _should_delete_spend_logs() is True

    # Test case 5: Valid duration string - minutes
    general_settings["maximum_spend_logs_retention_period"] = "60m"
    assert _should_delete_spend_logs() is True

    # Test case 6: Valid duration string - seconds
    general_settings["maximum_spend_logs_retention_period"] = "3600s"
    assert _should_delete_spend_logs() is True

    # Test case 7: Valid duration string - weeks
    general_settings["maximum_spend_logs_retention_period"] = "1w"
    assert _should_delete_spend_logs() is True

    # Test case 8: Valid duration string - months
    general_settings["maximum_spend_logs_retention_period"] = "1mo"
    assert _should_delete_spend_logs() is True

    # Test case 9: Invalid duration string
    general_settings["maximum_spend_logs_retention_period"] = "invalid"
    assert _should_delete_spend_logs() is False

    # Test case 10: None value
    general_settings["maximum_spend_logs_retention_period"] = None
    assert _should_delete_spend_logs() is False

    # Clean up
    general_settings.clear() 