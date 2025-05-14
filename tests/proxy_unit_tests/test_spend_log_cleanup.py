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
    # Test case 1: No maximum_retention_period set
    general_settings.clear()
    assert _should_delete_spend_logs() is False

    # Test case 2: Valid integer maximum_retention_period
    general_settings["maximum_retention_period"] = 30
    assert _should_delete_spend_logs() is True

    # Test case 3: Valid string maximum_retention_period
    general_settings["maximum_retention_period"] = "30"
    assert _should_delete_spend_logs() is True

    # Test case 4: Invalid string maximum_retention_period
    general_settings["maximum_retention_period"] = "invalid"
    assert _should_delete_spend_logs() is False

    # Test case 5: None value
    general_settings["maximum_retention_period"] = None
    assert _should_delete_spend_logs() is False

    # Clean up
    general_settings.clear() 