import pytest
from litellm.litellm_core_utils.duration_parser import duration_in_seconds

def test_unlimited_budget_duration():
    # 1. Test that the bug is fixed (Empty/None values safely return None)
    assert duration_in_seconds("None") is None
    assert duration_in_seconds("null") is None
    assert duration_in_seconds("") is None
    assert duration_in_seconds(None) is None

    # 2. Test that normal math still works (30 days = 2,592,000 seconds)
    assert duration_in_seconds("30d") == 2592000