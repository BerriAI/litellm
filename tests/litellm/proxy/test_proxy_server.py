import pytest

@pytest.mark.parametrize(
    "response_cost,expected_rounded_cost",
    [
        (0.1234567, "0.123457"),       # Test normal rounding
        (0.123456, "0.123456"),        # Test exact 6 decimal places
        (0, "0.0"),                    # Test zero
        (None, "None"),                # Test None value
        ("invalid", "invalid"),        # Test invalid string
        (1, "1.0"),                    # Test integer value
        (0.123, "0.123"),              # Test fewer decimal places
    ]
)
def test_response_cost_rounding(response_cost, expected_rounded_cost):
    """Test that response cost is safely rounded to 6 decimal places"""
    from litellm.proxy.utils import _safely_round_response_cost
    
    # Call the helper method directly
    result = _safely_round_response_cost(response_cost)
    
    # For numeric values, verify the actual float value is correct
    if response_cost is not None and not isinstance(response_cost, str):
        actual_cost = float(result)
        expected_cost = float(expected_rounded_cost)
        assert abs(actual_cost - expected_cost) < 1e-10, f"Expected {expected_rounded_cost}, got {result}"
    else:
        # For None or string values, compare the string representation
        assert result == expected_rounded_cost, f"Expected {expected_rounded_cost}, got {result}"