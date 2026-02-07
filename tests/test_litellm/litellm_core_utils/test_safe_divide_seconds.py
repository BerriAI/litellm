"""Test safe_divide_seconds utility function"""
import time

import pytest

from litellm.litellm_core_utils.core_helpers import safe_divide_seconds


def test_safe_divide_seconds_basic():
    """Test safe_divide_seconds with basic division"""
    result = safe_divide_seconds(10.0, 2.0)
    assert result == 5.0


def test_safe_divide_seconds_with_zero_denominator():
    """Test safe_divide_seconds with zero denominator returns default"""
    result = safe_divide_seconds(10.0, 0.0)
    assert result is None
    
    # With custom default
    result = safe_divide_seconds(10.0, 0.0, default=0.0)
    assert result == 0.0


def test_safe_divide_seconds_with_negative_denominator():
    """Test safe_divide_seconds with negative denominator returns default"""
    result = safe_divide_seconds(10.0, -5.0)
    assert result is None
    
    # With custom default
    result = safe_divide_seconds(10.0, -5.0, default=0.0)
    assert result == 0.0


def test_safe_divide_seconds_integration_with_time_time():
    """Test safe_divide_seconds with time.time() style values"""
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    
    response_seconds = end_time - start_time
    result = safe_divide_seconds(response_seconds, 10.0)
    assert result is not None
    assert result > 0