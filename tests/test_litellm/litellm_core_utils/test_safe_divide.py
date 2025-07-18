"""Test safe_divide utility function"""
import time
from datetime import timedelta

import pytest

from litellm.litellm_core_utils.core_helpers import safe_divide


def test_safe_divide_with_float():
    """Test safe_divide with float numerator"""
    result = safe_divide(10.0, 2.0)
    assert result == 5.0


def test_safe_divide_with_timedelta():
    """Test safe_divide with timedelta numerator"""
    td = timedelta(seconds=10)
    result = safe_divide(td, 2.0)
    assert result == 5.0


def test_safe_divide_with_zero_denominator():
    """Test safe_divide with zero denominator returns default"""
    result = safe_divide(10.0, 0.0)
    assert result is None
    
    # With custom default
    result = safe_divide(10.0, 0.0, default=0.0)
    assert result == 0.0


def test_safe_divide_with_negative_denominator():
    """Test safe_divide with negative denominator returns default"""
    result = safe_divide(10.0, -5.0)
    assert result is None
    
    # With custom default
    result = safe_divide(10.0, -5.0, default=0.0)
    assert result == 0.0


def test_safe_divide_with_timedelta_and_zero():
    """Test safe_divide with timedelta and zero denominator"""
    td = timedelta(seconds=10)
    result = safe_divide(td, 0.0)
    assert result is None
    
    # With custom default
    result = safe_divide(td, 0.0, default=100.0)
    assert result == 100.0


def test_safe_divide_integration_with_time_time():
    """Test safe_divide with time.time() style values"""
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    
    response_ms = end_time - start_time
    result = safe_divide(response_ms, 10.0)
    assert result is not None
    assert result > 0