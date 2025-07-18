"""Test safe_divide_seconds utility function"""
import time
from datetime import timedelta

import pytest

from litellm.litellm_core_utils.core_helpers import safe_divide_seconds


def test_safe_divide_seconds_with_float():
    """Test safe_divide_seconds with float time duration"""
    result = safe_divide_seconds(10.0, 2.0)
    assert result == 5.0


def test_safe_divide_seconds_with_timedelta():
    """Test safe_divide_seconds with timedelta duration"""
    td = timedelta(seconds=10)
    result = safe_divide_seconds(td, 2.0)
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


def test_safe_divide_seconds_with_timedelta_and_zero():
    """Test safe_divide_seconds with timedelta and zero denominator"""
    td = timedelta(seconds=10)
    result = safe_divide_seconds(td, 0.0)
    assert result is None
    
    # With custom default
    result = safe_divide_seconds(td, 0.0, default=100.0)
    assert result == 100.0


def test_safe_divide_seconds_integration_with_time_time():
    """Test safe_divide_seconds with time.time() style values"""
    start_time = time.time()
    time.sleep(0.1)
    end_time = time.time()
    
    response_ms = end_time - start_time
    result = safe_divide_seconds(response_ms, 10.0)
    assert result is not None
    assert result > 0