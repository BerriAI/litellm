"""
Performance utilities for LiteLLM proxy server.

This module provides performance monitoring and profiling functionality for endpoint
performance analysis using cProfile with configurable sampling rates.
"""

import asyncio
import cProfile
import functools
import threading
from pathlib import Path as PathLib

from litellm._logging import verbose_proxy_logger

# Global profiling state
_profile_lock = threading.Lock()
_profiler = None
_last_profile_file_path = None
_sample_counter = 0
_sample_counter_lock = threading.Lock()


def _should_sample(profile_sampling_rate: float) -> bool:
    """Determine if current request should be sampled based on sampling rate."""
    if profile_sampling_rate >= 1.0:
        return True  # Always sample
    elif profile_sampling_rate <= 0.0:
        return False  # Never sample
    
    # Use deterministic sampling based on counter for consistent rate
    global _sample_counter
    with _sample_counter_lock:
        _sample_counter += 1
        # Sample based on rate (e.g., 0.1 means sample every 10th request)
        should_sample = (_sample_counter % int(1.0 / profile_sampling_rate)) == 0
        return should_sample


def _start_profiling(profile_sampling_rate: float) -> None:
    """Start cProfile profiling once globally."""
    global _profiler
    with _profile_lock:
        if _profiler is None:
            _profiler = cProfile.Profile()
            _profiler.enable()
            verbose_proxy_logger.info(f"Profiling started with sampling rate: {profile_sampling_rate}")


def _start_profiling_for_request(profile_sampling_rate: float) -> bool:
    """Start profiling for a specific request (if sampling allows)."""
    if _should_sample(profile_sampling_rate):
        _start_profiling(profile_sampling_rate)
        return True
    return False


def _save_stats(profile_file: PathLib) -> None:
    """Save current stats directly to file."""
    with _profile_lock:
        if _profiler is None:
            return
        try:
            # Disable profiler temporarily to dump stats
            _profiler.disable()
            _profiler.dump_stats(str(profile_file))
            # Re-enable profiler to continue profiling
            _profiler.enable()
            verbose_proxy_logger.debug(f"Profiling stats saved to {profile_file}")
        except Exception as e:
            verbose_proxy_logger.error(f"Error saving profiling stats: {e}")
            # Make sure profiler is re-enabled even if there's an error
            try:
                _profiler.enable()
            except Exception:
                pass


def profile_endpoint(sampling_rate: float = 1.0):
    """Decorator to sample endpoint hits and save to a profile file.
    
    Args:
        sampling_rate: Rate of requests to profile (0.0 to 1.0)
                      - 1.0: Profile all requests (100%)
                      - 0.1: Profile 1 in 10 requests (10%)
                      - 0.0: Profile no requests (0%)
    """
    def decorator(func):
        def set_last_profile_path(path: PathLib) -> None:
            global _last_profile_file_path
            _last_profile_file_path = path

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                is_sampling = _start_profiling_for_request(sampling_rate)
                file_path_obj = PathLib("endpoint_profile.pstat")
                set_last_profile_path(file_path_obj)
                try:
                    result = await func(*args, **kwargs)
                    if is_sampling:
                        _save_stats(file_path_obj)
                    return result
                except Exception:
                    if is_sampling:
                        _save_stats(file_path_obj)
                    raise
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                is_sampling = _start_profiling_for_request(sampling_rate)
                file_path_obj = PathLib("endpoint_profile.pstat")
                set_last_profile_path(file_path_obj)
                try:
                    result = func(*args, **kwargs)
                    if is_sampling:
                        _save_stats(file_path_obj)
                    return result
                except Exception:
                    if is_sampling:
                        _save_stats(file_path_obj)
                    raise
            return sync_wrapper
    return decorator
