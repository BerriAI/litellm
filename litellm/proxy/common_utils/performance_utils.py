"""
Performance utilities for LiteLLM proxy server.

This module provides performance monitoring and profiling functionality for endpoint
performance analysis using cProfile with configurable sampling rates, and line_profiler
for line-by-line profiling.

See performance_utils.md for detailed usage examples and documentation.
"""

import atexit
import cProfile
import functools
import inspect
import threading
from pathlib import Path as PathLib
from typing import Any, Callable, Optional

from litellm._logging import verbose_proxy_logger

# Global profiling state
_profile_lock = threading.Lock()
_profiler = None
_last_profile_file_path = None
_sample_counter = 0
_sample_counter_lock = threading.Lock()

# Global line_profiler state
_line_profiler: Optional[Any] = None
_line_profiler_lock = threading.Lock()
_wrapped_functions: dict[str, Callable] = {}  # Store original functions


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

        if inspect.iscoroutinefunction(func):
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


def enable_line_profiler() -> None:
    """Enable line_profiler for dynamic function wrapping.
    
    Raises:
        ImportError: If line_profiler is not available
    """
    global _line_profiler
    from line_profiler import LineProfiler  # Will raise ImportError if not available
    
    with _line_profiler_lock:
        if _line_profiler is None:
            _line_profiler = LineProfiler()
            verbose_proxy_logger.info("Line profiler enabled")


def wrap_function_with_line_profiler(module: Any, function_name: str) -> bool:
    """Dynamically wrap a function with line_profiler.
    
    Args:
        module: The module containing the function
        function_name: Name of the function to wrap
        
    Returns:
        True if wrapping was successful, False otherwise
    """
    try:
        enable_line_profiler()  # May raise ImportError if not available
    except ImportError:
        return False
    
    if _line_profiler is None:
        return False
    
    try:
        original_function = getattr(module, function_name, None)
        if original_function is None:
            verbose_proxy_logger.warning(
                f"Function {function_name} not found in module {module.__name__}"
            )
            return False
        
        # Store original function if not already wrapped
        if function_name not in _wrapped_functions:
            _wrapped_functions[function_name] = original_function
        
        # Wrap with line_profiler
        profiled_function = _line_profiler(original_function)
        setattr(module, function_name, profiled_function)
        
        verbose_proxy_logger.info(
            f"Wrapped {module.__name__}.{function_name} with line_profiler"
        )
        return True
    except Exception as e:
        verbose_proxy_logger.error(
            f"Error wrapping {function_name} with line_profiler: {e}"
        )
        return False


def wrap_function_directly(func: Callable) -> Callable:
    """Wrap a function directly with line_profiler.
    
    This is the recommended way to profile functions, especially closures or
    functions created dynamically (like wrapper_async in litellm/utils.py).
    
    Args:
        func: The function to wrap
        
    Returns:
        The wrapped function that will be profiled when called
        
    Raises:
        ImportError: If line_profiler is not available
        RuntimeError: If line_profiler cannot be enabled or function cannot be wrapped
    """
    import warnings
    
    enable_line_profiler()  # Will raise ImportError if not available
    
    if _line_profiler is None:
        raise RuntimeError("Line profiler was not initialized")
    
    # Suppress warnings about __wrapped__ - we intentionally want to profile the wrapper
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*__wrapped__.*', category=UserWarning)
        # Add function to line_profiler and wrap it
        _line_profiler.add_function(func)
        profiled_function = _line_profiler(func)
    
    verbose_proxy_logger.info(
        f"Wrapped function {func.__name__} with line_profiler"
    )
    return profiled_function


def collect_line_profiler_stats(output_file: Optional[str] = None) -> None:
    """Collect and save line_profiler statistics.
    
    This can be called manually to collect stats at any time, or it's
    automatically called on shutdown if register_shutdown_handler() was used.
    
    Args:
        output_file: Optional path to save stats. If None, prints to stdout.
    """
    global _line_profiler
    
    with _line_profiler_lock:
        if _line_profiler is None:
            verbose_proxy_logger.debug("Line profiler not enabled, nothing to collect")
            return
        
        try:
            if output_file:
                # Save to file
                output_path = PathLib(output_file)
                _line_profiler.dump_stats(str(output_path))
                verbose_proxy_logger.info(
                    f"Line profiler stats saved to {output_path}"
                )
            else:
                # Print to stdout
                from io import StringIO
                
                stream = StringIO()
                _line_profiler.print_stats(stream=stream)
                stats_output = stream.getvalue()
                verbose_proxy_logger.info("Line profiler stats:\n" + stats_output)
        except Exception as e:
            verbose_proxy_logger.error(f"Error collecting line profiler stats: {e}")


def register_shutdown_handler(output_file: Optional[str] = None) -> None:
    """Register a shutdown handler to collect line_profiler stats.
    
    This registers an atexit handler that will automatically save profiling
    statistics when the Python process exits. Safe to call multiple times
    (only registers once).
    
    Args:
        output_file: Optional path to save stats on shutdown.
                     Defaults to 'line_profile_stats.lprof'
    """
    if output_file is None:
        output_file = "line_profile_stats.lprof"
    
    def shutdown_handler():
        collect_line_profiler_stats(output_file=output_file)
    
    atexit.register(shutdown_handler)
    verbose_proxy_logger.debug(f"Registered line_profiler shutdown handler for {output_file}")
