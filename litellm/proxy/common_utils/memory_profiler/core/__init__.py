"""
Core testing framework for memory leak detection.

This package provides the core functionality for:
- Cleanup and garbage collection utilities
- Test configuration builders
- Test execution and orchestration
"""

from .cleanup import (
    force_gc,
    cleanup_litellm_state,
    verify_module_id_consistency,
)
from .config import (
    get_completion_kwargs,
    get_completion_function,
    get_router_completion_kwargs,
    get_router_completion_function,
    create_fastapi_request,
    get_memory_test_config,
)
from .execution import (
    execute_single_request,
    run_warmup_phase,
    run_measurement_phase,
    run_memory_measurement_with_tracemalloc,
    analyze_and_detect_leaks,
)

__all__ = [
    # Cleanup utilities
    "force_gc",
    "cleanup_litellm_state",
    "verify_module_id_consistency",
    # Configuration builders
    "get_completion_kwargs",
    "get_completion_function",
    "get_router_completion_kwargs",
    "get_router_completion_function",
    "create_fastapi_request",
    "get_memory_test_config",
    # Execution orchestration
    "execute_single_request",
    "run_warmup_phase",
    "run_measurement_phase",
    "run_memory_measurement_with_tracemalloc",
    "analyze_and_detect_leaks",
]

