"""
Memory leak testing suite for LiteLLM.

This package provides a comprehensive framework for detecting memory leaks
using tracemalloc, with support for:

- **Analysis**: Memory growth analysis and leak detection algorithms
- **Core**: Testing framework (cleanup, configuration, execution orchestration)
- **Snapshot**: Memory snapshot capture and storage system
- **Utils**: Shared utilities for conversions and output formatting
- **Constants**: Centralized configuration constants

## Quick Start

```python
import litellm
from tests.memory_leak_tests import (
    run_memory_measurement_with_tracemalloc,
    analyze_and_detect_leaks,
    get_memory_test_config,
    get_completion_kwargs,
    get_completion_function,
)

# Configure test
config = get_memory_test_config(batch_size=50, num_batches=5)
completion_func = get_completion_function(litellm, use_async=True, streaming=False)
completion_kwargs = get_completion_kwargs()

# Run measurement (with optional smart capture tuning)
memory_samples, error_counts = await run_memory_measurement_with_tracemalloc(
    completion_func=completion_func,
    completion_kwargs=completion_kwargs,
    config=config,
    test_name="my_test",
    memory_increase_threshold_pct=2.0,  # Optional: capture when memory grows >2%
    periodic_sample_interval=100,        # Optional: capture every 100 requests
)

# Analyze results
analyze_and_detect_leaks(memory_samples, error_counts, config)
```

## Module Organization

- `analysis/` - Memory analysis and leak detection
- `core/` - Core testing framework
- `snapshot/` - Snapshot capture and storage
- `tests/` - Test suite implementations
- `utils/` - Shared utilities
- `constants.py` - Configuration constants
"""

# Core execution and orchestration
from .core.execution import (
    execute_single_request,
    run_warmup_phase,
    run_measurement_phase,
    run_memory_measurement_with_tracemalloc,
    analyze_and_detect_leaks,
)

# Cleanup utilities
from .core.cleanup import (
    force_gc,
    cleanup_litellm_state,
    verify_module_id_consistency,
)

# Configuration builders
from .core.config import (
    get_completion_kwargs,
    get_completion_function,
    get_router_completion_kwargs,
    get_router_completion_function,
    create_fastapi_request,
    get_memory_test_config,
)

# Analysis functions
from .analysis.detection import (
    detect_error_induced_memory_leak,
    detect_memory_leak,
)
from .analysis.growth import (
    prepare_memory_analysis,
    calculate_rolling_average,
    analyze_memory_growth,
)
from .analysis.noise import check_measurement_noise

# Snapshot functions
from .snapshot.capture import (
    should_capture_detailed_snapshot,
    capture_request_memory_snapshot,
)
from .snapshot.storage import (
    sanitize_filename,
    save_buffered_snapshots_to_json,
    get_snapshot_file_info,
    save_batch_snapshots,
    print_final_snapshot_summary,
)

# Utilities
from .utils.conversions import bytes_to_mb
from .utils.formatting import (
    print_analysis_header,
    print_test_header,
    print_growth_metrics,
    print_memory_samples,
)

# Constants
from .constants import (
    DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT,
    DEFAULT_PERIODIC_SAMPLE_INTERVAL,
)

__all__ = [
    # Core execution
    "execute_single_request",
    "run_warmup_phase",
    "run_measurement_phase",
    "run_memory_measurement_with_tracemalloc",
    "analyze_and_detect_leaks",
    # Cleanup
    "force_gc",
    "cleanup_litellm_state",
    "verify_module_id_consistency",
    # Configuration
    "get_completion_kwargs",
    "get_completion_function",
    "get_router_completion_kwargs",
    "get_router_completion_function",
    "create_fastapi_request",
    "get_memory_test_config",
    # Analysis
    "detect_error_induced_memory_leak",
    "detect_memory_leak",
    "prepare_memory_analysis",
    "calculate_rolling_average",
    "analyze_memory_growth",
    "check_measurement_noise",
    # Snapshot
    "should_capture_detailed_snapshot",
    "capture_request_memory_snapshot",
    "sanitize_filename",
    "save_buffered_snapshots_to_json",
    "get_snapshot_file_info",
    "save_batch_snapshots",
    "print_final_snapshot_summary",
    # Utilities
    "bytes_to_mb",
    "print_analysis_header",
    "print_test_header",
    "print_growth_metrics",
    "print_memory_samples",
    # Constants
    "DEFAULT_MEMORY_INCREASE_THRESHOLD_PCT",
    "DEFAULT_PERIODIC_SAMPLE_INTERVAL",
]

__version__ = "1.0.0"
