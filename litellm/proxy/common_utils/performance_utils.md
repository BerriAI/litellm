# Performance Utilities Documentation

This module provides performance monitoring and profiling functionality for LiteLLM proxy server using `cProfile` and `line_profiler`.

## Table of Contents

- [Line Profiler Usage](#line-profiler-usage)
  - [Example 1: Wrapping a function directly](#example-1-wrapping-a-function-directly)
  - [Example 2: Wrapping a module function dynamically](#example-2-wrapping-a-module-function-dynamically)
  - [Example 3: Manual stats collection](#example-3-manual-stats-collection)
  - [Example 4: Analyzing the profile output](#example-4-analyzing-the-profile-output)
  - [Example 5: Using in a decorator pattern](#example-5-using-in-a-decorator-pattern)
- [cProfile Usage](#cprofile-usage)
- [Installation](#installation)
- [Notes](#notes)

## Line Profiler Usage

### Example 1: Wrapping a function directly

This is how it's used in `litellm/utils.py` to profile `wrapper_async`:

```python
from litellm.proxy.common_utils.performance_utils import (
    register_shutdown_handler,
    wrap_function_directly,
)

def client(original_function):
    @wraps(original_function)
    async def wrapper_async(*args, **kwargs):
        # ... function implementation ...
        pass
    
    # Wrap the function with line_profiler
    wrapper_async = wrap_function_directly(wrapper_async)
    
    # Register shutdown handler to collect stats on server shutdown
    register_shutdown_handler(output_file="wrapper_async_line_profile.lprof")
    
    return wrapper_async
```

### Example 2: Wrapping a module function dynamically

```python
import my_module
from litellm.proxy.common_utils.performance_utils import (
    wrap_function_with_line_profiler,
    register_shutdown_handler,
)

# Wrap a function in a module
wrap_function_with_line_profiler(my_module, "expensive_function")

# Register shutdown handler
register_shutdown_handler(output_file="my_profile.lprof")

# Now all calls to my_module.expensive_function will be profiled
my_module.expensive_function()
```

### Example 3: Manual stats collection

```python
from litellm.proxy.common_utils.performance_utils import (
    wrap_function_directly,
    collect_line_profiler_stats,
)

def my_function():
    # ... implementation ...
    pass

# Wrap the function
my_function = wrap_function_directly(my_function)

# Run your code
my_function()

# Collect stats manually (instead of waiting for shutdown)
collect_line_profiler_stats(output_file="manual_profile.lprof")
```

### Example 4: Analyzing the profile output

After running your code, analyze the `.lprof` file:

```bash
# View the profile
python -m line_profiler wrapper_async_line_profile.lprof

# Save to text file
python -m line_profiler wrapper_async_line_profile.lprof > profile_report.txt
```

The output shows:
- **Line #**: Line number in the source file
- **Hits**: Number of times the line was executed
- **Time**: Total time spent on that line (in microseconds)
- **Per Hit**: Average time per execution
- **% Time**: Percentage of total function time
- **Line Contents**: The actual source code

Example output:
```
Timer unit: 1e-06 s

Total time: 3.73697 s
File: litellm/utils.py
Function: client.<locals>.wrapper_async at line 1657

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
  1657                                               @wraps(original_function)
  1658                                               async def wrapper_async(*args, **kwargs):
  1659      2005       7577.1      3.8      0.2          print_args_passed_to_litellm(...)
  1763      2005    1351909.0    674.3    36.2          result = await original_function(*args, **kwargs)
  1846      4010    1543688.1    385.0    41.3          update_response_metadata(...)
```

### Example 5: Using in a decorator pattern

```python
from litellm.proxy.common_utils.performance_utils import (
    wrap_function_directly,
    register_shutdown_handler,
)

def profile_decorator(func):
    # Wrap the function
    profiled_func = wrap_function_directly(func)
    
    # Register shutdown handler (only once)
    if not hasattr(profile_decorator, '_registered'):
        register_shutdown_handler(output_file="decorated_functions.lprof")
        profile_decorator._registered = True
    
    return profiled_func

@profile_decorator
async def my_async_function():
    # This function will be profiled
    pass
```

## cProfile Usage

### Example: Using the profile_endpoint decorator

```python
from litellm.proxy.common_utils.performance_utils import profile_endpoint

@profile_endpoint(sampling_rate=0.1)  # Profile 10% of requests
async def my_endpoint():
    # ... implementation ...
    pass
```

The `sampling_rate` parameter controls what percentage of requests are profiled:
- `1.0`: Profile all requests (100%)
- `0.1`: Profile 1 in 10 requests (10%)
- `0.0`: Profile no requests (0%)

## Installation

`line_profiler` must be installed to use the line profiling functionality:

```bash
pip install line_profiler
```

On Windows with Python 3.14+, you may need to install Microsoft Visual C++ Build Tools to compile `line_profiler` from source.

## Notes

- The profiler aggregates stats by source code location, so multiple instances of the same function (e.g., closures) will be profiled together
- Stats are automatically collected on server shutdown via `atexit` handler when using `register_shutdown_handler()`
- You can also manually collect stats using `collect_line_profiler_stats()`
- The line profiler will fail with an `ImportError` if `line_profiler` is not installed (as configured in `litellm/utils.py`)

## API Reference

### `wrap_function_directly(func: Callable) -> Callable`

Wrap a function directly with line_profiler. This is the recommended way to profile functions, especially closures or functions created dynamically.

**Raises:**
- `ImportError`: If line_profiler is not available
- `RuntimeError`: If line_profiler cannot be enabled or function cannot be wrapped

### `wrap_function_with_line_profiler(module: Any, function_name: str) -> bool`

Dynamically wrap a function in a module with line_profiler.

**Returns:** `True` if wrapping was successful, `False` otherwise

### `collect_line_profiler_stats(output_file: Optional[str] = None) -> None`

Collect and save line_profiler statistics. If `output_file` is provided, saves to file. Otherwise, prints to stdout.

### `register_shutdown_handler(output_file: Optional[str] = None) -> None`

Register an `atexit` handler that will automatically save profiling statistics when the Python process exits. Safe to call multiple times (only registers once).

**Default output file:** `line_profile_stats.lprof` if not specified

### `profile_endpoint(sampling_rate: float = 1.0)`

Decorator to sample endpoint hits and save to a profile file using cProfile.

**Args:**
- `sampling_rate`: Rate of requests to profile (0.0 to 1.0)

