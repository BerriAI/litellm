"""
Cleanup and garbage collection utilities for memory leak testing.

Provides functions for:
- Forcing thorough garbage collection
- Cleaning up litellm module state between tests
- Verifying module identity consistency during tests
"""

import gc
import time


def force_gc() -> None:
    """
    Run garbage collection three times with delays to ensure full cleanup.
    
    This helps ensure that all unreferenced objects are properly collected
    before measuring memory, reducing noise in memory measurements.
    
    The multiple passes with delays are necessary because:
    - First pass collects most objects
    - Second pass collects objects that became collectable after first pass
    - Third pass ensures complete cleanup
    - Delays allow finalizers to run between passes
    
    Example:
        >>> force_gc()  # Run before taking memory measurements
    """
    gc.collect()
    time.sleep(0.2)
    gc.collect()
    time.sleep(0.2)
    gc.collect()
    time.sleep(0.2)


def cleanup_litellm_state(litellm_module) -> None:
    """
    Clean up litellm module state and reload it to ensure test isolation.
    
    This function mirrors the cleanup pattern used in other litellm test suites
    (see tests/local_testing/conftest.py, tests/test_litellm/conftest.py).
    
    Steps:
    1. Flush HTTP client caches (critical for memory leak tests)
    2. Clear logging worker queue
    3. Reload module to reset all state (callbacks, caches, sessions)
    4. Force garbage collection
    
    Args:
        litellm_module: The litellm module instance to clean up
        
    Raises:
        Exception: If module reload fails (critical for test isolation)
        
    Example:
        >>> import litellm
        >>> cleanup_litellm_state(litellm)  # Clean state between tests
    """
    import asyncio
    import importlib
    
    # Step 1: Flush in-memory HTTP client cache
    # This is critical - prevents accumulation of HTTP clients across tests
    if hasattr(litellm_module, 'in_memory_llm_clients_cache'):
        try:
            litellm_module.in_memory_llm_clients_cache.flush_cache()
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not flush client cache: {e}", flush=True)
    
    # Step 2: Clear logging worker queue (prevents log accumulation)
    if hasattr(litellm_module, 'litellm_core_utils'):
        try:
            from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
            asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not clear logging queue: {e}", flush=True)
    
    # Step 3: Force garbage collection before reload
    # Ensures old objects are cleaned up before module reloads
    force_gc()
    
    # Step 4: Reload the module to get a completely fresh state
    # This resets all module-level state:
    # - Callbacks (success_callback, failure_callback, etc.)
    # - Caches (cache, in_memory_llm_clients_cache.cache_dict)
    # - Client sessions (client_session, aclient_session)
    # - All global configuration variables
    try:
        importlib.reload(litellm_module)
    except Exception as e:
        print(f"[CLEANUP] Error: Module reload failed: {e}", flush=True)
        raise  # Fail loudly if reload fails - this is critical for test isolation
    
    # Step 5: Force garbage collection after reload
    # Ensures old module state is fully cleaned up
    force_gc()


def verify_module_id_consistency(module, expected_id: int, stage: str = "") -> None:
    """
    Verify that a module's ID hasn't changed during test execution.
    
    This is useful for detecting if a module was accidentally reloaded during
    a test, which could invalidate memory measurements.
    
    Args:
        module: The module instance to check
        expected_id: The expected ID value (from id(module) at test start)
        stage: Optional description of when this check is being performed
    
    Raises:
        AssertionError: If the module ID doesn't match expected_id
        
    Example:
        >>> import litellm
        >>> module_id = id(litellm)
        >>> # ... run some test operations ...
        >>> verify_module_id_consistency(litellm, module_id, "after warmup")
    """
    current_id = id(module)
    stage_info = f" {stage}" if stage else ""
    assert current_id == expected_id, (
        f"Module ID changed{stage_info}! Expected {expected_id}, got {current_id}"
    )

