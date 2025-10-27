"""
Memory leak test for LiteLLM Proxy _user_api_key_auth_builder function.

This module tests the actual _user_api_key_auth_builder function to detect memory leaks.

There are two tests in this module:
1. test_user_api_key_auth_memory_leak - Tests WITHOUT database (prisma_client=None)
   - Uses master key validation path
   - Tests basic auth logic without DB queries
   
2. test_user_api_key_auth_memory_leak_with_db - Tests WITH database (mock PrismaClient)
   - Uses database lookup path
   - Tests full auth logic including DB queries, cache, and user permissions
"""

import sys
import os
import tracemalloc

import pytest
from unittest.mock import AsyncMock, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../.."))

from .memory_test_helpers import (
    run_warmup_phase,
    run_measurement_phase,
    calculate_rolling_average,
    analyze_memory_growth,
    detect_memory_leak,
    print_test_header,
    print_analysis_header,
    print_growth_metrics,
    print_memory_samples,
    create_auth_call_function,
    get_memory_test_config,
    create_mock_prisma_client,
    setup_proxy_server_dependencies,
    save_proxy_server_state,
    restore_proxy_server_state,
)
from .constants import TEST_API_KEY

API_KEY = TEST_API_KEY

@pytest.fixture
def setup_auth_dependencies_with_db():
    """
    Fixture to set up all dependencies needed by _user_api_key_auth_builder WITH database.
    
    This fixture creates a mock PrismaClient to test the database-backed auth path.
    With a mock DB, the auth flow:
    - Looks up API keys in the database
    - Retrieves user/team information
    - Validates user permissions and budgets
    - Returns appropriate user role
    
    This tests the full auth logic including database interactions.
    """
    print("\n[FIXTURE] Setting up auth dependencies with DB...", flush=True)
    
    # Import proxy_server to set up dependencies
    import litellm.proxy.proxy_server as proxy_server
    from litellm.caching.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    
    # Save original state for restoration
    original_values = save_proxy_server_state(proxy_server)
    
    # Create real cache and logging objects
    real_cache = DualCache()
    real_logging = ProxyLogging(user_api_key_cache=real_cache)
    
    # Create mock Prisma client
    mock_prisma_client = create_mock_prisma_client(API_KEY)
    
    # Set up the dependencies
    setup_proxy_server_dependencies(proxy_server, mock_prisma_client, real_cache, real_logging)
    
    print("[FIXTURE] Dependencies set up complete (with DB mock)", flush=True)
    
    yield proxy_server, mock_prisma_client
    
    # Restore original values
    print("[FIXTURE] Restoring original dependencies...", flush=True)
    restore_proxy_server_state(proxy_server, original_values)
    
    print("[FIXTURE] Cleanup complete", flush=True)



def test_user_api_key_auth_memory_leak_with_db(setup_auth_dependencies_with_db):
    """
    Memory leak test for _user_api_key_auth_builder function WITH database using tracemalloc.
    
    Goals:
    - Detect unbounded memory growth across repeated auth checks with DB lookups.
    - Filter out normal allocator noise and cache warm-up.
    - Fail only when memory truly grows over time.
    
    Technique:
    - Uses tracemalloc for deterministic Python-level memory tracking.
    - Double garbage collection ensures freed objects are fully released.
    - Warm-up phase excluded from growth analysis.
    - Rolling average smoothing suppresses transient allocator noise.
    
    Code Path Tested (WITH DATABASE MOCK):
    The fixture creates a mock PrismaClient, forcing the database-backed auth path:
    1. pre_db_read_auth_checks() - validates request structure
    2. get_api_key() - extracts API key from headers
    3. Skip JWT/OAuth2 checks (disabled in test config)
    4. get_key_object() with check_cache_only=True - cache miss (returns None)
    5. secrets.compare_digest(api_key, master_key) - SKIPPED (token in cache from step 4 cache miss)
    6. get_key_object() with check_cache_only=False - DB lookup via prisma_client.get_data()
    7. Returns LiteLLM_VerificationToken from mock DB
    8. Converts to UserAPIKeyAuth object
    9. asyncio.create_task(_cache_key_object()) - caches the result
    10. Returns UserAPIKeyAuth object with user permissions
    
    This path exercises:
    - Full auth logic including database queries
    - Cache lookups and writes (real cache, not mock)
    - User/team/permission lookups
    - Budget validation logic
    - UserAPIKeyAuth object creation with full metadata
    
    The fixture sets up proxy_server dependencies with a mock database.
    """
    # Ensure fixture is executed (sets up dependencies)
    _ = setup_auth_dependencies_with_db
    
    # Get standardized test configuration (all values from constants.py)
    config = get_memory_test_config()

    print_test_header(title="_user_api_key_auth_builder Memory Leak Detection Test (WITH DB)")
    
    # Create the auth call function using the test API key
    # Master key is set to different value to force DB lookup path
    test_api_key = API_KEY
    call_user_api_key_auth = create_auth_call_function(master_key=test_api_key)
    
    tracemalloc.start()

    try:
        # --- Warm-up Phase ---
        print(f"\nWarming up with {config['warmup_batches']} batches of {config['batch_size']} calls each...")
        run_warmup_phase(
            batch_size=config['batch_size'],
            warmup_batches=config['warmup_batches'],
            completion_func=call_user_api_key_auth,
            completion_kwargs={}
        )

        # --- Measurement Phase ---
        print(f"\nMeasuring memory over {config['num_batches']} batches...")
        memory_samples = run_measurement_phase(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            completion_func=call_user_api_key_auth,
            completion_kwargs={},
            tracemalloc_module=tracemalloc,
            litellm_module=None  # No need for litellm cleanup in auth test
        )

    finally:
        tracemalloc.stop()

    # --- Analysis Phase ---
    print_analysis_header(title="_user_api_key_auth_builder Memory Growth Analysis (WITH DB)")

    if len(memory_samples) < config['sample_window'] * 2:
        pytest.skip("Not enough samples for reliable growth analysis")

    # Calculate dynamic parameters based on sample_window
    rolling_avg = calculate_rolling_average(memory_samples, config['sample_window'])
    # Use 2x the sample_window for averaging (ensures we smooth over enough data)
    num_samples_for_avg = min(config['sample_window'] * 2, len(rolling_avg) // 3)
    # Use 3x the sample_window for tail analysis (detect continuous growth)
    tail_samples = min(config['sample_window'] * 3, len(memory_samples) // 2)
    
    growth_metrics = analyze_memory_growth(rolling_avg, num_samples_for_avg=num_samples_for_avg)
    
    print_growth_metrics(growth_metrics)

    # Detect memory leaks
    leak_detected, message = detect_memory_leak(
        growth_metrics=growth_metrics,
        memory_samples=memory_samples,
        max_growth_percent=config['max_growth_percent'],
        stabilization_tolerance_mb=config['stabilization_tolerance_mb'],
        tail_samples=tail_samples
    )

    if leak_detected:
        pytest.fail(message)
    
    print(message)
    print_memory_samples(memory_samples, num_samples=10)
    
    print("\n[TEST] ✓ _user_api_key_auth_builder memory test complete (WITH DB) - no leaks detected", flush=True)


if __name__ == "__main__":
    """Run the test directly."""
    pytest.main([__file__, "-v", "-s"])
