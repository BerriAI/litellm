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
    verify_module_id_consistency,
    create_auth_call_function,
    get_memory_test_config,
    create_mock_prisma_client,
    setup_proxy_server_dependencies,
    setup_proxy_server_dependencies_without_db,
    save_proxy_server_state,
    restore_proxy_server_state,
    force_gc,
)
from .constants import TEST_API_KEY, TEST_MASTER_KEY, TEST_DIFFERENT_MASTER_KEY

API_KEY = TEST_API_KEY

# Validate constants configuration at import time
assert TEST_API_KEY == TEST_MASTER_KEY, (
    "TEST_API_KEY must equal TEST_MASTER_KEY for WITHOUT DB test to work! "
    f"Got TEST_API_KEY={TEST_API_KEY}, TEST_MASTER_KEY={TEST_MASTER_KEY}"
)
assert TEST_API_KEY != TEST_DIFFERENT_MASTER_KEY, (
    "TEST_API_KEY must NOT equal TEST_DIFFERENT_MASTER_KEY for WITH DB test to work! "
    f"Got TEST_API_KEY={TEST_API_KEY}, TEST_DIFFERENT_MASTER_KEY={TEST_DIFFERENT_MASTER_KEY}"
)


@pytest.fixture(autouse=True, scope="function")
def cleanup_between_tests():
    """
    Fixture to ensure complete isolation between test cases.
    
    This fixture runs before and after each test to clean up state,
    ensuring tests don't interfere with each other.
    """
    # Cleanup before test
    print("\n[FIXTURE] Cleaning up state before test...", flush=True)
    force_gc()
    
    # Run the test
    yield
    
    # Cleanup after test
    print("\n[FIXTURE] Cleaning up state after test...", flush=True)
    force_gc()


@pytest.fixture
def setup_auth_dependencies(request):
    """
    Parametrizable fixture to set up auth dependencies WITH or WITHOUT database.
    
    Args:
        request: pytest request object with param indicating with_db (bool)
    
    This fixture creates either:
    - WITHOUT DB (with_db=False): Tests master key validation path
    - WITH DB (with_db=True): Tests full database-backed auth path
    
    Returns:
        proxy_server module (and mock_prisma if with_db=True)
    """
    with_db = request.param
    
    if with_db:
        print("\n[FIXTURE] Setting up auth dependencies WITH DB...", flush=True)
    else:
        print("\n[FIXTURE] Setting up auth dependencies WITHOUT DB...", flush=True)
    
    # Import proxy_server to set up dependencies
    import litellm.proxy.proxy_server as proxy_server
    from litellm.caching.caching import DualCache
    from litellm.proxy.utils import ProxyLogging
    
    # Save original state for restoration
    original_values = save_proxy_server_state(proxy_server)
    
    # Create real cache and logging objects
    real_cache = DualCache()
    real_logging = ProxyLogging(user_api_key_cache=real_cache)
    
    if with_db:
        # WITH DATABASE: Create mock Prisma client
        # Set master_key to DIFFERENT value to force database lookup path
        mock_prisma_client = create_mock_prisma_client(API_KEY)
        setup_proxy_server_dependencies(proxy_server, mock_prisma_client, real_cache, real_logging, TEST_DIFFERENT_MASTER_KEY)
        
        # Validate configuration
        assert proxy_server.master_key == TEST_DIFFERENT_MASTER_KEY, (
            f"Master key not set correctly! Expected {TEST_DIFFERENT_MASTER_KEY}, got {proxy_server.master_key}"
        )
        assert API_KEY != proxy_server.master_key, (
            f"API_KEY must NOT equal master_key for WITH DB test! Both are {API_KEY}"
        )
        
        print("[FIXTURE] Dependencies set up complete (WITH DB mock)", flush=True)
        print(f"[FIXTURE] Test API key: {API_KEY}", flush=True)
        print(f"[FIXTURE] Master key: {proxy_server.master_key}", flush=True)
        print(f"[FIXTURE] Keys are DIFFERENT → will trigger database path ✓", flush=True)
    else:
        # WITHOUT DATABASE: Use master key authentication path
        # Set master_key to SAME value as API_KEY to trigger master key path
        setup_proxy_server_dependencies_without_db(proxy_server, real_cache, real_logging, TEST_MASTER_KEY)
        
        # Validate configuration
        assert proxy_server.master_key == TEST_MASTER_KEY, (
            f"Master key not set correctly! Expected {TEST_MASTER_KEY}, got {proxy_server.master_key}"
        )
        assert API_KEY == proxy_server.master_key, (
            f"API_KEY must equal master_key for WITHOUT DB test! API_KEY={API_KEY}, master_key={proxy_server.master_key}"
        )
        
        print("[FIXTURE] Dependencies set up complete (WITHOUT DB - master key path)", flush=True)
        print(f"[FIXTURE] Test API key: {API_KEY}", flush=True)
        print(f"[FIXTURE] Master key: {proxy_server.master_key}", flush=True)
        print(f"[FIXTURE] Keys are SAME → will trigger master key path ✓", flush=True)
    
    yield proxy_server
    
    # Restore original values
    print("[FIXTURE] Restoring original dependencies...", flush=True)
    restore_proxy_server_state(proxy_server, original_values)
    print("[FIXTURE] Cleanup complete", flush=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setup_auth_dependencies,with_db,test_title",
    [
        (False, False, "Auth Memory Leak Test (WITHOUT DB - Master Key Path)"),
        (True, True, "Auth Memory Leak Test (WITH DB - Database Lookup Path)"),
    ],
    indirect=["setup_auth_dependencies"],
    ids=["without-db", "with-db"]
)
async def test_user_api_key_auth_memory_leak(setup_auth_dependencies, with_db, test_title):
    """
    Memory leak test for _user_api_key_auth_builder function using tracemalloc.
    Tests both WITH and WITHOUT database scenarios via parametrization.
    
    Goals:
    - Detect unbounded memory growth across repeated auth checks.
    - Filter out normal allocator noise and cache warm-up.
    - Fail only when memory truly grows over time.
    
    Technique:
    - Uses tracemalloc for deterministic Python-level memory tracking.
    - Double garbage collection ensures freed objects are fully released.
    - Warm-up phase excluded from growth analysis.
    - Rolling average smoothing suppresses transient allocator noise.
    
    Test Scenarios:
    
    1. WITHOUT DATABASE (with_db=False) - Master Key Path:
       - prisma_client=None
       - Validates API key against master_key using secrets.compare_digest()
       - Returns admin role immediately (no DB lookup)
       - Tests basic auth logic without database overhead
    
    2. WITH DATABASE (with_db=True) - Full Database Path:
       - Mock PrismaClient created
       - DB lookup via prisma_client.get_data()
       - Retrieves user/team information and permissions
       - Tests full auth logic including database queries
    
    Args:
        setup_auth_dependencies: Fixture that sets up auth dependencies
        with_db: Whether to test WITH database (True) or WITHOUT (False)
        test_title: Display title for this test variant
    """
    # Ensure fixture is executed and get proxy_server module
    proxy_server = setup_auth_dependencies
    
    # Track proxy_server module ID for consistency verification
    initial_id = id(proxy_server)
    print(f"\n[TEST] proxy_server module ID: {initial_id}", flush=True)
    verify_module_id_consistency(proxy_server, initial_id, "at test start")
    
    # Get standardized test configuration (all values from constants.py)
    config = get_memory_test_config()

    print_test_header(title=test_title)
    
    # Create the auth call function using the test API key
    # Master key is set to different value to force DB lookup path
    test_api_key = API_KEY
    call_user_api_key_auth = create_auth_call_function(master_key=test_api_key)
    
    tracemalloc.start()

    try:
        # --- Warm-up Phase ---
        print(f"\nWarming up with {config['warmup_batches']} batches of {config['batch_size']} calls each...")
        await run_warmup_phase(
            batch_size=config['batch_size'],
            warmup_batches=config['warmup_batches'],
            completion_func=call_user_api_key_auth,
            completion_kwargs={}
        )
        
        # Verify ID hasn't changed after warmup
        verify_module_id_consistency(proxy_server, initial_id, "after warmup")

        # --- Measurement Phase ---
        print(f"\nMeasuring memory over {config['num_batches']} batches...")
        memory_samples, error_counts = await run_measurement_phase(
            batch_size=config['batch_size'],
            num_batches=config['num_batches'],
            completion_func=call_user_api_key_auth,
            completion_kwargs={},
            tracemalloc_module=tracemalloc,
            litellm_module=None  # No need for litellm cleanup in auth test
        )
        
        # Verify ID hasn't changed after measurement
        verify_module_id_consistency(proxy_server, initial_id, "after measurement")

    finally:
        tracemalloc.stop()

    # --- Analysis Phase ---
    db_status = "WITH DB" if with_db else "WITHOUT DB"
    print_analysis_header(title=f"_user_api_key_auth_builder Memory Growth Analysis ({db_status})")

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

    # Detect memory leaks (including error-induced leaks)
    leak_detected, message = detect_memory_leak(
        growth_metrics=growth_metrics,
        memory_samples=memory_samples,
        error_counts=error_counts,
        max_growth_percent=config['max_growth_percent'],
        stabilization_tolerance_mb=config['stabilization_tolerance_mb'],
        tail_samples=tail_samples
    )

    if leak_detected:
        pytest.fail(message)
    
    print(message)
    print_memory_samples(memory_samples, num_samples=10)
    
    # Final verification that ID remained consistent throughout
    verify_module_id_consistency(proxy_server, initial_id, "at test end")
    print(f"\n[TEST] ✓ proxy_server module ID remained consistent throughout: {initial_id}", flush=True)
    print(f"[TEST] ✓ _user_api_key_auth_builder memory test complete ({db_status}) - no leaks detected", flush=True)


if __name__ == "__main__":
    """Run the test directly."""
    pytest.main([__file__, "-v", "-s"])
