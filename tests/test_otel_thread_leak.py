import sys
import os
import threading
import time
import pytest

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.types.utils import StandardCallbackDynamicParams

def get_thread_count() -> int:
    """Helper to get active thread count"""
    return threading.active_count()

@pytest.fixture
def otel_logger():
    """Fixture to provide a clean OTEL logger for each test"""
    config = OpenTelemetryConfig(
        exporter="console",
        enable_metrics=False,
        service_name="litellm-unit-test"
    )
    return OpenTelemetry(config=config)

def test_otel_thread_leak_dynamic_headers(otel_logger):
    """
    Unit test to verify that calling get_tracer_to_use_for_request with 
    dynamic headers doesn't cause a linear thread leak.
    
    This test reproduces the issue where each unique team/key credential
    set causes a new TracerProvider (and its background threads) to be 
    spawned but never closed.
    """
    
    # 1. Setup dynamic header simulation (monkey-patch)
    # This simulates what LangfuseOtelLogger does for per-team keys
    def mock_construct_dynamic_headers(standard_callback_dynamic_params):
        if standard_callback_dynamic_params:
            return {"Authorization": "Bearer fake_token"}
        return None
    
    otel_logger.construct_dynamic_otel_headers = mock_construct_dynamic_headers
    
    # 2. Establish Baseline
    initial_threads = get_thread_count()
    
    # 3. Simulate requests
    num_requests = 10
    latencies = []
    
    print("\n🚀 Simulating requests with dynamic headers:")
    for i in range(num_requests):
        kwargs = {
            "standard_callback_dynamic_params": StandardCallbackDynamicParams(
                langfuse_public_key=f"key_{i}",
                langfuse_secret_key=f"secret_{i}",
            )
        }
        
        # Measure latency
        start_time = time.perf_counter()
        tracer = otel_logger.get_tracer_to_use_for_request(kwargs)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        print(f"   Request {i+1:2d}: Latency = {latency_ms:6.2f} ms")
        
        # Verify a tracer was actually returned
        assert tracer is not None
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"\n📊 Average Latency: {avg_latency:.2f} ms")
        
    # 4. Check for leaks
    # Allow for a small constant increase (OTEL might start a few shared threads)
    # but a linear leak would result in +10 or more threads here.
    final_threads = get_thread_count()
    thread_delta = final_threads - initial_threads
    
    print(f"\nThread growth: {thread_delta} threads across {num_requests} requests")
    
    # ASSERTION: The growth should be significantly less than 1 thread per request.
    # If the bug exists, thread_delta will be >= num_requests.
    assert thread_delta < (num_requests / 2), (
        f"Thread leak detected! Threads grew by {thread_delta} over {num_requests} requests. "
        "Each request with dynamic headers appears to be leaking background threads."
    )
