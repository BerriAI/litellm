"""
LiteLLM Benchmark Load Testing with Locust

This script replicates the benchmark testing described in:
https://docs.litellm.ai/docs/benchmarks

Usage:
    # Set environment variables
    export LITELLM_HOST="http://your-load-balancer-url"
    export LITELLM_MASTER_KEY="your-master-key"

    # Run with benchmark parameters (1000 users, 500 spawn rate, 5 minutes)
    locust -f locustfile.py --host=$LITELLM_HOST --users=1000 --spawn-rate=500 --run-time=5m --headless

    # Run with web UI for interactive testing
    locust -f locustfile.py --host=$LITELLM_HOST

    # Run with custom parameters
    locust -f locustfile.py --host=$LITELLM_HOST --users=500 --spawn-rate=100 --run-time=10m --headless
"""

import os
import time
import json
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner


class LiteLLMUser(HttpUser):
    """
    Simulates a user making requests to LiteLLM proxy server.
    """

    # Wait time between tasks (benchmark guide specifies 0.5-1 second)
    wait_time = between(0.5, 1)

    def on_start(self):
        """
        Called when a simulated user starts.
        Sets up authentication and headers.
        """
        self.master_key = os.environ.get("LITELLM_MASTER_KEY")
        if not self.master_key:
            raise ValueError(
                "LITELLM_MASTER_KEY environment variable is required. "
                "Set it with: export LITELLM_MASTER_KEY='your-key'"
            )

        self.headers = {
            "Authorization": f"Bearer {self.master_key}",
            "Content-Type": "application/json",
        }

    @task(10)
    def chat_completion(self):
        """
        Main task: Send chat completion request to LiteLLM.
        This is weighted at 10 to be the primary task.
        """
        payload = {
            "model": "fake-openai-endpoint",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ],
        }

        with self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="Chat Completion"
        ) as response:
            if response.status_code == 200:
                # Check for LiteLLM overhead header
                overhead = response.headers.get("x-litellm-overhead-duration-ms")
                if overhead:
                    # Record custom metric for LiteLLM overhead
                    events.request.fire(
                        request_type="OVERHEAD",
                        name="LiteLLM Overhead (ms)",
                        response_time=float(overhead),
                        response_length=0,
                        exception=None,
                        context={}
                    )
                response.success()
            else:
                response.failure(f"Failed with status {response.status_code}: {response.text}")

    @task(1)
    def health_check(self):
        """
        Health check task to verify service is running.
        This is weighted at 1 to run occasionally.
        """
        with self.client.get(
            "/health/readiness",
            catch_response=True,
            name="Health Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(5)
    def streaming_completion(self):
        """
        Streaming chat completion request.
        This is weighted at 5 to run less frequently than regular completions.
        """
        payload = {
            "model": "fake-openai-endpoint",
            "messages": [
                {"role": "user", "content": "Tell me a short story"}
            ],
            "stream": True,
        }

        with self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json=payload,
            catch_response=True,
            stream=True,
            name="Streaming Completion"
        ) as response:
            if response.status_code == 200:
                # Consume the stream
                for chunk in response.iter_lines():
                    if chunk:
                        pass  # Process chunks if needed
                response.success()
            else:
                response.failure(f"Streaming failed: {response.status_code}")


class BenchmarkUser(HttpUser):
    """
    Simplified user class for pure benchmark testing.
    This mimics the exact behavior from the benchmark guide.
    """
    wait_time = between(0, 0.1)  # Minimal wait time for maximum load

    def on_start(self):
        self.master_key = os.environ.get("LITELLM_MASTER_KEY")
        if not self.master_key:
            raise ValueError("LITELLM_MASTER_KEY environment variable is required")

        self.headers = {
            "Authorization": f"Bearer {self.master_key}",
            "Content-Type": "application/json",
        }

    @task
    def benchmark_request(self):
        """
        Single benchmark request matching the benchmark guide.
        """
        payload = {
            "model": "fake-openai-endpoint",
            "messages": [{"role": "user", "content": "test"}],
        }

        start_time = time.time()
        with self.client.post(
            "/v1/chat/completions",
            headers=self.headers,
            json=payload,
            catch_response=True,
            name="Benchmark Request"
        ) as response:
            total_time = (time.time() - start_time) * 1000  # Convert to ms

            if response.status_code == 200:
                # Extract LiteLLM overhead
                overhead = response.headers.get("x-litellm-overhead-duration-ms", "0")
                litellm_overhead = float(overhead)

                # Record metrics
                events.request.fire(
                    request_type="METRIC",
                    name="LiteLLM Overhead",
                    response_time=litellm_overhead,
                    response_length=0,
                    exception=None,
                    context={}
                )

                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


# Custom event handlers for enhanced reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """
    Print test configuration when test starts.
    """
    print("\n" + "=" * 60)
    print("LiteLLM Benchmark Load Test")
    print("=" * 60)
    print(f"Host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if hasattr(environment.runner, 'target_user_count') else 'N/A'}")
    print("Benchmark Configuration: 4 instances × 4 workers")
    print("Expected Performance:")
    print("  - Median latency: ~100 ms")
    print("  - P95 latency: ~150 ms")
    print("  - Throughput: ~1,170 RPS")
    print("  - LiteLLM overhead: ~2 ms")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """
    Print summary when test stops.
    """
    print("\n" + "=" * 60)
    print("Test Completed")
    print("=" * 60)
    print("Compare your results with the benchmark:")
    print("https://docs.litellm.ai/docs/benchmarks")
    print("=" * 60 + "\n")


# Instructions for users
if __name__ == "__main__":
    print(__doc__)
