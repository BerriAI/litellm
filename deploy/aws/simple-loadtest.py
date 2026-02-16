"""
Simple load test for LiteLLM that tests health and readiness endpoints
"""

import os
import time
from locust import HttpUser, task, between, events


class HealthCheckUser(HttpUser):
    """
    Simple user that only tests health endpoints to measure infrastructure performance
    """
    wait_time = between(0.1, 0.3)

    @task(10)
    def readiness_check(self):
        """Test readiness endpoint"""
        with self.client.get(
            "/health/readiness",
            catch_response=True,
            name="Readiness Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(5)
    def liveliness_check(self):
        """Test liveliness endpoint"""
        with self.client.get(
            "/health/liveliness",
            catch_response=True,
            name="Liveliness Check"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")

    @task(1)
    def models_list(self):
        """Test models endpoint"""
        with self.client.get(
            "/v1/models",
            catch_response=True,
            name="List Models"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status: {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "=" * 60)
    print("LiteLLM Infrastructure Performance Test")
    print("=" * 60)
    print(f"Host: {environment.host}")
    print("Testing: Health endpoints and infrastructure")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("\n" + "=" * 60)
    print("Test Completed")
    print("=" * 60 + "\n")
