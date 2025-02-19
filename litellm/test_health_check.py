import unittest
import asyncio
import copy
import os

# Constants
MAX_HEALTH_CHECK_RETRIES = 3  # Maximum retries before stopping health checks
use_background_health_checks = True  # Ensures we run checks

# Mock model list
llm_model_list = [
    {"model_name": "gpt-3.5-turbo", "health_check_enabled": True},
    {"model_name": "fake-openai-endpoint", "health_check_enabled": False},
]

health_check_results = {}
health_check_interval = 10  # Adjusted for faster debugging
health_check_details = True  # Debug logs enabled

async def perform_health_check(model_list, details):
    """Mock function to simulate health check"""
    healthy = []
    unhealthy = []

    for model in model_list:
        if not model.get("health_check_enabled", True):
            continue  # Skip models that have health checks disabled
        # Simulating a 50% chance of passing
        if os.urandom(1)[0] % 2 == 0:
            healthy.append(model["model_name"])
        else:
            unhealthy.append(model["model_name"])

    return healthy, unhealthy

async def _run_background_health_check():
    """Periodically run health checks in the background"""
    global health_check_results, llm_model_list, health_check_interval, health_check_details, use_background_health_checks

    _llm_model_list = copy.deepcopy(llm_model_list)

    if not _llm_model_list:
        return

    iteration = 0
    health_check_retries = 0

    while use_background_health_checks:
        iteration += 1
        try:
            healthy_endpoints, unhealthy_endpoints = await perform_health_check(
                model_list=_llm_model_list, details=health_check_details
            )

            health_check_results.update({
                "healthy_endpoints": healthy_endpoints,
                "unhealthy_endpoints": unhealthy_endpoints,
                "healthy_count": len(healthy_endpoints),
                "unhealthy_count": len(unhealthy_endpoints),
            })

            if not unhealthy_endpoints:
                break

            health_check_retries += 1
            if health_check_retries >= MAX_HEALTH_CHECK_RETRIES:
                break

        except Exception as e:
            print(f"[ERROR] Exception during health check: {str(e)}")

        await asyncio.sleep(health_check_interval)

class TestHealthCheck(unittest.TestCase):
    def test_health_check_runs(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_run_background_health_check())
        
        self.assertIn("healthy_endpoints", health_check_results)
        self.assertIn("unhealthy_endpoints", health_check_results)
        self.assertLessEqual(health_check_results["unhealthy_count"], MAX_HEALTH_CHECK_RETRIES)

if __name__ == "__main__":
    unittest.main()
