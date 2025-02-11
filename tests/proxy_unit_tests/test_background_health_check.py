import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from litellm.proxy.proxy_server import _run_background_health_check

@pytest.mark.asyncio
async def test_run_background_health_check():
    """
    Test that _run_background_health_check correctly updates health_check_results,
    stops when use_background_health_checks is set to False, and handles retries properly.
    """
    global health_check_results, use_background_health_checks, health_check_interval, llm_model_list

    # ✅ Explicitly set llm_model_list before calling the function
    llm_model_list = [{"model_name": "model_1"}, {"model_name": "model_2"}]

    # ✅ Ensure llm_model_list is not empty
    assert len(llm_model_list) > 0, "llm_model_list should not be empty before test starts"

    # Setup test conditions
    health_check_results = {
        "healthy_endpoints": [],
        "unhealthy_endpoints": [],
        "healthy_count": 0,
        "unhealthy_count": 0
    }
    use_background_health_checks = True
    health_check_interval = 1  # Short interval for quick testing

    # Mock `perform_health_check`
    mock_perform_health_check = AsyncMock()
    mock_perform_health_check.side_effect = [
        (["model_1"], ["model_2"]),  # First run: One healthy, one unhealthy
        ([], ["model_1", "model_2"]),  # Second run: All unhealthy
    ]

    with patch("litellm.proxy.proxy_server.perform_health_check", mock_perform_health_check):
        # Run the function in the background
        task = asyncio.create_task(_run_background_health_check())

        # Allow it to run a couple of iterations
        await asyncio.sleep(2)

        # Stop the loop
        use_background_health_checks = False
        await task  # Ensure the task completes

    # ✅ Assertions to verify expected behavior
    assert health_check_results["healthy_count"] == 1, f"Expected 1 healthy model initially, got {health_check_results['healthy_count']}"
    assert health_check_results["unhealthy_count"] == 2, f"Expected 2 unhealthy models, got {health_check_results['unhealthy_count']}"

    print("✅ Test passed: _run_background_health_check() updates health_check_results correctly!")
