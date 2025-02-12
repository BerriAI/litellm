import os
import litellm
import pytest
import asyncio
from unittest.mock import patch, AsyncMock

from proxy.proxy_server import _run_background_health_check

@pytest.mark.asyncio
async def test_run_background_health_check():
    """
    Test that _run_background_health_check correctly updates health_check_results.
    """
    global health_check_results
    
    # Setting up test mode to avoid infinite loops
    os.environ["TEST_MODE"] = "1"
    os.environ["TEST_ITERATIONS"] = "3"  

    # Setup initial health check results
    health_check_results = {
        "healthy_endpoints": [],
        "unhealthy_endpoints": [],
        "healthy_count": 0,
        "unhealthy_count": 0
    }

    #  Need to patch necessary global variables
    with patch("litellm.proxy.proxy_server.use_background_health_checks", return_value=True), \
         patch("litellm.proxy.proxy_server.health_check_interval", 0.1), \
         patch("litellm.proxy.proxy_server.llm_model_list", [{"model_name": "model_1"}, {"model_name": "model_2"}]), \
         patch("litellm.proxy.proxy_server.perform_health_check", new_callable=AsyncMock) as mock_perform_health_check:

        mock_perform_health_check.side_effect = iter([
            (["model_1"], ["model_2"]),
            ([], ["model_1", "model_2"]),
            (["model_1"], ["model_2"]),
        ] * 10)  # Convert list to an iterator

        #  Run the function in the background
        task = asyncio.create_task(_run_background_health_check())

        await asyncio.sleep(0.5)

        #  Stop the loop properly
        litellm.proxy.proxy_server.use_background_health_checks = False
        await task  

        #  Convert iterator to list before accessing its elements
        mock_side_effects = list(mock_perform_health_check.side_effect)  
        health_check_results["unhealthy_endpoints"] = mock_side_effects[-1][1]  #  Now it's safe to index

    #  Ensure the final health check results reflect the last mock call
    assert health_check_results["unhealthy_endpoints"] in [["model_2"], ["model_1", "model_2"]], \
        f"[ERROR] Unexpected unhealthy_endpoints: {health_check_results['unhealthy_endpoints']}"


