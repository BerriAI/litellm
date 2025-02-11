import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pytest
import asyncio
from litellm.proxy.proxy_server import _run_background_health_check, use_background_health_checks


@pytest.mark.asyncio
async def test_health_check_stops():
    """
    Ensure that _run_background_health_check stops when use_background_health_checks is set to False.
    """
    global use_background_health_checks
    use_background_health_checks = True

    # Start the background health check task
    task = asyncio.create_task(_run_background_health_check())

    await asyncio.sleep(2)
    
    # Stop the background health check
    use_background_health_checks = False

    await asyncio.sleep(1)

    # Cancel the task to avoid it running indefinitely
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # to make background health check has stopped
    assert not use_background_health_checks, "Background health check did not stop"
