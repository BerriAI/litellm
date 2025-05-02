# conftest.py

import importlib
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    curr_dir = os.getcwd()  # Get the current working directory
    sys.path.insert(
        0, os.path.abspath("../..")
    )  # Adds the project directory to the system path

    import litellm
    from litellm import Router

    # Store any existing Redis connections before reload
    redis_connections = []
    if hasattr(litellm, "cache") and hasattr(litellm.cache, "redis_cache"):
        redis_connections.append(litellm.cache.redis_cache)

    importlib.reload(litellm)

    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            import litellm.proxy.proxy_server
            importlib.reload(litellm.proxy.proxy_server)
    except Exception as e:
        print(f"Error reloading litellm.proxy.proxy_server: {e}")

    import asyncio

    # Create new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    print(litellm)
    yield

    # Teardown code (executes after the yield point)
    try:
        # Close any Redis connections
        for redis_conn in redis_connections:
            if hasattr(redis_conn, "close"):
                try:
                    loop.run_until_complete(redis_conn.close())
                except Exception as e:
                    print(f"Error closing Redis connection: {e}")
    except Exception as e:
        print(f"Error in Redis cleanup: {e}")

    try:
        # Cancel all running tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    except Exception as e:
        print(f"Error cancelling tasks: {e}")

    try:
        loop.close()
    except Exception as e:
        print(f"Error closing event loop: {e}")
    
    asyncio.set_event_loop(None)  # Remove the reference to the loop


def pytest_collection_modifyitems(config, items):
    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [
        item for item in items if "custom_logger" in item.parent.name
    ]
    other_tests = [item for item in items if "custom_logger" not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests
