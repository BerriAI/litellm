# conftest.py

import importlib
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
import asyncio

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

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
    import asyncio

    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    # flush all logs
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())


    importlib.reload(litellm)

    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            import litellm.proxy.proxy_server

            importlib.reload(litellm.proxy.proxy_server)
    except Exception as e:
        print(f"Error reloading litellm.proxy.proxy_server: {e}")

    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    print(litellm)
    # from litellm import Router, completion, aembedding, acompletion, embedding
    yield

    # Teardown code (executes after the yield point)
    loop.close()  # Close the loop created earlier
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
