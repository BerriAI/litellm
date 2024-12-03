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


import pytest
import time
from collections import defaultdict

# Store durations and retry information
test_durations = defaultdict(float)
retry_counts = defaultdict(int)


# Track test durations and retries
def pytest_runtest_protocol(item, nextitem):
    retry_count = getattr(item, "retry_count", 0)  # From pytest-retry plugin
    start_time = time.time()
    yield
    duration = time.time() - start_time
    test_durations[item.nodeid] += duration  # Cumulative duration
    retry_counts[item.nodeid] = retry_count  # Record retries


# Add total test time and retries to the summary
def pytest_terminal_summary(terminalreporter, config):
    terminalreporter.write("\nSummary of Test Durations (Including Retries):\n")

    for nodeid, duration in sorted(
        test_durations.items(), key=lambda x: x[1], reverse=True
    ):
        retries = retry_counts.get(nodeid, 0)
        terminalreporter.write(f"{duration:.2f}s (retries: {retries}) {nodeid}\n")

    terminalreporter.write("\nSlowest Tests (Including Retries):\n")
    slowest_tests = sorted(test_durations.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]
    for nodeid, duration in slowest_tests:
        retries = retry_counts.get(nodeid, 0)
        terminalreporter.write(f"{duration:.2f}s (retries: {retries}) {nodeid}\n")
