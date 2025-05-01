# conftest.py

import asyncio
import importlib

import pytest


@pytest.fixture()
def reload_litellm():
    """
    Speed up testing by removing callbacks being chained?
    """
    import litellm

    importlib.reload(litellm)

    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            import litellm.proxy.proxy_server

            importlib.reload(litellm.proxy.proxy_server)
    except Exception as e:
        print(f"Error reloading litellm.proxy.proxy_server: {e}")


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
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
