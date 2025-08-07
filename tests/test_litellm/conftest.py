# conftest.py

import importlib
import os
import sys
import tempfile
import random
import string

import pytest

# Set up a temporary log directory and file BEFORE importing litellm
temp_dir = tempfile.mkdtemp(prefix="litellm_test_")
test_log_file = os.path.join(temp_dir, "test_litellm.log")

# Store original log file for cleanup
orig_log_file = os.getenv("LITELLM_LOG_FILE")

# Set environment variables to use temporary files BEFORE importing litellm
os.environ["LITELLM_LOG_FILE"] = test_log_file

# Import litellm after setting up the environment
sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm




@pytest.fixture(scope="function")
def temp_log_file():
    """
    Creates a temporary log file in /tmp/litellm<random_number>.log for testing.
    Returns the path to the temporary log file and cleans it up after the test.
    """
    # Generate a random number for the log file
    random_number = ''.join(random.choices(string.digits, k=8))
    log_file_path = f"/tmp/litellm{random_number}.log"
    
    # Set the environment variable for litellm to use this temporary log file
    original_log_file = os.environ.get("LITELLM_LOG_FILE")
    os.environ["LITELLM_LOG_FILE"] = log_file_path
    
    yield log_file_path
    
    # Cleanup: Restore original environment variable and remove the temporary file
    if original_log_file is not None:
        os.environ["LITELLM_LOG_FILE"] = original_log_file
    else:
        os.environ.pop("LITELLM_LOG_FILE", None)
    
    # Remove the temporary log file if it exists
    if os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
        except OSError:
            pass  # Ignore errors if file can't be removed


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_log_dir():
    """
    Cleans up the temporary log directory created at module import time.
    This runs once per test session after all tests are complete.
    """
    yield
        
    if orig_log_file is not None:
            os.environ["LITELLM_LOG_FILE"] = orig_log_file
    else:
        os.environ.pop("LITELLM_LOG_FILE", None)

    # Cleanup: Remove the temporary directory created at module import time
    if os.path.exists(temp_dir):
        try:
            # Remove the test log file first
            if os.path.exists(test_log_file):
                os.remove(test_log_file)
            
            # Remove the temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            pass  # Ignore errors if cleanup fails


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

    litellm.in_memory_llm_clients_cache.flush_cache()

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

