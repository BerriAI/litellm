# conftest.py

import pytest, sys, os
import importlib
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
    sys.path.insert(0, os.path.abspath("../.."))  # Adds the project directory to the system path
    import litellm
    importlib.reload(litellm)
    print(litellm)
    # from litellm import Router, completion, aembedding, acompletion, embedding
    yield

def pytest_collection_modifyitems(config, items):
    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [item for item in items if 'custom_logger' in item.parent.name]
    other_tests = [item for item in items if 'custom_logger' not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests