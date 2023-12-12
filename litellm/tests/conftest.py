# conftest.py

import pytest

def pytest_collection_modifyitems(config, items):
    # Separate tests in 'test_amazing_proxy_custom_logger.py' and other tests
    custom_logger_tests = [item for item in items if 'custom_logger' in item.parent.name]
    other_tests = [item for item in items if 'custom_logger' not in item.parent.name]

    # Sort tests based on their names
    custom_logger_tests.sort(key=lambda x: x.name)
    other_tests.sort(key=lambda x: x.name)

    # Reorder the items list
    items[:] = custom_logger_tests + other_tests