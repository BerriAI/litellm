import pytest

def pytest_collection_modifyitems(config, items):
    for item in items:
        # everything in this folder is considered the "heavy" shard
        item.add_marker(pytest.mark.shard_b)
