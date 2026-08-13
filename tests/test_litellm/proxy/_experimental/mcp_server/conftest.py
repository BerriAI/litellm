import os

import pytest


@pytest.fixture(autouse=True)
def _hermetic_server_root_path():
    """Isolate MCP discovery tests from a leaked ``SERVER_ROOT_PATH``.

    ``tests/test_litellm/proxy/test_custom_proxy.py`` sets ``SERVER_ROOT_PATH`` at import time
    (its app mounts under a custom path) and never restores it, so in a shared shard the value
    leaks into this process. The discovery routes and the 401 challenges read it, so a leaked
    value would silently rewrite every ``resource_metadata`` URL and make these tests depend on
    shard ordering. Clearing it here pins the default (root-mounted) deployment; a test that
    exercises a sub-path deployment sets the value explicitly within its own body.
    """
    saved = os.environ.pop("SERVER_ROOT_PATH", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["SERVER_ROOT_PATH"] = saved
