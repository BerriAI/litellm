import os

import pytest


@pytest.fixture(autouse=True)
def _hermetic_server_root_path():
    """Isolate MCP discovery tests from an ambient ``SERVER_ROOT_PATH``.

    The discovery routes and the 401 challenges read it, so a value inherited from the
    environment would silently rewrite every ``resource_metadata`` URL and make these tests
    depend on how the shard was invoked. Clearing it here pins the default (root-mounted)
    deployment; a test that exercises a sub-path deployment sets the value explicitly within
    its own body.
    """
    saved = os.environ.pop("SERVER_ROOT_PATH", None)
    try:
        yield
    finally:
        if saved is not None:
            os.environ["SERVER_ROOT_PATH"] = saved
