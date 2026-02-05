import os
from unittest import mock
from litellm.proxy import utils


# Test the utility function logic
def test_get_server_root_path_unset():
    """
    Test that get_server_root_path returns empty string when SERVER_ROOT_PATH is unset
    """
    with mock.patch.dict(os.environ, {}, clear=True):
        # We need to make sure SERVER_ROOT_PATH is not in env
        if "SERVER_ROOT_PATH" in os.environ:
            del os.environ["SERVER_ROOT_PATH"]

        root_path = utils.get_server_root_path()
        assert (
            root_path == ""
        ), "Should return empty string when unset to allow X-Forwarded-Prefix"


def test_get_server_root_path_set():
    """
    Test that get_server_root_path returns the value when SERVER_ROOT_PATH is set
    """
    with mock.patch.dict(os.environ, {"SERVER_ROOT_PATH": "/my-path"}, clear=True):
        root_path = utils.get_server_root_path()
        assert root_path == "/my-path", "Should return the set value"


def test_get_server_root_path_empty_string():
    """
    Test that get_server_root_path returns empty string when SERVER_ROOT_PATH is explicitly empty
    """
    with mock.patch.dict(os.environ, {"SERVER_ROOT_PATH": ""}, clear=True):
        root_path = utils.get_server_root_path()
        assert (
            root_path == ""
        ), "Should return empty string when explicitly set to empty"


# Integration test simulation for FastAPI app initialization
def test_fastapi_app_initialization_mock():
    """
    Simulate how proxy_server.py initializes FastAPI app with the root_path.
    We don't import proxy_server because it has global side effects/singletons.
    Instead we verify the logic flow.
    """
    from fastapi import FastAPI

    # CASE 1: Proxy Mode (Unset)
    with mock.patch.dict(os.environ, {}, clear=True):
        if "SERVER_ROOT_PATH" in os.environ:
            del os.environ["SERVER_ROOT_PATH"]

        server_root_path = utils.get_server_root_path()
        app = FastAPI(root_path=server_root_path)
        assert app.root_path == ""

    # CASE 2: Direct Mode (Set)
    with mock.patch.dict(os.environ, {"SERVER_ROOT_PATH": "/custom-root"}, clear=True):
        server_root_path = utils.get_server_root_path()
        app = FastAPI(root_path=server_root_path)
        assert app.root_path == "/custom-root"
