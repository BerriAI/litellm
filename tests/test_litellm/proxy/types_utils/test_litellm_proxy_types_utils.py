import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

from litellm.proxy.types_utils.utils import security_checks

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


def test_security_checks_blocks_dangerous_modules():
    """
    Resolves: https://huntr.com/bounties/1d98bebb-6cf4-46c9-87c3-d3b1972973b5

    This test checks if the security_checks function correctly blocks the import of dangerous modules.
    """
    dangerous_module = "/usr/lib/python3/os.system"
    with pytest.raises(ImportError) as exc_info:
        security_checks(dangerous_module)

    assert "not allowed for security reasons" in str(exc_info.value)
    assert dangerous_module in str(exc_info.value)


def test_security_checks_various_dangerous_modules():
    dangerous_modules = [
        "subprocess.run",
        "socket.socket",
        "pickle.loads",
        "marshal.loads",
        "ctypes.CDLL",
        "builtins.eval",
        "__builtin__.exec",
        "shutil.rmtree",
        "multiprocessing.Process",
        "threading.Thread",
    ]

    for module in dangerous_modules:
        with pytest.raises(ImportError) as exc_info:
            security_checks(module)
        assert "not allowed for security reasons" in str(exc_info.value)
        assert module in str(exc_info.value)


def test_security_checks_case_insensitive():
    # Test that the check is case-insensitive
    variations = ["OS.system", "os.System", "Os.SyStEm", "SUBPROCESS.run"]

    for module in variations:
        with pytest.raises(ImportError) as exc_info:
            security_checks(module)
        assert "not allowed for security reasons" in str(exc_info.value)


def test_security_checks_nested_paths():
    # Test nested paths that contain dangerous modules
    nested_paths = [
        "some/path/to/os/system",
        "myproject/utils/subprocess_wrapper",
        "lib/helpers/socket_utils",
        "../../../system/os.py",
    ]

    for path in nested_paths:
        with pytest.raises(ImportError) as exc_info:
            security_checks(path)
        assert "not allowed for security reasons" in str(exc_info.value)
