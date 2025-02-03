import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.requests import HTTPConnection
import os
import sys

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import warnings


def test_import_litellm():
    """
    Test contributed by https://github.com/SmartManoj
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import litellm

        assert (
            len(w) == 0
        ), f"Warnings were raised: {[str(warning.message) for warning in w]}"
