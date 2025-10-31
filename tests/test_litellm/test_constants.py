import inspect
import json
import os
import sys
from unittest import mock

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../.."))  #

import importlib

import litellm
from litellm import constants


def test_all_numeric_constants_can_be_overridden():
    """
    Test that all integer and float constants in constants.py can be overridden with environment variables.
    This ensures that any new constants added in the future will be configurable via environment variables.
    """
    # Get all attributes from the constants module
    constants_attributes = inspect.getmembers(constants)

    # Filter for uppercase constants (by convention) that are integers or floats
    # Exclude booleans since bool is a subclass of int in Python
    numeric_constants = [
        (name, value)
        for name, value in constants_attributes
        if name.isupper() and isinstance(value, (int, float)) and not isinstance(value, bool)
    ]

    # Ensure we found some constants to test
    assert len(numeric_constants) > 0, "No numeric constants found to test"

    print("all numeric constants", json.dumps(numeric_constants, indent=4))

    # Verify all numeric constants have environment variable support
    for name, value in numeric_constants:
        # Skip constants that are not meant to be overridden (if any)
        if name.startswith("_"):
            continue

        # Create a test value that's different from the default
        test_value = value + 1 if isinstance(value, int) else value + 0.1

        # Set the environment variable
        with mock.patch.dict(os.environ, {name: str(test_value)}):
            print("overriding", name, "with", test_value)
            importlib.reload(constants)

            # Get the new value after reload
            new_value = getattr(constants, name)

            # Verify the value was overridden
            assert (
                new_value == test_value
            ), f"Failed to override {name} with environment variable. Expected {test_value}, got {new_value}"
