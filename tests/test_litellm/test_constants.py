import ast
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


def _build_constant_env_var_map() -> dict[str, str]:
    """
    Build a mapping of CONSTANT_NAME -> ENV_VAR_NAME by parsing constants.py.

    This keeps the test resilient when a constant name and env var name differ
    (e.g., aliases like LITELLM_* env vars).
    """
    env_var_map: dict[str, str] = {}
    constants_source = inspect.getsource(constants)
    parsed = ast.parse(constants_source)

    for node in parsed.body:
        if not isinstance(node, ast.Assign):
            continue

        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue

        constant_name = node.targets[0].id
        env_var_name = None

        for child in ast.walk(node.value):
            if not isinstance(child, ast.Call):
                continue

            # os.getenv("ENV_NAME", default)
            if (
                isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "os"
                and child.func.attr == "getenv"
                and len(child.args) >= 1
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
            ):
                env_var_name = child.args[0].value
                break

            # get_env_int("ENV_NAME", default)
            if (
                isinstance(child.func, ast.Name)
                and child.func.id == "get_env_int"
                and len(child.args) >= 1
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
            ):
                env_var_name = child.args[0].value
                break

        if env_var_name:
            env_var_map[constant_name] = env_var_name

    return env_var_map


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
        if name.isupper()
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ]

    # Ensure we found some constants to test
    assert len(numeric_constants) > 0, "No numeric constants found to test"

    print("all numeric constants", json.dumps(numeric_constants, indent=4))

    # Discover exact env vars from constants.py to avoid brittle hardcoded mappings.
    constant_to_env_var = _build_constant_env_var_map()

    # Verify all numeric constants have environment variable support
    for name, value in numeric_constants:
        # Skip constants that are not meant to be overridden (if any)
        if name.startswith("_"):
            continue

        # Create a test value that's different from the default
        test_value = value + 1 if isinstance(value, int) else value + 0.1

        # Use the env var name that the constants module actually reads
        env_var_name = constant_to_env_var.get(name, name)

        # Set the environment variable
        with mock.patch.dict(os.environ, {env_var_name: str(test_value)}):
            print("overriding", name, "with", test_value)
            importlib.reload(constants)

            # Get the new value after reload
            new_value = getattr(constants, name)

            # Verify the value was overridden
            assert (
                new_value == test_value
            ), f"Failed to override {name} with environment variable. Expected {test_value}, got {new_value}"
