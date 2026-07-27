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


@pytest.mark.parametrize(
    "env_value, expected",
    [("0", 0.1), ("-5", 0.1), ("0.05", 0.1), ("30", 30.0), (None, 10.0)],
)
def test_proxy_shutdown_spend_drain_timeout_is_clamped_positive(env_value, expected):
    with mock.patch.dict(
        os.environ,
        {} if env_value is None else {"PROXY_SHUTDOWN_SPEND_DRAIN_TIMEOUT_SECONDS": env_value},
        clear=False,
    ):
        if env_value is None:
            os.environ.pop("PROXY_SHUTDOWN_SPEND_DRAIN_TIMEOUT_SECONDS", None)
        reloaded = importlib.reload(constants)
        try:
            assert reloaded.PROXY_SHUTDOWN_SPEND_DRAIN_TIMEOUT_SECONDS == expected
        finally:
            importlib.reload(constants)
