import ast
import inspect
import json
import logging
from unittest import mock

import httpx
import pytest
import respx
from fastapi.testclient import TestClient


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
    ("cli_value", "litellm_cli_value", "expected"),
    [
        ("48", None, 48),
        (None, "48", 48),
        (None, None, 24),
        ("48", "72", 48),
    ],
    ids=("canonical-only", "alias-only", "default", "canonical-wins"),
)
def test_cli_jwt_expiration_hours_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    cli_value: str | None,
    litellm_cli_value: str | None,
    expected: int,
) -> None:
    monkeypatch.delenv("CLI_JWT_EXPIRATION_HOURS", raising=False)
    monkeypatch.delenv("LITELLM_CLI_JWT_EXPIRATION_HOURS", raising=False)

    try:
        if cli_value is not None:
            monkeypatch.setenv("CLI_JWT_EXPIRATION_HOURS", cli_value)
        if litellm_cli_value is not None:
            monkeypatch.setenv("LITELLM_CLI_JWT_EXPIRATION_HOURS", litellm_cli_value)

        importlib.reload(litellm.constants)
        assert litellm.constants.CLI_JWT_EXPIRATION_HOURS == expected
    finally:
        monkeypatch.delenv("CLI_JWT_EXPIRATION_HOURS", raising=False)
        monkeypatch.delenv("LITELLM_CLI_JWT_EXPIRATION_HOURS", raising=False)
        importlib.reload(litellm.constants)


@pytest.mark.parametrize(
    ("raw", "expected", "warns"),
    [
        (None, 5.0, False),
        ("2.5", 2.5, False),
        (" 12 ", 12.0, False),
        ("inf", 5.0, True),
        ("-inf", 5.0, True),
        ("nan", 5.0, True),
        ("0", 5.0, True),
        ("-1", 5.0, True),
        ("five", 5.0, True),
        ("", 5.0, True),
    ],
    ids=("unset", "override", "padded-override", "inf", "negative-inf", "nan", "zero", "negative", "text", "empty"),
)
def test_registry_load_timeout_accepts_only_finite_positive_values(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raw: str | None,
    expected: float,
    warns: bool,
) -> None:
    """A timeout of inf never fires and one of zero always does, so neither may reach the registry loads."""
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.addHandler(caplog.handler)
    monkeypatch.delenv("REGISTRY_LOAD_TIMEOUT_SECONDS", raising=False)

    try:
        if raw is not None:
            monkeypatch.setenv("REGISTRY_LOAD_TIMEOUT_SECONDS", raw)

        importlib.reload(litellm.constants)
        assert litellm.constants.REGISTRY_LOAD_TIMEOUT_SECONDS == expected
        warned = [
            record
            for record in caplog.records
            if record.levelno == logging.WARNING and "REGISTRY_LOAD_TIMEOUT_SECONDS" in record.getMessage()
        ]
        assert bool(warned) is warns
    finally:
        litellm_logger.removeHandler(caplog.handler)
        monkeypatch.delenv("REGISTRY_LOAD_TIMEOUT_SECONDS", raising=False)
        importlib.reload(litellm.constants)
