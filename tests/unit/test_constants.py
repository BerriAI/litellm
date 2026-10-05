import ast
import inspect
import json
from typing import Final
from unittest import mock

import httpx
import pytest
import respx
from fastapi.testclient import TestClient


import importlib

import litellm
from litellm import constants


def test_azure_computer_use_default_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Match Azure pricing, verified 2026-10-04: https://azure.microsoft.com/en-us/pricing/details/openai/"""
    budget_notice_state: Final[bool] = constants.budget_reservation_disabled_info_emitted
    try:
        with monkeypatch.context() as clean_environment:
            clean_environment.delenv("AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS", raising=False)
            clean_environment.delenv("AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS", raising=False)
            default_constants: Final = importlib.reload(constants)
            assert default_constants.AZURE_COMPUTER_USE_INPUT_COST_PER_1K_TOKENS == 0.003
            assert default_constants.AZURE_COMPUTER_USE_OUTPUT_COST_PER_1K_TOKENS == 0.012
    finally:
        importlib.reload(constants)
        constants.budget_reservation_disabled_info_emitted = budget_notice_state


def test_clarifai_models_are_distinct() -> None:
    assert "clarifai/qwen.qwenLM.Qwen3-30B-A3B-Thinking-2507" in constants.clarifai_models
    assert "clarifai/openai.chat-completion.gpt-5-nano" in constants.clarifai_models


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
