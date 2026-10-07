import ast, asyncio, os
import inspect
import json
from unittest import mock

import httpx
import pytest
import respx
from fastapi.testclient import TestClient


import importlib

import litellm
from litellm import constants, get_llm_provider, MorphChatConfig
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome
from unittest.mock import patch


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


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def setup_and_teardown(event_loop):
    import litellm

    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    asyncio.set_event_loop(event_loop)
    yield
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    pending = asyncio.all_tasks(event_loop)
    for task in pending:
        task.cancel()
    if pending:
        event_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
    "cohere_key": getattr(litellm, "cohere_key", None),
}


@pytest.fixture
def _populate_known_models_for_morph_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    for attr, value in vars(litellm).items():
        if attr.endswith("_models") and isinstance(value, set):
            monkeypatch.setattr(litellm, attr, value.copy())
    monkeypatch.setattr(
        litellm,
        "models_by_provider",
        {
            provider: models.copy()
            for provider, models in litellm.models_by_provider.items()
        },
    )
    litellm.add_known_models()


@pytest.mark.usefixtures(
    "_vcr_outcome_gate", "setup_and_teardown", "_populate_known_models_for_morph_tests"
)
def test_morph_config_get_provider_info():
    """Test that MorphChatConfig returns correct provider info."""
    config = MorphChatConfig()

    # Test with environment variable
    with patch.dict(os.environ, {"MORPH_API_KEY": "test-key-from-env"}):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.morphllm.com/v1"
        assert api_key == "test-key-from-env"

    # Test with passed api_key
    api_base, api_key = config._get_openai_compatible_provider_info(None, "direct-key")
    assert api_base == "https://api.morphllm.com/v1"
    assert api_key == "direct-key"

    # Test with custom api_base
    api_base, api_key = config._get_openai_compatible_provider_info("https://custom.morph.com", "key")
    assert api_base == "https://custom.morph.com"
    assert api_key == "key"

@pytest.mark.usefixtures(
    "_vcr_outcome_gate", "setup_and_teardown", "_populate_known_models_for_morph_tests"
)
def test_morph_get_llm_provider():
    """Test that get_llm_provider correctly identifies morph models."""
    # Test with morph/model format
    _, custom_llm_provider, _, _ = get_llm_provider("morph/morph-v3-large")
    assert custom_llm_provider == "morph"

    _, custom_llm_provider, _, _ = get_llm_provider("morph/morph-v3-fast")
    assert custom_llm_provider == "morph"

@pytest.mark.usefixtures(
    "_vcr_outcome_gate", "setup_and_teardown", "_populate_known_models_for_morph_tests"
)
def test_morph_in_provider_lists():
    """Test that morph is included in all necessary provider lists."""
    import litellm
    from litellm.constants import (
        openai_compatible_endpoints,
        openai_compatible_providers,
    )

    # Check morph is in openai_compatible_providers
    assert "morph" in openai_compatible_providers

    # Check morph endpoint is in openai_compatible_endpoints
    assert "https://api.morphllm.com/v1" in openai_compatible_endpoints

    # Check morph is in provider_list
    assert "morph" in litellm.provider_list

    # Check models are in model_list after initialization
    assert all(model in litellm.model_list for model in ["morph/morph-v3-large", "morph/morph-v3-fast"])

@pytest.mark.usefixtures(
    "_vcr_outcome_gate", "setup_and_teardown", "_populate_known_models_for_morph_tests"
)
def test_morph_supported_params():
    """Test that MorphChatConfig returns correct supported parameters."""
    config = MorphChatConfig()
    supported_params = config.get_supported_openai_params("morph/morph-v3-large")

    expected_params = [
        "messages",
        "model",
        "stream",
    ]

    assert all(param in supported_params for param in expected_params)

@pytest.mark.usefixtures(
    "_vcr_outcome_gate", "setup_and_teardown", "_populate_known_models_for_morph_tests"
)
def test_morph_custom_llm_provider():
    """Test that morph models are correctly identified."""
    config = MorphChatConfig()
    assert config.custom_llm_provider == "morph"
