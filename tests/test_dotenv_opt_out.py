import os
from unittest.mock import patch

from litellm import _should_load_dotenv


def test_should_load_dotenv_defaults():
    """By default in DEV mode, dotenv should load."""
    with patch.dict(os.environ, {"LITELLM_MODE": "DEV", "LITELLM_DISABLE_DOTENV": ""}, clear=True):
        assert _should_load_dotenv() is True


def test_should_load_dotenv_production():
    """In PRODUCTION mode, dotenv should not load."""
    with patch.dict(os.environ, {"LITELLM_MODE": "PRODUCTION", "LITELLM_DISABLE_DOTENV": ""}, clear=True):
        assert _should_load_dotenv() is False


def test_should_load_dotenv_disabled():
    """When LITELLM_DISABLE_DOTENV is truthy, dotenv should not load even in DEV mode."""
    for truthy_val in ("1", "true", "True", "t", "T", "yes", "YES", "y", "Y"):
        with patch.dict(os.environ, {"LITELLM_MODE": "DEV", "LITELLM_DISABLE_DOTENV": truthy_val}, clear=True):
            assert _should_load_dotenv() is False, f"Failed for LITELLM_DISABLE_DOTENV={truthy_val}"


def test_should_load_dotenv_falsy():
    """When LITELLM_DISABLE_DOTENV is explicitly false, dotenv should load in DEV mode."""
    for falsy_val in ("0", "false", "False", "no", "NO"):
        with patch.dict(os.environ, {"LITELLM_MODE": "DEV", "LITELLM_DISABLE_DOTENV": falsy_val}, clear=True):
            assert _should_load_dotenv() is True, f"Failed for LITELLM_DISABLE_DOTENV={falsy_val}"
