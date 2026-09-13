import os
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest


_CONTROL_ENV_VARIABLES: Final = frozenset(
    {
        "LITELLM_DISABLE_DOTENV",
        "LITELLM_DEV_ENV_HOT_RELOAD",
        "LITELLM_DOTENV_LOADED",
        "LITELLM_MODE",
    }
)
_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[2]


def _dotenv_was_loaded(module: str, environment: Mapping[str, str]) -> bool:
    process_environment: Final = {
        key: value for key, value in os.environ.items() if key not in _CONTROL_ENV_VARIABLES
    } | dict(environment)
    script: Final = textwrap.dedent(
        f"""
        import os
        import dotenv

        def load_dotenv(*args, **kwargs):
            os.environ["LITELLM_DOTENV_LOADED"] = "1"
            return True

        dotenv.load_dotenv = load_dotenv
        import {module}
        print(f"LITELLM_DOTENV_LOADED={{os.environ.get('LITELLM_DOTENV_LOADED', '0')}}")
        """
    )
    result: Final = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPOSITORY_ROOT,
        env=process_environment,
        capture_output=True,
        text=True,
        check=True,
    )
    marker: Final = "LITELLM_DOTENV_LOADED="
    output_line: Final = next(line for line in result.stdout.splitlines() if line.startswith(marker))
    return output_line.removeprefix(marker) == "1"


def test_default_dev_mode_loads_dotenv() -> None:
    assert _dotenv_was_loaded("litellm", {})


@pytest.mark.parametrize("value", ("1", "true", "t", "yes", "y", "TRUE", "YeS"))
def test_disable_dotenv_truthy_values_prevent_loading(value: str) -> None:
    assert not _dotenv_was_loaded("litellm", {"LITELLM_DISABLE_DOTENV": value})


def test_disable_dotenv_false_value_preserves_loading() -> None:
    assert _dotenv_was_loaded("litellm", {"LITELLM_DISABLE_DOTENV": "false"})


def test_production_mode_still_skips_dotenv() -> None:
    assert not _dotenv_was_loaded("litellm", {"LITELLM_MODE": "PRODUCTION"})


def test_disable_dotenv_wins_over_dev_hot_reload() -> None:
    environment: Final = {
        "LITELLM_DISABLE_DOTENV": "1",
        "LITELLM_DEV_ENV_HOT_RELOAD": "True",
    }
    assert not _dotenv_was_loaded("litellm", environment)


def test_proxy_cli_respects_disable_dotenv() -> None:
    assert not _dotenv_was_loaded("litellm.proxy.proxy_cli", {"LITELLM_DISABLE_DOTENV": "1"})


def test_proxy_cli_default_dev_mode_loads_dotenv() -> None:
    assert _dotenv_was_loaded("litellm.proxy.proxy_cli", {})
