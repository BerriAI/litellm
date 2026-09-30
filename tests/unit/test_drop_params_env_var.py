import os
import subprocess
import sys

import pytest


def _import_litellm_with(configured: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", "import litellm; print(litellm.drop_params)"],
        env={**os.environ, "LITELLM_DROP_PARAMS": configured},
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.mark.parametrize("configured, expected", [("false", "False"), ("true", "True"), ("", "False")])
def test_litellm_drop_params_env_var_is_parsed_as_a_flag(configured, expected):
    result = _import_litellm_with(configured)

    assert result.stdout.strip() == expected
    assert "is not a flag value" not in result.stderr


def test_litellm_drop_params_env_var_non_flag_value_stays_on_with_a_warning():
    result = _import_litellm_with("temperature")

    assert result.stdout.strip() == "True"
    assert (
        "LITELLM_DROP_PARAMS='temperature' is not a flag value, treating it as on. Set it to true or false"
        in result.stderr
    )
