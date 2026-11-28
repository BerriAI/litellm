"""`litellm --validate_config`: the one-shot command that loads a config without serving it.

`ProxyInitializationHelpers.run_config_validation` (`litellm/proxy/proxy_cli.py`) runs the same
`ProxyConfig.load_config` the server runs at startup, then prints `LiteLLM: config OK (<n> models)` and
exits 0, or prints `LiteLLM: config validation failed: <error>` to stderr and exits 1. Deploy pipelines
use it as a pre-flight check, so the exit codes and the two message prefixes are a contract the Rust
config loader must keep when it takes over this command.

These tests start their own process and need no proxy, database or Redis, so they live in a
`*_real_io.py` file and request none of the gateway fixtures. They run the CLI with `DATABASE_URL`
removed, which is also the contract: validation must not need a database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.integration.configuration.dec1_cfgload_support import (
    VALIDATION_FAILED_PREFIX,
    VALIDATION_OK_PREFIX,
    validate_config,
    write_yaml_text,
)
from tests.integration.configuration.dec1_config_support import deployment, write_config

_BASE: Final = "http://127.0.0.1:9/v1"


def test_a_valid_config_exits_zero_and_reports_its_model_count(tmp_path: Path) -> None:
    """A config with two `model_list` entries. The command must exit 0 and print `LiteLLM: config OK (2
    models)` on stdout, with nothing starting with the failure prefix on stderr."""
    config: Final = write_config(tmp_path, {"model_list": [deployment("dec1-a", _BASE), deployment("dec1-b", _BASE)]})
    result: Final = validate_config(config)
    assert result.exit_code == 0, result.stderr[-4000:]
    assert "LiteLLM: config OK (2 models)" in result.stdout
    assert VALIDATION_FAILED_PREFIX not in result.stderr


def test_models_from_included_files_are_counted(tmp_path: Path) -> None:
    """The root declares one model and includes `more.yaml` with two more. The command must exit 0 and
    print `LiteLLM: config OK (3 models)`: validation runs the include merge."""
    write_config(tmp_path, {"model_list": [deployment("dec1-b", _BASE), deployment("dec1-c", _BASE)]}, "more.yaml")
    config: Final = write_config(tmp_path, {"model_list": [deployment("dec1-a", _BASE)], "include": ["more.yaml"]})
    result: Final = validate_config(config)
    assert result.exit_code == 0, result.stderr[-4000:]
    assert "LiteLLM: config OK (3 models)" in result.stdout


def test_a_config_with_no_models_is_valid_with_a_count_of_zero(tmp_path: Path) -> None:
    """A config with only `litellm_settings: {drop_params: true}`. The command must exit 0 and print
    `LiteLLM: config OK (0 models)`: a model list is optional."""
    config: Final = write_config(tmp_path, {"litellm_settings": {"drop_params": True}})
    result: Final = validate_config(config)
    assert result.exit_code == 0, result.stderr[-4000:]
    assert "LiteLLM: config OK (0 models)" in result.stdout


def test_a_reference_to_an_unset_variable_does_not_fail_validation(tmp_path: Path) -> None:
    """A deployment's `api_key` is `os.environ/DEC1_VALIDATE_UNSET` and that variable is absent. The
    command must exit 0: an unset reference resolves to null and is only a problem at call time, so a
    pre-flight check run without production secrets still passes."""
    config: Final = write_config(
        tmp_path, {"model_list": [deployment("dec1-a", _BASE, api_key="os.environ/DEC1_VALIDATE_UNSET")]}
    )
    result: Final = validate_config(config, remove_environment=("DEC1_VALIDATE_UNSET",))
    assert result.exit_code == 0, result.stderr[-4000:]
    assert VALIDATION_OK_PREFIX in result.stdout


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("model_list: [{model_name: x\n", "while parsing a flow"),
        ("# only a comment\n", "Config cannot be None or Empty."),
        ("include: [dec1-absent.yaml]\n", "Included file not found:"),
        ("include: dec1-not-a-list.yaml\n", "'include' must be a list of file paths"),
        ("general_settings:\n  key_management_system: dec1_not_a_manager\n", "Invalid Key Management System selected"),
        (
            (
                "model_list:\n  - model_name: x\n    litellm_params:\n      model: openai/gpt-4o-mini\n"
                "      api_key: os.environ/oidc/env/DEC1_VALIDATE_ABSENT\n"
            ),
            "Environment variable DEC1_VALIDATE_ABSENT not found",
        ),
        ("router_settings:\n  routing_strategy: dec1-not-a-strategy\n", "dec1-not-a-strategy"),
        ("general_settings:\n  master_key: !Ref DEC1\n", "could not determine a constructor for the tag '!Ref'"),
    ],
    ids=[
        "invalid_yaml",
        "comment_only",
        "missing_include",
        "include_not_a_list",
        "unknown_secret_manager",
        "unreadable_oidc_reference",
        "unknown_routing_strategy",
        "custom_tag",
    ],
)
def test_a_config_error_exits_one_with_the_failure_prefix_and_the_cause(
    tmp_path: Path, text: str, message: str
) -> None:
    """Each config holds one error that stops a real boot: invalid YAML, a comment-only file, a missing
    or malformed include, an unknown `key_management_system`, an `oidc/env/` reference to an unset
    variable, an unknown routing strategy, a custom YAML tag. The command must exit 1 and stderr must
    contain `LiteLLM: config validation failed: ` and the cause. A pre-flight that exits 0 on any of
    these would let a broken config reach production."""
    config: Final = write_yaml_text(tmp_path, text)
    result: Final = validate_config(config, remove_environment=("DEC1_VALIDATE_ABSENT",))
    assert result.exit_code == 1, (result.stdout[-2000:], result.stderr[-4000:])
    assert VALIDATION_FAILED_PREFIX in result.stderr
    assert message in result.stderr, result.stderr[-4000:]


def test_a_missing_config_file_exits_one_and_names_the_path(tmp_path: Path) -> None:
    """`--config` names a file that does not exist. The command must exit 1 and stderr must contain
    `LiteLLM: config validation failed: Config file not found: <that path>` (`_get_config_from_file`)."""
    missing: Final = tmp_path / "dec1-no-such-config.yaml"
    result: Final = validate_config(missing)
    assert result.exit_code == 1
    assert f"{VALIDATION_FAILED_PREFIX}Config file not found: {missing}" in result.stderr, result.stderr[-4000:]


def test_validate_config_without_a_config_path_is_a_usage_error() -> None:
    """`--validate_config` with no `--config`. The command must exit 2 (click's usage error status) and
    stderr must contain `--validate_config requires --config <path>`."""
    result: Final = validate_config(None)
    assert result.exit_code == 2, result.stderr[-4000:]
    assert "--validate_config requires --config <path>" in result.stderr
