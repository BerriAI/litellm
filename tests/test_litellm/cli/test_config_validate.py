"""Tests for the ``litellm config-validate`` CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from litellm.cli.config_validate import (
    ConfigValidation,
    cli,
    validate_config,
)


VALID_YAML = """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY
router_settings:
  routing_strategy: usage-based-routing-v2
litellm_settings:
  drop_params: true
"""


# ---------- validate_config (programmatic API) ----------


def test_validate_config_passes_on_clean_yaml(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML, encoding="utf-8")
    result = validate_config(str(p))
    assert isinstance(result, ConfigValidation)
    assert result.has_failures is False
    assert {c.name for c in result.checks} >= {
        "file",
        "parse",
        "top-level-keys",
        "model-list-shape",
        "model-name-uniq",
        "litellm-params-model",
        "router-strategy",
        "cred-pattern",
    }


def test_validate_config_detects_duplicate_model_names(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: same
    litellm_params:
      model: openai/gpt-4o
  - model_name: same
    litellm_params:
      model: openai/gpt-4o-mini
""",
        encoding="utf-8",
    )
    result = validate_config(str(p))
    dup_check = next(c for c in result.checks if c.name == "model-name-uniq")
    assert dup_check.status == "fail"
    assert "same" in dup_check.details
    assert result.has_failures is True


def test_validate_config_detects_non_string_model_in_litellm_params(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: bad
    litellm_params:
      model:
        nested: object
  - model_name: empty
    litellm_params:
      model: ""
""",
        encoding="utf-8",
    )
    result = validate_config(str(p))
    model_check = next(c for c in result.checks if c.name == "litellm-params-model")
    assert model_check.status == "fail"
    assert "2 entries" in model_check.details


def test_validate_config_detects_unknown_top_level_key_as_warn(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
models_list: []
""",
        encoding="utf-8",
    )
    result = validate_config(str(p))
    top_check = next(c for c in result.checks if c.name == "top-level-keys")
    assert top_check.status == "warn"
    assert "models_list" in top_check.details
    assert result.has_failures is False


def test_validate_config_strict_promotes_warn_to_fail(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
models_list: []
""",
        encoding="utf-8",
    )
    result = validate_config(str(p), strict=True)
    top_check = next(c for c in result.checks if c.name == "top-level-keys")
    assert top_check.status == "fail"
    assert result.has_failures is True


def test_validate_config_detects_unknown_routing_strategy(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
router_settings:
  routing_strategy: not-a-real-strategy
""",
        encoding="utf-8",
    )
    result = validate_config(str(p))
    rs_check = next(c for c in result.checks if c.name == "router-strategy")
    assert rs_check.status == "fail"
    assert "not-a-real-strategy" in rs_check.details


def test_validate_config_warns_on_unset_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISSING_LITELLM_TEST_KEY", raising=False)
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/MISSING_LITELLM_TEST_KEY
""",
        encoding="utf-8",
    )
    result = validate_config(str(p))
    cred_check = next(c for c in result.checks if c.name == "cred-pattern")
    assert cred_check.status == "warn"
    assert "MISSING_LITELLM_TEST_KEY" in cred_check.details


def test_validate_config_passes_when_env_var_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_TEST_KEY_SET", "sk-fake")
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/LITELLM_TEST_KEY_SET
""",
        encoding="utf-8",
    )
    result = validate_config(str(p))
    cred_check = next(c for c in result.checks if c.name == "cred-pattern")
    assert cred_check.status == "pass"


def test_validate_config_missing_file_returns_fail():
    result = validate_config("/nonexistent/path/config.yaml")
    file_check = next(c for c in result.checks if c.name == "file")
    assert file_check.status == "fail"
    assert "does not exist" in file_check.details
    assert result.has_failures is True


def test_validate_config_malformed_yaml_returns_fail(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text("model_list:\n  - : invalid yaml here :\n    ::", encoding="utf-8")
    result = validate_config(str(p))
    parse_check = next(c for c in result.checks if c.name == "parse")
    assert parse_check.status == "fail"
    assert "invalid YAML" in parse_check.details


def test_validate_config_malformed_json_returns_fail(tmp_path: Path):
    p = tmp_path / "config.json"
    p.write_text("{not json", encoding="utf-8")
    result = validate_config(str(p))
    parse_check = next(c for c in result.checks if c.name == "parse")
    assert parse_check.status == "fail"
    assert "invalid JSON" in parse_check.details


def test_validate_config_top_level_must_be_mapping(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    result = validate_config(str(p))
    parse_check = next(c for c in result.checks if c.name == "parse")
    assert parse_check.status == "fail"
    assert "mapping" in parse_check.details


def test_validate_config_model_list_must_be_list(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text("model_list: not-a-list\n", encoding="utf-8")
    result = validate_config(str(p))
    shape_check = next(c for c in result.checks if c.name == "model-list-shape")
    assert shape_check.status == "fail"
    assert "list" in shape_check.details


def test_validate_config_empty_source_raises():
    with pytest.raises(ValueError) as exc:
        validate_config("")
    assert "required" in str(exc.value)


def test_validate_config_reads_from_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML, encoding="utf-8")
    text = p.read_text(encoding="utf-8")
    monkeypatch.setattr("sys.stdin", type("S", (), {"read": staticmethod(lambda: text)})())
    result = validate_config("-")
    file_check = next(c for c in result.checks if c.name == "file")
    assert "stdin" in file_check.details
    assert result.has_failures is False


def test_to_jsonable_round_trip(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML, encoding="utf-8")
    result = validate_config(str(p))
    payload = json.loads(json.dumps(result.to_jsonable()))
    assert payload["path"] == str(p)
    assert isinstance(payload["checks"], list)
    for entry in payload["checks"]:
        assert set(entry.keys()) == {"name", "status", "details"}


# ---------- cli (CLI surface) ----------


def test_cli_passes_on_clean_yaml(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML, encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p)])
    assert result.exit_code == 0, result.output
    assert "router-strategy" in result.output
    assert "usage-based-routing-v2" in result.output


def test_cli_json_output_is_valid_json(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(VALID_YAML, encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert isinstance(payload["checks"], list)
    assert {c["name"] for c in payload["checks"]} >= {"file", "router-strategy"}


def test_cli_missing_config_exits_2():
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", "/no/such/file.yaml"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_cli_strict_promotes_warn_to_fail(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
models_list: []
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p)])
    assert result.exit_code == 0
    assert "warn" in result.output

    result_strict = runner.invoke(cli, ["--config", str(p), "--strict"])
    assert result_strict.exit_code == 1
    assert "fail" in result_strict.output


def test_cli_unknown_routing_strategy_exits_1(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
router_settings:
  routing_strategy: typo-strategy
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p)])
    assert result.exit_code == 1
    assert "typo-strategy" in result.output


def test_cli_duplicate_model_name_exits_1(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text(
        """
model_list:
  - model_name: dup
    litellm_params:
      model: openai/gpt-4o
  - model_name: dup
    litellm_params:
      model: openai/gpt-4o-mini
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p)])
    assert result.exit_code == 1
    assert "duplicate" in result.output.lower()


def test_cli_malformed_yaml_exits_1(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text("model_list:\n  - : invalid\n    ::\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p)])
    assert result.exit_code == 1
    assert "invalid YAML" in result.output
