"""Tests for the ``litellm doctor`` CLI subcommand."""

from __future__ import annotations

import json
import sys

from click.testing import CliRunner

from litellm.cli import doctor
from litellm.cli.doctor import cli, run_all_checks, _exit_code


def _names(results):
    return [r.name for r in results]


def test_run_all_checks_returns_six_results_in_registration_order():
    results = run_all_checks()
    assert _names(results) == [
        "python",
        "litellm",
        "env",
        "api-keys",
        "model-costs",
        "token-counter",
    ]


def test_exit_code_all_pass():
    results = run_all_checks()
    results = [r for r in results if r.status == "pass"]
    assert _exit_code(results) == 0


def test_exit_code_warn_only():
    fake = doctor.CheckResult("x", "warn", "y")
    assert _exit_code([fake]) == 2


def test_exit_code_fail():
    fake = doctor.CheckResult("x", "fail", "y")
    assert _exit_code([fake]) == 1


def test_cli_runs_and_prints_table():
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert "litellm doctor" in result.output
    assert "check" in result.output
    assert "status" in result.output
    assert "details" in result.output
    assert "token-counter" in result.output
    assert result.exit_code in (0, 1, 2)


def test_cli_json_output_is_valid_json():
    runner = CliRunner()
    result = runner.invoke(cli, ["--json"])
    payload = json.loads(result.output.strip())
    assert isinstance(payload, list)
    for entry in payload:
        assert set(entry.keys()) == {"name", "status", "details"}
    assert result.exit_code in (0, 1, 2)


def test_api_keys_check_lists_set_vars_only_by_name(monkeypatch):
    for name in doctor._KNOWN_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-shhh-dont-print-me")
    runner = CliRunner()
    result = runner.invoke(cli, ["--json"])
    assert "sk-shhh-dont-print-me" not in result.output, "doctor printed a secret value to stdout"
    payload = json.loads(result.output.strip())
    api_keys = next(entry for entry in payload if entry["name"] == "api-keys")
    assert "OPENAI_API_KEY" in api_keys["details"]
    assert api_keys["status"] == "pass"
    assert result.exit_code in (0, 1, 2)


def test_api_keys_check_warns_when_nothing_set(monkeypatch):
    for name in doctor._KNOWN_KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    result = doctor._check_api_keys()
    assert result.status == "warn"
    assert "no known provider API key env vars are set" in result.details


def test_python_check_passes_on_supported_version():
    result = doctor._check_python()
    assert result.status == "pass"
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in result.details


def test_litellm_check_passes_when_importable():
    result = doctor._check_litellm()
    assert result.status == "pass"
    assert "version" in result.details


def test_env_check_warns_when_no_env_file_present(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    result = doctor._check_env_file()
    assert result.status == "warn"
    assert "no .env file" in result.details


def test_env_check_passes_when_env_file_present(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-test\n")
    monkeypatch.chdir(tmp_path)
    result = doctor._check_env_file()
    assert result.status == "pass"
    assert str(env_file) in result.details


def test_model_costs_check_passes():
    result = doctor._check_model_costs()
    assert result.status == "pass"
    assert "models loaded" in result.details
    assert "local copy" in result.details


def test_model_costs_check_fails_when_loader_raises(monkeypatch):
    def _boom():
        raise RuntimeError("simulated loader failure")

    monkeypatch.setattr(
        "litellm.litellm_core_utils.get_model_cost_map.GetModelCostMap.load_local_model_cost_map",
        staticmethod(_boom),
    )
    result = doctor._check_model_costs()
    assert result.status == "fail"
    assert "simulated loader failure" in result.details


def test_token_counter_check_passes():
    result = doctor._check_token_counter()
    assert result.status == "pass"
    assert "tokens" in result.details


def test_token_counter_check_fails_when_token_counter_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("simulated token counter failure")

    monkeypatch.setattr("litellm.token_counter", _boom)
    result = doctor._check_token_counter()
    assert result.status == "fail"
    assert "simulated token counter failure" in result.details


def test_token_counter_check_fails_on_zero_or_negative(monkeypatch):
    monkeypatch.setattr("litellm.token_counter", lambda *a, **k: 0)
    result = doctor._check_token_counter()
    assert result.status == "fail"


def test_cli_exit_code_propagates_fail(monkeypatch):
    def _boom():
        raise RuntimeError("simulated loader failure")

    monkeypatch.setattr(
        "litellm.litellm_core_utils.get_model_cost_map.GetModelCostMap.load_local_model_cost_map",
        staticmethod(_boom),
    )
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 1
