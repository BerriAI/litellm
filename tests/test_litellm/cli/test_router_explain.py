"""Tests for the ``litellm router-explain`` CLI subcommand."""

from __future__ import annotations

import io
import json
import os
import stat
from pathlib import Path

import pytest
from click.testing import CliRunner

from litellm.cli.router_explain import (
    Anomaly,
    ModelGroup,
    RouterExplanation,
    _KNOWN_ROUTING_STRATEGIES,
    _build_explanation,
    _parse_yaml_or_json,
    _provider_from_model,
    cli,
    explain_router,
)


VALID_YAML = """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: gpt-4o
    litellm_params:
      model: azure/gpt-4o
      api_key: os.environ/AZURE_API_KEY
      api_base: https://example.openai.azure.com
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-6
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-3-sonnet
      api_key: os.environ/AWS_ACCESS_KEY_ID
router_settings:
  routing_strategy: usage-based-routing-v2
  num_retries: 2
  timeout: 600
litellm_settings:
  model_access_groups:
    - premium
    - internal
general_settings:
  model_group_alias:
    gpt4: gpt-4o
    sonnet: claude-sonnet
"""


# ---------- pure helpers ----------


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai/gpt-4o", "openai"),
        ("anthropic/claude-sonnet-4-6", "anthropic"),
        ("bedrock/anthropic.claude-3-sonnet", "bedrock"),
        ("gpt-4o", None),
        ("", None),
        (None, None),
        (123, None),
    ],
)
def test_provider_from_model(model, expected):
    assert _provider_from_model(model) == expected


def test_parse_yaml_or_json_parses_yaml():
    parsed, err = _parse_yaml_or_json("a: 1\nb: 2\n", "config.yaml")
    assert err is None
    assert parsed == {"a": 1, "b": 2}


def test_parse_yaml_or_json_parses_json_by_extension():
    parsed, err = _parse_yaml_or_json('{"a": 1}', "config.json")
    assert err is None
    assert parsed == {"a": 1}


def test_parse_yaml_or_json_rejects_non_mapping_top_level():
    parsed, err = _parse_yaml_or_json("- 1\n- 2\n", "config.yaml")
    assert parsed is None
    assert err is not None
    assert "top-level value" in err


def test_parse_yaml_or_json_handles_empty_input():
    parsed, err = _parse_yaml_or_json("", "config.yaml")
    assert parsed == {}
    assert err is None


# ---------- explain_router (programmatic API) ----------


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_explain_router_passes_on_clean_yaml(tmp_path: Path):
    p = _write(tmp_path / "config.yaml", VALID_YAML)
    explanation = explain_router(str(p))
    assert isinstance(explanation, RouterExplanation)
    assert explanation.deployments == 4
    assert explanation.model_names == ("gpt-4o", "claude-sonnet")
    assert explanation.providers == ("anthropic", "azure", "bedrock", "openai")
    assert explanation.routing_strategy == "usage-based-routing-v2"
    assert explanation.num_retries == 2
    assert explanation.timeout == 600
    assert explanation.model_access_groups == ("premium", "internal")
    assert explanation.model_group_aliases == {"gpt4": "gpt-4o", "sonnet": "claude-sonnet"}
    assert explanation.has_anomalies is False
    # Two model groups, with the correct deployment counts.
    by_name = {g.name: g for g in explanation.model_groups}
    assert by_name["gpt-4o"].deployment_count == 2
    assert by_name["gpt-4o"].providers == ("azure", "openai")
    assert by_name["claude-sonnet"].deployment_count == 2
    assert by_name["claude-sonnet"].providers == ("anthropic", "bedrock")


def test_explain_router_detects_single_deployment_group(tmp_path: Path):
    p = _write(
        tmp_path / "config.yaml",
        """
model_list:
  - model_name: solo
    litellm_params:
      model: openai/gpt-4o
""",
    )
    explanation = explain_router(str(p))
    assert explanation.has_anomalies is True
    single = next(a for a in explanation.anomalies if a.name == "single-deployment-group")
    assert single.severity == "warn"
    assert "'solo'" in single.details


def test_explain_router_detects_duplicate_litellm_params(tmp_path: Path):
    p = _write(
        tmp_path / "config.yaml",
        """
model_list:
  - model_name: gpt-4o-prod
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: gpt-4o-staging
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
""",
    )
    explanation = explain_router(str(p))
    dup = next(a for a in explanation.anomalies if a.name == "duplicate-litellm-params")
    assert dup.severity == "warn"
    assert "gpt-4o-prod" in dup.details
    assert "gpt-4o-staging" in dup.details


def test_explain_router_detects_unknown_routing_strategy(tmp_path: Path):
    p = _write(
        tmp_path / "config.yaml",
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
router_settings:
  routing_strategy: foo
""",
    )
    explanation = explain_router(str(p))
    bad = next(a for a in explanation.anomalies if a.name == "unknown-routing-strategy")
    assert bad.severity == "fail"
    assert "'foo'" in bad.details
    for known in _KNOWN_ROUTING_STRATEGIES:
        assert repr(known) in bad.details


def test_explain_router_detects_negative_router_settings(tmp_path: Path):
    p = _write(
        tmp_path / "config.yaml",
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
router_settings:
  num_retries: -1
  timeout: 0
  cooldown_time: -5
""",
    )
    explanation = explain_router(str(p))
    by_name = {a.name: a for a in explanation.anomalies}
    assert by_name["negative-num-retries"].severity == "fail"
    assert by_name["non-positive-timeout"].severity == "fail"
    assert by_name["negative-cooldown-time"].severity == "fail"


def test_explain_router_detects_empty_model_list(tmp_path: Path):
    p = _write(tmp_path / "config.yaml", "model_list: []\n")
    explanation = explain_router(str(p))
    empty = next(a for a in explanation.anomalies if a.name == "empty-model-list")
    assert empty.severity == "warn"
    assert explanation.deployments == 0
    assert explanation.model_names == ()


def test_explain_router_detects_orphan_alias(tmp_path: Path):
    p = _write(
        tmp_path / "config.yaml",
        """
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
general_settings:
  model_group_alias:
    gpt4: gpt-4o
    missing: does-not-exist
""",
    )
    explanation = explain_router(str(p))
    orphan = next(a for a in explanation.anomalies if a.name == "orphan-model-group-alias")
    assert orphan.severity == "warn"
    assert "'missing'" in orphan.details


def test_explain_router_handles_all_anomalies_at_once(tmp_path: Path):
    p = _write(
        tmp_path / "config.yaml",
        """
model_list:
  - model_name: solo
    litellm_params:
      model: openai/gpt-4o
router_settings:
  routing_strategy: typo-strategy
  num_retries: -3
general_settings:
  model_group_alias:
    bad: does-not-exist
""",
    )
    explanation = explain_router(str(p))
    names = {a.name for a in explanation.anomalies}
    assert names == {
        "single-deployment-group",
        "unknown-routing-strategy",
        "negative-num-retries",
        "orphan-model-group-alias",
    }
    assert all(a.severity in {"warn", "fail"} for a in explanation.anomalies)


def test_explain_router_reads_from_stdin(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(VALID_YAML))
    explanation = explain_router("-")
    assert explanation.deployments == 4
    assert explanation.has_anomalies is False


def test_explain_router_reports_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.yaml"
    explanation = explain_router(str(missing))
    assert explanation.deployments == 0
    assert len(explanation.anomalies) == 1
    assert explanation.anomalies[0].name == "file"
    assert explanation.anomalies[0].severity == "fail"


def test_explain_router_reports_unreadable_file(tmp_path: Path):
    p = _write(tmp_path / "config.yaml", VALID_YAML)
    os.chmod(p, 0)
    try:
        explanation = explain_router(str(p))
        assert explanation.deployments == 0
        assert explanation.anomalies[0].name == "file"
        # "Permission denied" on POSIX, may differ on Windows; just check the file anomaly.
    finally:
        os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


def test_explain_router_reports_parse_error(tmp_path: Path):
    p = _write(tmp_path / "config.yaml", "model_list: [unterminated\n")
    explanation = explain_router(str(p))
    assert explanation.deployments == 0
    parse = next(a for a in explanation.anomalies if a.name == "parse")
    assert parse.severity == "fail"


def test_explain_router_raises_value_error_for_empty_source():
    with pytest.raises(ValueError):
        explain_router("")


def test_explain_router_parses_json_config(tmp_path: Path):
    p = _write(
        tmp_path / "config.json",
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": "gpt-4o",
                        "litellm_params": {"model": "openai/gpt-4o"},
                    }
                ],
                "router_settings": {"routing_strategy": "simple-shuffle"},
            }
        ),
    )
    explanation = explain_router(str(p))
    assert explanation.deployments == 1
    assert explanation.routing_strategy == "simple-shuffle"
    assert explanation.providers == ("openai",)


# ---------- to_jsonable ----------


def test_to_jsonable_is_json_serializable(tmp_path: Path):
    p = _write(tmp_path / "config.yaml", VALID_YAML)
    explanation = explain_router(str(p))
    encoded = json.dumps(explanation.to_jsonable())
    decoded = json.loads(encoded)
    assert decoded["path"].endswith("config.yaml")
    assert decoded["deployments"] == 4
    assert decoded["providers"] == ["anthropic", "azure", "bedrock", "openai"]
    assert decoded["routing_strategy"] == "usage-based-routing-v2"
    assert decoded["model_group_aliases"] == {"gpt4": "gpt-4o", "sonnet": "claude-sonnet"}
    assert decoded["anomalies"] == []


# ---------- _build_explanation direct (sanity) ----------


def test_build_explanation_skips_malformed_entries():
    data = {
        "model_list": [
            {"model_name": "good", "litellm_params": {"model": "openai/gpt-4o"}},
            "not-a-mapping",
            {"model_name": "", "litellm_params": {"model": "openai/gpt-4o-mini"}},
            {"model_name": "no-params"},
            {"model_name": "non-string-params", "litellm_params": "not-a-mapping"},
        ]
    }
    explanation = _build_explanation("config.yaml", data)
    # `deployments` counts every entry in model_list; only well-formed
    # entries (mapping + non-empty model_name) contribute to model_names.
    assert explanation.deployments == 4
    assert explanation.model_names == ("good",)
    assert explanation.providers == ("openai",)
    # Two entries have no usable model_name; they are not in model_names but
    # still count toward deployments. Single-deployment-group is the only
    # anomaly that fires here (all groups have exactly 1 deployment).
    single = next(a for a in explanation.anomalies if a.name == "single-deployment-group")
    assert single.severity == "warn"


# ---------- CLI surface ----------


def test_cli_help_prints_usage():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "router-explain" in result.output
    assert "--config" in result.output
    assert "--json" in result.output


def test_cli_emits_table_on_valid_config(tmp_path: Path):
    p = _write(tmp_path / "config.yaml", VALID_YAML)
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p)])
    assert result.exit_code == 0
    assert "router-explain:" in result.output
    assert "Summary" in result.output
    assert "Model groups" in result.output
    assert "Anomalies: none" in result.output


def test_cli_emits_json_on_valid_config(tmp_path: Path):
    p = _write(tmp_path / "config.yaml", VALID_YAML)
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["deployments"] == 4
    assert payload["routing_strategy"] == "usage-based-routing-v2"
    assert payload["anomalies"] == []


def test_cli_exits_1_when_anomaly_present(tmp_path: Path):
    p = _write(
        tmp_path / "config.yaml",
        """
model_list:
  - model_name: solo
    litellm_params:
      model: openai/gpt-4o
""",
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p)])
    assert result.exit_code == 1
    assert "Anomalies (1)" in result.output


def test_cli_reads_from_stdin(monkeypatch, tmp_path: Path):
    stdin_path = tmp_path / "stdin.yaml"
    stdin_path.write_text(VALID_YAML, encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", "-"], input=stdin_path.read_text(encoding="utf-8"))
    assert result.exit_code == 0
    assert "router-explain: <stdin>" in result.output


def test_cli_reports_missing_file(tmp_path: Path):
    missing = tmp_path / "nope.yaml"
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(missing)])
    assert result.exit_code == 1
    assert "file" in result.output


def test_cli_reports_parse_error(tmp_path: Path):
    p = _write(tmp_path / "config.yaml", "model_list: [unterminated\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", str(p)])
    assert result.exit_code == 1
    assert "parse" in result.output


def test_cli_empty_config_value_raises_usage_error():
    runner = CliRunner()
    result = runner.invoke(cli, ["--config", ""])
    assert result.exit_code == 2  # click.UsageError
    assert "source is required" in result.output


# ---------- dataclass sanity ----------


def test_anomaly_is_frozen():
    a = Anomaly(name="x", severity="warn", details="y")
    with pytest.raises((AttributeError, Exception)):
        a.name = "z"  # type: ignore[misc]


def test_model_group_is_frozen():
    g = ModelGroup(name="g", deployment_count=2, providers=("openai",))
    with pytest.raises((AttributeError, Exception)):
        g.deployment_count = 3  # type: ignore[misc]


def test_router_explanation_has_anomalies_false_when_empty():
    e = RouterExplanation(
        path="<inline>",
        deployments=0,
        model_names=(),
        providers=(),
        routing_strategy=None,
        num_retries=None,
        timeout=None,
        cooldown_time=None,
        model_access_groups=(),
        model_group_aliases={},
        model_groups=(),
    )
    assert e.has_anomalies is False
