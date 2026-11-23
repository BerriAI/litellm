from pathlib import Path
from typing import Final

import pytest

from litellm.scripts.lint_config import format_finding, lint_file, main

VALID: Final = """\
model_list:
  - model_name: chat
    litellm_params:
      model: openai/gpt-5
      api_key: os.environ/OPENAI_API_KEY
      num_retries: 2
  - model_name: backup
    litellm_params:
      model: anthropic/claude-sonnet-4-5
litellm_settings:
  drop_params: true
  fallbacks: [{"chat": ["backup"]}]
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
router_settings:
  routing_strategy: simple-shuffle
"""
ENVIRON: Final = {"OPENAI_API_KEY": "sk-test", "LITELLM_MASTER_KEY": "sk-test"}


def _lint(tmp_path: Path, text: str, environ: dict[str, str] | None = None) -> list[str]:
    config_path: Final = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    return [
        format_finding(Path("config.yaml"), finding)
        for finding in lint_file(config_path, ENVIRON if environ is None else environ)
    ]


def test_valid_config_has_no_findings(tmp_path: Path) -> None:
    assert _lint(tmp_path, VALID) == []


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        (
            "model_name: backup",
            "model_nme: backup",
            [
                "config.yaml:7: error: model_list[1]: missing model_name",
                'config.yaml:7: warning: model_list[1].model_nme: unknown key, did you mean "model_name"?',
                'config.yaml:12: warning: litellm_settings.fallbacks[0]: model group "backup" is not in model_list',
            ],
        ),
        (
            "model: anthropic/claude-sonnet-4-5",
            "modle: anthropic/claude-sonnet-4-5",
            [
                "config.yaml:8: error: model_list[1].litellm_params: missing model",
                'config.yaml:9: warning: model_list[1].litellm_params.modle: unknown key, did you mean "model"?',
            ],
        ),
        (
            "drop_params: true",
            "drop_parms: true",
            ['config.yaml:11: warning: litellm_settings.drop_parms: unknown key, did you mean "drop_params"?'],
        ),
        (
            "general_settings:",
            "general_setings:",
            ['config.yaml:13: warning: general_setings: unknown key, did you mean "general_settings"?'],
        ),
        (
            "master_key: os.environ/LITELLM_MASTER_KEY",
            "master_kee: os.environ/LITELLM_MASTER_KEY",
            ['config.yaml:14: warning: general_settings.master_kee: unknown key, did you mean "master_key"?'],
        ),
        (
            '[{"chat": ["backup"]}]',
            '[{"chat": ["missing"]}]',
            ['config.yaml:12: warning: litellm_settings.fallbacks[0]: model group "missing" is not in model_list'],
        ),
        (
            "routing_strategy: simple-shuffle",
            "routing_strategy: simple-shufle",
            [
                "config.yaml:16: error: router_settings.routing_strategy: "
                'unknown strategy "simple-shufle", did you mean "simple-shuffle"?'
            ],
        ),
        (
            "  drop_params: true",
            "  drop_params: true\n  drop_params: false",
            ["config.yaml:12: error: litellm_settings.drop_params: duplicate key, only the last value is used"],
        ),
    ],
)
def test_one_mistake_is_reported_at_its_line(tmp_path: Path, old: str, new: str, expected: list[str]) -> None:
    assert old in VALID
    assert _lint(tmp_path, VALID.replace(old, new)) == expected


def test_malformed_yaml_is_one_error_with_a_line(tmp_path: Path) -> None:
    findings: Final = _lint(tmp_path, VALID.replace("litellm_settings:", "model_list: [\nlitellm_settings:"))
    assert len(findings) == 1
    assert findings[0].startswith("config.yaml:12: error: <root>: invalid YAML: "), findings[0]


def test_unset_environment_variable_is_reported_unless_declared_in_config(tmp_path: Path) -> None:
    assert _lint(tmp_path, VALID, {"LITELLM_MASTER_KEY": "sk-test"}) == [
        "config.yaml:5: warning: model_list[0].litellm_params.api_key: environment variable OPENAI_API_KEY is not set"
    ]
    declared: Final = VALID + "environment_variables:\n  OPENAI_API_KEY: sk-test\n"
    assert _lint(tmp_path, declared, {"LITELLM_MASTER_KEY": "sk-test"}) == []


def test_fallback_to_model_defined_in_included_file_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text(
        "model_list:\n  - model_name: included\n    litellm_params:\n      model: openai/gpt-5\n", encoding="utf-8"
    )
    text: Final = VALID.replace('["backup"]', '["included"]') + "include:\n  - models.yaml\n"
    assert _lint(tmp_path, text) == []


def test_non_mapping_root_is_an_error(tmp_path: Path) -> None:
    assert _lint(tmp_path, "- just\n- a list\n") == ["config.yaml: error: <root>: config must be a YAML mapping"]


@pytest.mark.parametrize(
    ("text", "flags", "exit_code"),
    [
        (VALID, [], 0),
        (VALID.replace("drop_params", "drop_parms"), [], 0),
        (VALID.replace("drop_params", "drop_parms"), ["--strict"], 1),
        (VALID.replace("simple-shuffle", "nope"), [], 1),
    ],
)
def test_exit_code_fails_on_errors_and_on_warnings_only_when_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str, flags: list[str], exit_code: int
) -> None:
    for name, value in ENVIRON.items():
        monkeypatch.setenv(name, value)
    config_path: Final = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    assert main([str(config_path), *flags]) == exit_code


def test_litellm_setting_under_general_settings_is_reported(tmp_path: Path) -> None:
    text: Final = VALID.replace("general_settings:\n", "general_settings:\n  drop_params: true\n")
    assert _lint(tmp_path, text) == [
        "config.yaml:14: warning: general_settings.drop_params: "
        "not read from general_settings, move it to litellm_settings"
    ]


def test_fallback_targets_that_are_not_a_list_are_an_error(tmp_path: Path) -> None:
    assert _lint(tmp_path, VALID.replace('["backup"]', "backup")) == [
        'config.yaml:12: error: litellm_settings.fallbacks[0]: fallbacks for "chat" must be a list of model groups, '
        "like {chat: [backup]}"
    ]


def test_mistake_in_included_file_is_reported_at_the_include_line(tmp_path: Path) -> None:
    (tmp_path / "settings.yaml").write_text("router_settings:\n  num_retrys: 2\n", encoding="utf-8")
    text: Final = VALID.replace(
        "router_settings:\n  routing_strategy: simple-shuffle\n", "include:\n  - settings.yaml\n"
    )
    assert _lint(tmp_path, text) == [
        'config.yaml:15: warning: router_settings.num_retrys: unknown key, did you mean "num_retries"? '
        "(in an included file)"
    ]


def test_duplicate_key_in_included_file_is_reported_beside_its_other_mistakes(tmp_path: Path) -> None:
    included: Final = tmp_path / "settings.yaml"
    included.write_text(
        "litellm_settings:\n  set_verbose: true\n  set_verbose: false\n  drop_parms: true\n", encoding="utf-8"
    )
    assert _lint(tmp_path, VALID + "include:\n  - settings.yaml\n") == [
        f"config.yaml:17: error: include: {included}:3: duplicate key litellm_settings.set_verbose, "
        "only the last value is used",
        'config.yaml:17: warning: litellm_settings.drop_parms: unknown key, did you mean "drop_params"? '
        "(in an included file)",
    ]
