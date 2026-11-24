import os
from pathlib import Path
from typing import Final, cast

import pytest

import litellm
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
      api_key: os.environ/ANTHROPIC_API_KEY
    model_info:
      max_output_tokens: 8192
litellm_settings:
  drop_params: true
  fallbacks: [{"chat": ["backup"]}]
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
router_settings:
  routing_strategy: simple-shuffle
"""
ENVIRON: Final = {"OPENAI_API_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-test", "LITELLM_MASTER_KEY": "sk-test"}


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
                'config.yaml:15: warning: litellm_settings.fallbacks[0]: model group "backup" is not in model_list',
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
            ['config.yaml:14: warning: litellm_settings.drop_parms: unknown key, did you mean "drop_params"?'],
        ),
        (
            "general_settings:",
            "general_setings:",
            ['config.yaml:16: warning: general_setings: unknown key, did you mean "general_settings"?'],
        ),
        (
            "master_key: os.environ/LITELLM_MASTER_KEY",
            "master_kee: os.environ/LITELLM_MASTER_KEY",
            ['config.yaml:17: warning: general_settings.master_kee: unknown key, did you mean "master_key"?'],
        ),
        (
            '[{"chat": ["backup"]}]',
            '[{"chat": ["missing"]}]',
            ['config.yaml:15: warning: litellm_settings.fallbacks[0]: model group "missing" is not in model_list'],
        ),
        (
            "routing_strategy: simple-shuffle",
            "routing_strategy: simple-shufle",
            [
                "config.yaml:19: error: router_settings.routing_strategy: "
                'unknown strategy "simple-shufle", did you mean "simple-shuffle"?'
            ],
        ),
        (
            "  drop_params: true",
            "  drop_params: true\n  drop_params: false",
            ["config.yaml:15: error: litellm_settings.drop_params: duplicate key, only the last value is used"],
        ),
    ],
)
def test_one_mistake_is_reported_at_its_line(tmp_path: Path, old: str, new: str, expected: list[str]) -> None:
    assert old in VALID
    assert _lint(tmp_path, VALID.replace(old, new)) == expected


def test_malformed_yaml_is_one_error_with_a_line(tmp_path: Path) -> None:
    findings: Final = _lint(tmp_path, VALID.replace("litellm_settings:", "model_list: [\nlitellm_settings:"))
    assert len(findings) == 1
    assert findings[0].startswith("config.yaml:15: error: <root>: invalid YAML: "), findings[0]


def test_unset_environment_variable_is_reported_unless_declared_in_config(tmp_path: Path) -> None:
    assert _lint(tmp_path, VALID, {"LITELLM_MASTER_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-test"}) == [
        "config.yaml:5: warning: model_list[0].litellm_params.api_key: environment variable OPENAI_API_KEY is not set"
    ]
    declared: Final = VALID + "environment_variables:\n  OPENAI_API_KEY: sk-test\n"
    assert _lint(tmp_path, declared, {"LITELLM_MASTER_KEY": "sk-test", "ANTHROPIC_API_KEY": "sk-test"}) == []


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
        "config.yaml:17: warning: general_settings.drop_params: "
        "not read from general_settings, move it to litellm_settings"
    ]


def test_fallback_targets_that_are_not_a_list_are_an_error(tmp_path: Path) -> None:
    assert _lint(tmp_path, VALID.replace('["backup"]', "backup")) == [
        'config.yaml:15: error: litellm_settings.fallbacks[0]: fallbacks for "chat" must be a list of model groups, '
        "like {chat: [backup]}"
    ]


def test_mistake_in_included_file_is_reported_at_the_include_line(tmp_path: Path) -> None:
    (tmp_path / "settings.yaml").write_text("router_settings:\n  num_retrys: 2\n", encoding="utf-8")
    text: Final = VALID.replace(
        "router_settings:\n  routing_strategy: simple-shuffle\n", "include:\n  - settings.yaml\n"
    )
    assert _lint(tmp_path, text) == [
        'config.yaml:18: warning: router_settings.num_retrys: unknown key, did you mean "num_retries"? '
        "(in an included file)"
    ]


def test_duplicate_key_in_included_file_is_reported_beside_its_other_mistakes(tmp_path: Path) -> None:
    included: Final = tmp_path / "settings.yaml"
    included.write_text(
        "litellm_settings:\n  set_verbose: true\n  set_verbose: false\n  drop_parms: true\n", encoding="utf-8"
    )
    assert _lint(tmp_path, VALID + "include:\n  - settings.yaml\n") == [
        f"config.yaml:20: error: include: {included}:3: duplicate key litellm_settings.set_verbose, "
        "only the last value is used",
        'config.yaml:20: warning: litellm_settings.drop_parms: unknown key, did you mean "drop_params"? '
        "(in an included file)",
    ]


def test_fallback_shape_is_checked_when_models_live_in_the_db(tmp_path: Path) -> None:
    text: Final = "litellm_settings:\n  fallbacks: chat\n  default_fallbacks: backup\n"
    assert _lint(tmp_path, text) == [
        "config.yaml:2: error: litellm_settings.fallbacks: must be a list like [{model_group: [fallback_group]}]",
        "config.yaml:3: error: litellm_settings.default_fallbacks: must be a list of model groups like [fallback_group]",
    ]


def test_fallbacks_to_db_models_are_not_reported_without_a_model_list(tmp_path: Path) -> None:
    text: Final = "router_settings:\n  fallbacks: [{chat: [backup]}]\n  default_fallbacks: [backup]\n"
    assert _lint(tmp_path, text) == []


def test_null_fallback_settings_are_treated_as_unset(tmp_path: Path) -> None:
    text: Final = "router_settings:\n  fallbacks: null\n  context_window_fallbacks:\n  default_fallbacks: null\n"
    assert _lint(tmp_path, text) == []


def test_mistake_in_included_section_that_replaces_a_root_section_is_attributed_to_the_include(tmp_path: Path) -> None:
    (tmp_path / "settings.yaml").write_text("router_settings:\n  routing_strategy: simple-shufle\n", encoding="utf-8")
    assert _lint(tmp_path, VALID + "include:\n  - settings.yaml\n") == [
        'config.yaml:20: error: router_settings.routing_strategy: unknown strategy "simple-shufle", '
        'did you mean "simple-shuffle"? (in an included file)'
    ]


def test_included_model_list_that_is_not_a_list_is_attributed_to_the_include(tmp_path: Path) -> None:
    (tmp_path / "models.yaml").write_text("model_list:\n  model_name: extra\n", encoding="utf-8")
    assert _lint(
        tmp_path, VALID.replace('  fallbacks: [{"chat": ["backup"]}]\n', "") + "include:\n  - models.yaml\n"
    ) == ["config.yaml:19: error: model_list: must be a list of model entries (in an included file)"]


def _model_entry(litellm_params: str, model_info: str = "") -> str:
    return f"model_list:\n  - model_name: chat\n    litellm_params: {litellm_params}\n" + (
        f"    model_info: {model_info}\n" if model_info else ""
    )


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [
        (
            "{max_output_tokns: 100}",
            [
                'config.yaml:4: warning: model_list[0].model_info.max_output_tokns: unknown key, did you mean "max_output_tokens"?'
            ],
        ),
        (
            "{input_cost_per_token: abc}",
            [
                "config.yaml:4: warning: model_list[0].model_info.input_cost_per_token: "
                "unexpected value: Input should be a valid number, unable to parse string as a number"
            ],
        ),
        (
            "{max_output_tokens: lots}",
            [
                "config.yaml:4: warning: model_list[0].model_info.max_output_tokens: "
                "unexpected value: expected a number, got 'lots'"
            ],
        ),
        (
            "{cache_creation_input_token_cost_above_1hr: -1}",
            [
                "config.yaml:4: warning: model_list[0].model_info.cache_creation_input_token_cost_above_1hr: "
                "unexpected value: must not be negative, got -1"
            ],
        ),
        (
            '{input_cost_per_token: "-1"}',
            [
                "config.yaml:4: warning: model_list[0].model_info.input_cost_per_token: "
                "unexpected value: must not be negative, got '-1'"
            ],
        ),
        ("[max_output_tokens]", ["config.yaml:4: error: model_list[0].model_info: must be a mapping"]),
        ("{my_team_label: blue, max_output_tokens: 100, input_cost_per_token: os.environ/OPENAI_API_KEY}", []),
    ],
)
def test_model_info_mistake_is_reported(tmp_path: Path, model_info: str, expected: list[str]) -> None:
    assert _lint(tmp_path, _model_entry("{model: openai/gpt-5, api_key: sk-test}", model_info)) == expected


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [
        (
            "",
            [
                "config.yaml:2: warning: model_list[0]: azure/my-deployment is not in LiteLLM's model cost map, "
                "so cost tracking and token limits need model_info.base_model or model_info.input_cost_per_token, "
                "model_info.output_cost_per_token, model_info.max_input_tokens, model_info.max_output_tokens"
            ],
        ),
        (
            "{input_cost_per_token: 0.1, max_input_tokens: 1000}",
            [
                "config.yaml:4: warning: model_list[0].model_info: azure/my-deployment is not in LiteLLM's model cost "
                "map, so cost tracking and token limits need model_info.base_model or "
                "model_info.output_cost_per_token, model_info.max_output_tokens"
            ],
        ),
        ("{base_model: azure/gpt-4o}", []),
        (
            "{input_cost_per_token: 0.1, output_cost_per_token: 0.2, max_input_tokens: 1000, max_output_tokens: 100}",
            [],
        ),
    ],
)
def test_model_missing_from_cost_map_needs_its_price_and_limits(
    tmp_path: Path, model_info: str, expected: list[str]
) -> None:
    params: Final = "{model: azure/my-deployment, api_key: k, api_base: https://example.com, api_version: v}"
    assert _lint(tmp_path, _model_entry(params, model_info)) == expected


def test_prices_under_litellm_params_count_for_a_model_missing_from_the_cost_map(tmp_path: Path) -> None:
    params: Final = (
        "{model: azure/my-deployment, api_key: k, api_base: https://example.com, api_version: v, "
        "input_cost_per_token: 0.1, output_cost_per_token: 0.2}"
    )
    assert _lint(tmp_path, _model_entry(params, "{max_input_tokens: 1000, max_output_tokens: 100}")) == []


def test_token_limits_under_litellm_params_do_not_count_for_a_model_missing_from_the_cost_map(tmp_path: Path) -> None:
    params: Final = (
        "{model: azure/my-deployment, api_key: k, api_base: https://example.com, api_version: v, "
        "max_input_tokens: 1000, max_output_tokens: 100}"
    )
    assert _lint(tmp_path, _model_entry(params, "{input_cost_per_token: 0.1, output_cost_per_token: 0.2}")) == [
        "config.yaml:4: warning: model_list[0].model_info: azure/my-deployment is not in LiteLLM's model cost map, "
        "so cost tracking and token limits need model_info.base_model or "
        "model_info.max_input_tokens, model_info.max_output_tokens"
    ]


def test_wildcard_model_is_not_checked_against_the_cost_map(tmp_path: Path) -> None:
    assert _lint(tmp_path, _model_entry('{model: "openai/*"}')) == []


@pytest.mark.parametrize(
    ("litellm_params", "environ", "expected"),
    [
        (
            "{model: vertex_ai/gemini-2.5-pro, vertex_project: demo}",
            {},
            [
                "config.yaml:3: warning: model_list[0].litellm_params: vertex_ai/gemini-2.5-pro needs "
                "VERTEXAI_LOCATION in the environment or litellm_params.vertex_location"
            ],
        ),
        ("{model: vertex_ai/gemini-2.5-pro, vertex_project: demo, vertex_location: us-central1}", {}, []),
        (
            "{model: vertex_ai/gemini-2.5-pro}",
            {"VERTEXAI_PROJECT": "demo", "VERTEXAI_LOCATION": "us-central1"},
            [],
        ),
        (
            "{model: azure/gpt-4o, api_base: https://example.com, api_version: v}",
            {},
            [
                "config.yaml:3: warning: model_list[0].litellm_params: azure/gpt-4o needs "
                "AZURE_API_KEY in the environment or litellm_params.api_key"
            ],
        ),
        ("{model: azure/gpt-4o, api_base: https://example.com, api_version: v, azure_ad_token: t}", {}, []),
        (
            "{model: bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0, aws_access_key_id: id}",
            {},
            [
                "config.yaml:3: warning: model_list[0].litellm_params: "
                "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0 needs "
                "AWS_SECRET_ACCESS_KEY in the environment or litellm_params.aws_secret_access_key"
            ],
        ),
        ("{model: bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0, aws_role_name: role}", {}, []),
        (
            "{model: bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0, aws_session_token: t}",
            {},
            [
                "config.yaml:3: warning: model_list[0].litellm_params: "
                "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0 needs "
                "AWS_ACCESS_KEY_ID in the environment or litellm_params.aws_access_key_id",
                "config.yaml:3: warning: model_list[0].litellm_params: "
                "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0 needs "
                "AWS_SECRET_ACCESS_KEY in the environment or litellm_params.aws_secret_access_key",
            ],
        ),
        ("{model: vertex_ai/gemini-2.5-pro, vertex_location: us-central1}", {"VERTEXAI_PROJECT": "demo"}, []),
    ],
)
def test_provider_setting_missing_from_params_and_environment_is_reported(
    tmp_path: Path, litellm_params: str, environ: dict[str, str], expected: list[str]
) -> None:
    assert _lint(tmp_path, _model_entry(litellm_params), environ) == expected


def test_provider_key_declared_under_environment_variables_is_accepted(tmp_path: Path) -> None:
    text: Final = _model_entry("{model: openai/gpt-5}") + "environment_variables:\n  OPENAI_API_KEY: sk-test\n"
    assert _lint(tmp_path, text, {}) == []


@pytest.mark.parametrize(
    ("litellm_params", "expected"),
    [
        (
            "{model: gpt-4o, custom_llm_provider: azure, api_base: https://example.com, api_version: v}",
            [
                "config.yaml:3: warning: model_list[0].litellm_params: azure/gpt-4o needs "
                "AZURE_API_KEY in the environment or litellm_params.api_key"
            ],
        ),
        ("{model: azure/gpt-4o, litellm_credential_name: shared}", []),
        (
            "{model: azure/gpt-4o, api_base: https://example.com, api_version: v, "
            "tenant_id: t, client_id: c, client_secret: s}",
            [],
        ),
    ],
)
def test_provider_field_and_stored_credentials_decide_what_a_model_needs(
    tmp_path: Path, litellm_params: str, expected: list[str]
) -> None:
    assert _lint(tmp_path, _model_entry(litellm_params), {}) == expected


def test_azure_client_secret_needs_its_tenant_and_client_from_settings_or_the_environment(tmp_path: Path) -> None:
    text: Final = _model_entry("{model: azure/gpt-4o, api_base: https://example.com, api_version: v, client_secret: s}")
    assert _lint(tmp_path, text, {"AZURE_TENANT_ID": "t", "AZURE_CLIENT_ID": "c"}) == []
    assert _lint(tmp_path, text, {"AZURE_TENANT_ID": "t"}) == [
        "config.yaml:3: warning: model_list[0].litellm_params: azure/gpt-4o needs "
        "AZURE_API_KEY in the environment or litellm_params.api_key"
    ]


def test_declared_environment_variable_can_reference_an_earlier_declared_one(tmp_path: Path) -> None:
    text: Final = _model_entry("{model: openai/gpt-5}") + (
        "environment_variables:\n  SHARED_KEY: sk-test\n  OPENAI_API_KEY: os.environ/SHARED_KEY\n"
    )
    assert _lint(tmp_path, text, {}) == []


def test_declared_environment_variable_wins_over_the_process_environment(tmp_path: Path) -> None:
    text: Final = _model_entry("{model: openai/gpt-5}") + "environment_variables:\n  OPENAI_API_KEY: sk-test\n"
    assert _lint(tmp_path, text, {"OPENAI_API_KEY": ""}) == []


def test_provider_key_value_declared_under_environment_variables_is_used(tmp_path: Path) -> None:
    text: Final = _model_entry("{model: gnani/prisma-v2.5}") + "environment_variables:\n  GNANI_API_KEY: test\n"
    assert _lint(tmp_path, text, {}) == []


@pytest.mark.parametrize("model", ["chatgpt/gpt-5", "github_copilot/gpt-4"])
def test_linting_never_resolves_a_provider_that_logs_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("lint resolved a provider that starts a login")

    monkeypatch.setattr("litellm.utils.litellm_utils.get_llm_provider", fail)
    full: Final = "{input_cost_per_token: 0, output_cost_per_token: 0, max_input_tokens: 1, max_output_tokens: 1}"
    assert _lint(tmp_path, _model_entry(f"{{model: {model}}}", full), {}) == []


def test_linting_leaves_the_process_environment_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LINT_CONFIG_SENTINEL", "kept")
    monkeypatch.delenv("LITELLM_MASTER_KEY", raising=False)
    _lint(tmp_path, VALID)
    assert os.environ["LINT_CONFIG_SENTINEL"] == "kept"
    assert "LITELLM_MASTER_KEY" not in os.environ


def test_declared_environment_variable_pointing_at_an_unset_one_stays_unset(
    tmp_path: Path,
) -> None:
    text: Final = _model_entry("{model: openai/gpt-5}") + (
        "environment_variables:\n  OPENAI_API_KEY: os.environ/LATER_KEY\n  LATER_KEY: sk-test\n"
    )
    needs: Final = [
        "config.yaml:3: warning: model_list[0].litellm_params: openai/gpt-5 needs "
        "OPENAI_API_KEY in the environment or litellm_params.api_key"
    ]
    assert _lint(tmp_path, text, {}) == needs
    assert _lint(tmp_path, text, {"OPENAI_API_KEY": "sk-process"}) == []


_OPUS_MAP_ENTRY: Final = {
    "litellm_provider": "anthropic",
    "mode": "chat",
    "max_input_tokens": 200000,
    "max_output_tokens": 64000,
    "input_cost_per_token": 5e-06,
    "output_cost_per_token": 2.5e-05,
    "cache_creation_input_token_cost": 6.25e-06,
    "cache_creation_input_token_cost_above_1hr": 1e-05,
    "cache_read_input_token_cost": 5e-07,
    "input_cost_per_token_above_200k_tokens": 1e-05,
    "search_context_cost_per_query": {"search_context_size_low": 0.01},
    "output_cost_per_reasoning_token": None,
}
_GPT_MAP_ENTRY: Final = {
    "litellm_provider": "openai",
    "mode": "chat",
    "max_input_tokens": 128000,
    "max_output_tokens": 16384,
    "input_cost_per_token": 2.5e-06,
    "output_cost_per_token": 1e-05,
    "cache_read_input_token_cost": 1.25e-06,
    "regional_processing_uplift_multiplier_eu": 1.1,
    "fedramp_uplift_multiplier": 1.1,
}
_OPUS_PRICES: Final = (
    "{input_cost_per_token: 1, output_cost_per_token: 1, cache_creation_input_token_cost: 1, "
    "cache_read_input_token_cost: 1, input_cost_per_token_above_200k_tokens: 1, "
    "search_context_cost_per_query: {search_context_size_low: 1}}"
)
_GPT_PRICES: Final = (
    "{input_cost_per_token: 1, output_cost_per_token: 1, cache_read_input_token_cost: 1, "
    "regional_processing_uplift_multiplier_eu: 1, fedramp_uplift_multiplier: 1}"
)
_OPUS_NEEDS_1HR: Final = [
    "config.yaml:4: error: model_list[0].model_info: anthropic/lint-opus would take these prices from "
    "LiteLLM's model cost map, set them in model_info: cache_creation_input_token_cost_above_1hr"
]


@pytest.fixture
def price_map(monkeypatch: pytest.MonkeyPatch) -> None:
    model_cost: Final = cast("dict[str, object]", litellm.model_cost)  # pyright: ignore[reportUnknownMemberType]
    monkeypatch.setitem(model_cost, "lint-opus", _OPUS_MAP_ENTRY)
    monkeypatch.setitem(model_cost, "lint-gpt", _GPT_MAP_ENTRY)


def _lint_prices(tmp_path: Path, text: str, *, require_explicit_prices: bool = True) -> list[str]:
    config_path: Final = tmp_path / "config.yaml"
    config_path.write_text(text, encoding="utf-8")
    return [
        format_finding(Path("config.yaml"), finding)
        for finding in lint_file(config_path, ENVIRON, require_explicit_prices=require_explicit_prices)
    ]


@pytest.mark.usefixtures("price_map")
def test_require_explicit_prices_flags_a_price_the_cost_map_has_for_the_model(tmp_path: Path) -> None:
    text: Final = _model_entry("{model: anthropic/lint-opus}", _OPUS_PRICES)
    assert _lint_prices(tmp_path, text) == _OPUS_NEEDS_1HR
    assert _lint_prices(tmp_path, text, require_explicit_prices=False) == []
    assert _lint(tmp_path, text) == []


@pytest.mark.usefixtures("price_map")
def test_require_explicit_prices_passes_once_every_mapped_price_is_set(tmp_path: Path) -> None:
    complete: Final = _OPUS_PRICES.replace("{input", "{cache_creation_input_token_cost_above_1hr: 1, input", 1)
    assert _lint_prices(tmp_path, _model_entry("{model: anthropic/lint-opus}", complete)) == []
    in_params: Final = "{model: anthropic/lint-opus, cache_creation_input_token_cost_above_1hr: 1}"
    assert _lint_prices(tmp_path, _model_entry(in_params, _OPUS_PRICES)) == []


@pytest.mark.usefixtures("price_map")
def test_require_explicit_prices_only_asks_for_prices_in_that_models_map_entry(tmp_path: Path) -> None:
    assert _lint_prices(tmp_path, _model_entry("{model: openai/lint-gpt}", _GPT_PRICES)) == []


@pytest.mark.usefixtures("price_map")
def test_require_explicit_prices_asks_for_billing_multipliers(tmp_path: Path) -> None:
    info: Final = "{input_cost_per_token: 1, output_cost_per_token: 1, cache_read_input_token_cost: 1}"
    assert _lint_prices(tmp_path, _model_entry("{model: openai/lint-gpt}", info)) == [
        "config.yaml:4: error: model_list[0].model_info: openai/lint-gpt would take these prices from LiteLLM's "
        "model cost map, set them in model_info: fedramp_uplift_multiplier, regional_processing_uplift_multiplier_eu"
    ]


@pytest.mark.usefixtures("price_map")
def test_require_explicit_prices_uses_base_model_and_counts_a_null_price_as_unset(tmp_path: Path) -> None:
    info: Final = _OPUS_PRICES.replace(
        "{input", "{base_model: anthropic/lint-opus, cache_creation_input_token_cost_above_1hr: null, input", 1
    )
    params: Final = "{model: azure/my-deployment, api_base: https://example.com, api_version: v, api_key: k}"
    assert _lint_prices(tmp_path, _model_entry(params, info)) == _OPUS_NEEDS_1HR


@pytest.mark.usefixtures("price_map")
def test_require_explicit_prices_lists_every_missing_price_when_model_info_is_absent(tmp_path: Path) -> None:
    assert _lint_prices(tmp_path, _model_entry("{model: openai/lint-gpt}")) == [
        "config.yaml:2: error: model_list[0]: openai/lint-gpt would take these prices from LiteLLM's model cost map, "
        "set them in model_info: cache_read_input_token_cost, fedramp_uplift_multiplier, input_cost_per_token, "
        "output_cost_per_token, regional_processing_uplift_multiplier_eu"
    ]


def test_require_explicit_prices_skips_wildcard_and_unmapped_models(tmp_path: Path) -> None:
    assert _lint_prices(tmp_path, _model_entry("{model: openai/*}")) == []
    unmapped: Final = _lint_prices(tmp_path, _model_entry("{model: openai/my-own-model}"))
    assert [line.split(": ")[1] for line in unmapped] == ["warning"]


@pytest.mark.usefixtures("price_map")
def test_require_explicit_prices_flag_alone_makes_the_exit_code_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in ENVIRON.items():
        monkeypatch.setenv(name, value)
    config_path: Final = tmp_path / "config.yaml"
    config_path.write_text(_model_entry("{model: anthropic/lint-opus}", _OPUS_PRICES), encoding="utf-8")
    assert main([str(config_path)]) == 0
    assert main([str(config_path), "--require-explicit-prices"]) == 1
