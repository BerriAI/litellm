"""How the bytes of config.yaml become values: PyYAML `safe_load` semantics, pinned end to end.

The Python proxy reads its config with `yaml.safe_load` (YAML 1.1 rules). The Rust gateway now reads the
same file through `litellm-yaml`, a hand port of `safe_load` (#52023, `litellm/rust/crates/yaml`), whose
parity table (`tests/pyyaml.rs`) lists the cases below. These tests pin the same cases through a running
proxy, so a value that changes type in either loader shows up as a different response.

The probe is `model_info`: free-form keys there are echoed unchanged by `GET /model/info`, so a scalar
written as `0755` must come back as the JSON number 493, `yes` as `true`, and `09` as the string "09".
Behavioural probes cover the cases where the type decides what the proxy does (a YAML 1.1 `yes` turning a
boolean setting on, a quoted "yes" leaving it off, anchors and merge keys sharing provider params, a
repeated key keeping its last value). Malformed documents must stop the boot with PyYAML's message.

Every config here is written as raw text and declares `router_settings.num_retries`, because the harness
rewrites (and so re-serialises) a config without it (`private_configuration` in `_support/process.py`).
With `INTEGRATION_WORKER_ISOLATION=1` the harness re-serialises every config; these tests then only prove
the Python reading, which is the first thing to check if one fails only in that mode.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, gateway_from_environment, object_value
from tests.integration._support.process import OwnedProxy
from tests.integration.configuration.dec1_cfgload_support import refused_with_exit, write_yaml_text
from tests.integration.configuration.dec1_config_support import (
    UPSTREAM_KEY,
    booted,
    chat,
    model_entries,
    scripted_upstream,
    sent,
    served_model_names,
)

_PROBE: Final = "dec1-cfgload-yaml-probe"
_TAIL: Final = "router_settings:\n  num_retries: 0\n"

_PROBE_CONFIG: Final = f"""\
model_list:
  - model_name: {_PROBE}
    litellm_params:
      model: openai/gpt-4o-mini
      api_base: http://127.0.0.1:9/v1
      api_key: {UPSTREAM_KEY}
    model_info:
      v_yes: yes
      v_capital_yes: Yes
      v_on: on
      v_off: Off
      v_upper_no: NO
      v_y: y
      v_n: n
      v_mixed_case_true: tRUE
      v_octal: 0755
      v_zero_o: 0o17
      v_hex: 0x1F
      v_binary: 0b101
      v_underscores: 1_000
      v_plus: +1
      v_base_60: 1:30
      v_base_60_float: 1:30.5
      v_base_60_bad_group: 1:60
      v_leading_zero_nine: 09
      v_exponent_without_dot: 1e3
      v_exponent_without_sign: 1.0e3
      v_signed_exponent: 1.0e+3
      v_trailing_dot: 1.
      v_leading_dot: .5
      v_signed_leading_dot: -.5
      v_float_underscores: 1.5_0
      v_api_version: 2024-02-15-preview
      v_date_short_parts: 2026-1-5
      v_single_quoted_yes: 'yes'
      v_double_quoted_octal: "0755"
      v_str_tag: !!str 0755
      v_int_tag_on_quoted: !!int '12'
      v_float_tag_on_int: !!float 1
      v_bool_tag: !!bool yes
      v_non_specific_tag: ! 123
      v_verbatim_str_tag: !<tag:yaml.org,2002:str> 1
      v_folded: >-
        folded
        text
      v_literal: |
        line
        next
      v_flow_pair_in_sequence: [b: c]
      v_escapes: "\\/\\t"
      v_doubled_quote: 'it''s'
{_TAIL}"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("dec1-cfgload-yaml")
    with (
        gateway_from_environment() as shared,
        booted(shared, directory, write_yaml_text(directory, _PROBE_CONFIG)) as owned,
    ):
        yield owned


def _info(owned: OwnedProxy) -> dict[str, JsonValue]:
    return object_value(model_entries(owned.gateway, _PROBE)[0]["model_info"])


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("v_yes", True),
        ("v_capital_yes", True),
        ("v_on", True),
        ("v_off", False),
        ("v_upper_no", False),
        ("v_y", "y"),
        ("v_n", "n"),
        ("v_mixed_case_true", "tRUE"),
        ("v_octal", 493),
        ("v_zero_o", "0o17"),
        ("v_hex", 31),
        ("v_binary", 5),
        ("v_underscores", 1000),
        ("v_plus", 1),
        ("v_base_60", 90),
        ("v_base_60_float", 90.5),
        ("v_base_60_bad_group", "1:60"),
        ("v_leading_zero_nine", "09"),
        ("v_exponent_without_dot", "1e3"),
        ("v_exponent_without_sign", "1.0e3"),
        ("v_signed_exponent", 1000.0),
        ("v_trailing_dot", 1.0),
        ("v_leading_dot", 0.5),
        ("v_signed_leading_dot", "-.5"),
        ("v_float_underscores", 1.5),
        ("v_api_version", "2024-02-15-preview"),
        ("v_date_short_parts", "2026-1-5"),
        ("v_single_quoted_yes", "yes"),
        ("v_double_quoted_octal", "0755"),
        ("v_str_tag", "0755"),
        ("v_int_tag_on_quoted", 12),
        ("v_float_tag_on_int", 1.0),
        ("v_bool_tag", True),
        ("v_non_specific_tag", 123),
        ("v_verbatim_str_tag", "1"),
        ("v_folded", "folded text"),
        ("v_literal", "line\nnext\n"),
        ("v_flow_pair_in_sequence", [{"b": "c"}]),
        ("v_escapes", "/\t"),
        ("v_doubled_quote", "it's"),
    ],
)
def test_a_plain_scalar_in_the_config_resolves_to_the_type_pyyaml_safe_load_gives_it(
    probe: OwnedProxy, key: str, expected: JsonValue
) -> None:
    """One proxy boots from a raw YAML file whose `model_info` holds one scalar per YAML 1.1 rule. `GET
    /model/info` must echo each value with the value and JSON type PyYAML 6 `safe_load` gives it: `yes`,
    `on` and `Yes` are true, `y` and `tRUE` stay text, `0755` is octal 493, `1:30` is base 60 (90), `1e3`
    without a dot stays text, quoting or `!!str` keeps digits as text. The expected values are the
    `loads_like_pyyaml_safe_load` table in `litellm/rust/crates/yaml/tests/pyyaml.rs`, so a Rust loader
    that drifts from PyYAML changes a config's meaning silently."""
    actual: Final = _info(probe)[key]
    assert actual == expected
    assert type(actual) is type(expected), (key, actual)


@pytest.mark.parametrize("spelling", ["yes", "Yes", "on", "ON", "True"])
def test_a_yaml_1_1_boolean_spelling_turns_a_boolean_general_setting_on(
    gateway: Gateway, tmp_path: Path, spelling: str
) -> None:
    """`general_settings.allow_public_health_readiness_details` acts only when it is the boolean `True`
    (`_health_endpoints.py:2257`, `is True`). Written as the unquoted YAML 1.1 spelling `yes`, `on` or
    `True`, the public `GET /health/readiness` (no key) must include `litellm_version`: those spellings
    are booleans, not strings, by the time settings are read."""
    text: Final = f"general_settings:\n  allow_public_health_readiness_details: {spelling}\n{_TAIL}"
    with booted(gateway, tmp_path, write_yaml_text(tmp_path, text)) as owned:
        body: Final = owned.gateway.client.get("/health/readiness").json()
    assert "litellm_version" in body, body


@pytest.mark.parametrize("spelling", ["'yes'", '"true"', "y", "1"])
def test_a_quoted_or_non_yaml_boolean_spelling_leaves_a_strict_boolean_setting_off(
    gateway: Gateway, tmp_path: Path, spelling: str
) -> None:
    """The same strict setting written as a quoted `'yes'`, a quoted `"true"`, a bare `y` or `1` is a
    string or an integer, not `True`, so the public `GET /health/readiness` must stay the two-field
    probe without `litellm_version`. A loader that coerced these to booleans would expose version
    details to unauthenticated callers."""
    text: Final = f"general_settings:\n  allow_public_health_readiness_details: {spelling}\n{_TAIL}"
    with booted(gateway, tmp_path, write_yaml_text(tmp_path, text)) as owned:
        body: Final = owned.gateway.client.get("/health/readiness").json()
    assert "litellm_version" not in body, body


def test_an_anchor_and_alias_share_provider_params_between_two_deployments(gateway: Gateway, tmp_path: Path) -> None:
    """The first entry's `litellm_params` is anchored `&shared` and the second entry's `litellm_params`
    is the alias `*shared`. A chat completion to each model name must reach the scripted upstream with
    `Authorization: Bearer <the anchored api_key>`: an alias is a full copy of the anchored mapping."""
    with scripted_upstream() as upstream:
        text: Final = f"""\
model_list:
  - model_name: dec1-anchor-a
    litellm_params: &shared
      model: openai/gpt-4o-mini
      api_base: {upstream.url}/v1
      api_key: sk-dec1-anchored
  - model_name: dec1-anchor-b
    litellm_params: *shared
{_TAIL}"""
        with booted(gateway, tmp_path, write_yaml_text(tmp_path, text)) as owned:
            assert chat(owned.gateway, "dec1-anchor-a").status_code == 200
            assert chat(owned.gateway, "dec1-anchor-b").status_code == 200
            requests: Final = sent(upstream)
    assert [request.headers["authorization"] for request in requests] == ["Bearer sk-dec1-anchored"] * 2


def test_a_merge_key_copies_params_and_a_key_of_the_entrys_own_wins(gateway: Gateway, tmp_path: Path) -> None:
    """A top-level `x-dec1-base` anchor holds `model`, `api_base` and `api_key`; the deployment's
    `litellm_params` is `{<<: *base, api_key: sk-dec1-own}`. The upstream must receive `Bearer
    sk-dec1-own` at the anchored `api_base`: `<<` merges the anchored keys and a key the mapping sets
    itself wins over the merged one (PyYAML `flatten_mapping`). The unknown top-level `x-dec1-base` key
    is ignored."""
    with scripted_upstream() as upstream:
        text: Final = f"""\
x-dec1-base: &base
  model: openai/gpt-4o-mini
  api_base: {upstream.url}/v1
  api_key: sk-dec1-merged
model_list:
  - model_name: dec1-merge
    litellm_params:
      <<: *base
      api_key: sk-dec1-own
{_TAIL}"""
        with booted(gateway, tmp_path, write_yaml_text(tmp_path, text)) as owned:
            response: Final = chat(owned.gateway, "dec1-merge")
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [(request.target, request.headers["authorization"]) for request in requests] == [
        ("/v1/chat/completions", "Bearer sk-dec1-own")
    ]


def test_a_repeated_top_level_key_keeps_only_its_last_value(gateway: Gateway, tmp_path: Path) -> None:
    """The file declares `model_list` twice, first with `dec1-first` and later with `dec1-second`.
    `GET /v1/models` must list `dec1-second` and not `dec1-first`: a repeated mapping key is not an
    error, the last value replaces the earlier one."""
    text: Final = f"""\
model_list:
  - model_name: dec1-first
    litellm_params: {{model: openai/gpt-4o-mini, api_base: "http://127.0.0.1:9/v1", api_key: {UPSTREAM_KEY}}}
model_list:
  - model_name: dec1-second
    litellm_params: {{model: openai/gpt-4o-mini, api_base: "http://127.0.0.1:9/v1", api_key: {UPSTREAM_KEY}}}
{_TAIL}"""
    with booted(gateway, tmp_path, write_yaml_text(tmp_path, text)) as owned:
        names: Final = served_model_names(owned.gateway)
    assert "dec1-second" in names
    assert "dec1-first" not in names


def test_a_repeated_key_inside_litellm_params_sends_the_last_value(gateway: Gateway, tmp_path: Path) -> None:
    """A deployment's `litellm_params` sets `api_key` twice, `sk-dec1-earlier` then `sk-dec1-later`. The
    upstream must receive `Bearer sk-dec1-later`."""
    with scripted_upstream() as upstream:
        text: Final = f"""\
model_list:
  - model_name: dec1-repeat
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: sk-dec1-earlier
      api_base: {upstream.url}/v1
      api_key: sk-dec1-later
{_TAIL}"""
        with booted(gateway, tmp_path, write_yaml_text(tmp_path, text)) as owned:
            assert chat(owned.gateway, "dec1-repeat").status_code == 200
            requests: Final = sent(upstream)
    assert [request.headers["authorization"] for request in requests] == ["Bearer sk-dec1-later"]


def test_a_byte_order_mark_before_the_first_key_boots_normally(gateway: Gateway, tmp_path: Path) -> None:
    """The file starts with a UTF-8 byte order mark followed by `model_list`. The proxy must boot and
    serve the model: editors on some platforms add the mark and PyYAML skips it."""
    text: Final = f"""﻿model_list:
  - model_name: dec1-bom
    litellm_params: {{model: openai/gpt-4o-mini, api_base: "http://127.0.0.1:9/v1", api_key: {UPSTREAM_KEY}}}
{_TAIL}"""
    with booted(gateway, tmp_path, write_yaml_text(tmp_path, text)) as owned:
        assert "dec1-bom" in served_model_names(owned.gateway)


def test_a_yaml_directive_and_explicit_document_markers_are_accepted(gateway: Gateway, tmp_path: Path) -> None:
    """The file opens with `%YAML 1.1`, `---` and ends with `...`. The proxy must boot and serve the
    declared model: a single explicit document is still one document."""
    text: Final = f"""\
%YAML 1.1
---
model_list:
  - model_name: dec1-directive
    litellm_params: {{model: openai/gpt-4o-mini, api_base: "http://127.0.0.1:9/v1", api_key: {UPSTREAM_KEY}}}
{_TAIL}...
"""
    with booted(gateway, tmp_path, write_yaml_text(tmp_path, text)) as owned:
        assert "dec1-directive" in served_model_names(owned.gateway)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("---\nmodel_list: []\n---\nlitellm_settings: {}\n", "expected a single document in the stream"),
        ("general_settings:\n  master_key: !Ref DEC1_KEY\n", "could not determine a constructor for the tag '!Ref'"),
        (
            "general_settings:\n  completion_model: !!python/name:os.system\n",
            "could not determine a constructor for the tag 'tag:yaml.org,2002:python/name:os.system'",
        ),
        ("general_settings:\n\tcompletion_model: x\n", "that cannot start any token"),
        ("general_settings:\n  completion_model: *dec1_missing\n", "found undefined alias 'dec1_missing'"),
        ("litellm_settings:\n  num_retries: !!int abc\n", "invalid literal for int() with base 10: 'abc'"),
        ("x-base: &base [1]\nlitellm_settings:\n  <<: *base\n", "for merging"),
    ],
    ids=[
        "two_documents",
        "custom_tag",
        "python_tag",
        "tab_indent",
        "undefined_alias",
        "int_tag_on_text",
        "merge_of_a_sequence",
    ],
)
def test_a_document_pyyaml_safe_load_rejects_stops_the_boot_with_its_message(
    gateway: Gateway, tmp_path: Path, text: str, message: str
) -> None:
    """Each config is text `yaml.safe_load` refuses: two documents, a custom or `!!python/` tag (code
    execution must never be reachable from config), a tab used for indentation, an undefined alias,
    `!!int` on letters, a merge of a sequence of scalars. The proxy must exit non-zero before becoming
    ready and the log must contain PyYAML's message (the parity table in
    `litellm/rust/crates/yaml/tests/pyyaml.rs` lists each as `error`)."""
    refusal: Final = refused_with_exit(gateway, tmp_path, write_yaml_text(tmp_path, text))
    assert refusal.exit_code != 0
    assert message in refusal.log, refusal.log[-4000:]


def test_a_yaml_parse_failure_exits_with_status_one(gateway: Gateway, tmp_path: Path) -> None:
    """A config with an unclosed flow mapping (`model_list: [{model_name: x`) cannot be parsed. The proxy
    process must exit with status 1 (an uncaught exception in the CLI's config read,
    `proxy_cli.py` `asyncio.run(proxy_config.get_config(...))`), and the log must name a YAML parser
    error. Deploy tooling keys restarts on this code.
    UNVERIFIED: that no wrapper maps the exception to another status; read from `proxy_cli.py` only."""
    refusal: Final = refused_with_exit(gateway, tmp_path, write_yaml_text(tmp_path, "model_list: [{model_name: x\n"))
    assert refusal.exit_code == 1, refusal.log[-4000:]
    assert "while parsing a flow" in refusal.log or "expected ',' or ']'" in refusal.log, refusal.log[-4000:]


def test_a_config_file_holding_only_comments_refuses_to_boot_as_empty(gateway: Gateway, tmp_path: Path) -> None:
    """A file whose only content is `# dec1 nothing here` loads as null, like an empty file. The proxy
    must exit non-zero and log `Config cannot be None or Empty.` (`proxy_server.py`
    `_get_config_from_file`)."""
    refusal: Final = refused_with_exit(gateway, tmp_path, write_yaml_text(tmp_path, "# dec1 nothing here\n"))
    assert refusal.exit_code != 0
    assert "Config cannot be None or Empty." in refusal.log, refusal.log[-4000:]


def test_a_root_config_that_is_a_yaml_list_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """The whole file is a YAML list (`- model_name: x`). The proxy must exit non-zero before becoming
    ready: a config root must be a mapping. Python fails while merging includes on a list
    (`config_includes.py` `_without_include` calls `.items()`); the Rust loader fails with `invalid YAML
    config <path>` (`config-load/src/includes.rs`).
    UNVERIFIED: the exact Python message (expected to mention `'list' object has no attribute 'items'`)."""
    refusal: Final = refused_with_exit(
        gateway, tmp_path, write_yaml_text(tmp_path, "- model_name: dec1-list-root\n  litellm_params: {}\n")
    )
    assert refusal.exit_code != 0
    assert "list" in refusal.log, refusal.log[-4000:]
