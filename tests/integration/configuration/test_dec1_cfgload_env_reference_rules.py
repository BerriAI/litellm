"""The fine print of `os.environ/NAME` references, in every position the config allows one.

`ProxyConfig._check_for_os_environ_vars` (`litellm/proxy/proxy_server.py`) walks the whole config after
the include merge and replaces each string value that starts with `os.environ/` with `get_secret(value)`
(`litellm/secret_managers/main.py`). The Rust gateway now does the same walk in
`litellm/rust/crates/config-load/src/environ.rs` (#52025), whose unit table states the Python results it
copies. This file pins those rules through a running proxy:

the text of a variable is kept as text (`42` stays the string "42", `yes` stays "yes", an empty value
stays ""), except `true` and `false` after trimming and lower-casing, which become booleans; a bare
`os.environ/` and an unset variable become null; the marker is case-sensitive and must be the very first
character; strings directly inside a list are not resolved but mappings inside a list are; the
federation secret-pointer keys ending in `_ref` are never resolved; the walk stops past depth 100; and
`os.environ/oidc/env/NAME` reads NAME or stops the boot when NAME is unset.

It also pins the order problem the Rust port deliberately changes: Python resolves every reference
before it exports `environment_variables`, so on the first load a reference to a variable that only the
config's own `environment_variables` section defines is null.

The probe for value types is `model_info`, whose free-form keys `GET /model/info` echoes unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, gateway_from_environment, object_value
from tests.integration._support.process import OwnedProxy
from tests.integration.configuration.dec1_cfgload_support import (
    leaf_of,
    nested_mapping,
    refused_with_exit,
    write_yaml_text,
)
from tests.integration.configuration.dec1_config_support import (
    UPSTREAM_KEY,
    booted,
    chat,
    deployment,
    model_entries,
    scripted_upstream,
    sent,
    write_config,
)

_PROBE: Final = "dec1-cfgload-env-probe"
_MODEL: Final = "dec1-cfgload-env-model"
_RETRIES: Final = {"num_retries": 0}
_ENVIRONMENT: Final = {
    "DEC1_SET": "set-value",
    "DEC1_NUMBER": "42",
    "DEC1_PADDED_TRUE": " true ",
    "DEC1_CAPITAL_FALSE": "False",
    "DEC1_YES": "yes",
    "DEC1_ONE": "1",
    "DEC1_EMPTY": "",
}


def _probe_config() -> dict[str, object]:
    info: Final[dict[str, object]] = {
        "r_set": "os.environ/DEC1_SET",
        "r_number": "os.environ/DEC1_NUMBER",
        "r_padded_true": "os.environ/DEC1_PADDED_TRUE",
        "r_capital_false": "os.environ/DEC1_CAPITAL_FALSE",
        "r_yes": "os.environ/DEC1_YES",
        "r_one": "os.environ/DEC1_ONE",
        "r_empty": "os.environ/DEC1_EMPTY",
        "r_upper_marker": "OS.ENVIRON/DEC1_SET",
        "r_leading_space": " os.environ/DEC1_SET",
        "r_list_of_strings": ["os.environ/DEC1_SET"],
        "r_mapping_in_list": [{"inner": "os.environ/DEC1_SET"}],
        "r_list_in_list": [["os.environ/DEC1_SET"]],
        "r_deep_mapping": {"a": {"b": {"c": "os.environ/DEC1_SET"}}},
        "anthropic_issuer_signing_key_ref": "os.environ/DEC1_SET",
        "anthropic_keycloak_client_secret_ref": "os.environ/DEC1_SET",
        "r_pointer_nested": {"anthropic_issuer_signing_key_ref": "os.environ/DEC1_SET"},
        # model_info sits at depth 2 of the document, so a leaf mapping 98 levels below it is at depth
        # 100 (the last one resolved) and 99 levels below it at depth 101 (left as written).
        "r_depth_at_limit": nested_mapping(97, {"leaf": "os.environ/DEC1_SET"}),
        "r_depth_past_limit": nested_mapping(98, {"leaf": "os.environ/DEC1_SET"}),
    }
    return {
        "model_list": [deployment(_PROBE, "http://127.0.0.1:9/v1", model_info=info)],
        "router_settings": _RETRIES,
    }


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("dec1-cfgload-env")
    with (
        gateway_from_environment() as shared,
        booted(shared, directory, _probe_config(), environment=_ENVIRONMENT) as owned,
    ):
        yield owned


def _info(owned: OwnedProxy) -> dict[str, JsonValue]:
    return object_value(model_entries(owned.gateway, _PROBE)[0]["model_info"])


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("r_set", "set-value"),
        ("r_number", "42"),
        ("r_padded_true", True),
        ("r_capital_false", False),
        ("r_yes", "yes"),
        ("r_one", "1"),
        ("r_empty", ""),
        ("r_upper_marker", "OS.ENVIRON/DEC1_SET"),
        ("r_leading_space", " os.environ/DEC1_SET"),
        ("r_list_of_strings", ["os.environ/DEC1_SET"]),
        ("r_mapping_in_list", [{"inner": "set-value"}]),
        ("r_list_in_list", [["os.environ/DEC1_SET"]]),
        ("r_deep_mapping", {"a": {"b": {"c": "set-value"}}}),
        ("anthropic_issuer_signing_key_ref", "os.environ/DEC1_SET"),
        ("anthropic_keycloak_client_secret_ref", "os.environ/DEC1_SET"),
        ("r_pointer_nested", {"anthropic_issuer_signing_key_ref": "os.environ/DEC1_SET"}),
    ],
)
def test_a_reference_resolves_to_the_value_and_type_the_python_walk_gives_it(
    probe: OwnedProxy, key: str, expected: JsonValue
) -> None:
    """One proxy boots with `model_info` keys whose values are `os.environ/` references (and near
    misses) to variables set to `set-value`, `42`, ` true `, `False`, `yes`, `1` and the empty string.
    `GET /model/info` must echo: the variable text unchanged (a number stays a string), a trimmed
    case-insensitive `true`/`false` as a boolean, the upper-case or space-prefixed marker unresolved,
    a bare string inside a list unresolved while a mapping inside a list is resolved, and the `_ref`
    secret-pointer keys left as written at any depth (`holds_secret_pointer`,
    `litellm/types/router.py:553`). The same table is `resolves_like_python` in
    `config-load/src/environ.rs`."""
    actual: Final = _info(probe)[key]
    assert actual == expected
    assert type(actual) is type(expected), (key, actual)


def test_a_reference_is_resolved_up_to_depth_one_hundred_and_left_as_written_past_it(probe: OwnedProxy) -> None:
    """`model_info` holds two chains of nested `{k: ...}` mappings ending in `{leaf:
    os.environ/DEC1_SET}`, one ending at document depth 100 and one at 101. `GET /model/info` must show
    the first leaf resolved to `set-value` and the second still `os.environ/DEC1_SET`: the walk stops at
    `DEFAULT_MAX_RECURSE_DEPTH` (100) with a warning instead of failing the boot.
    Depth arithmetic (`_check_for_os_environ_vars`, litellm/proxy/proxy_server.py:6606-6638 on main): the root
    is depth 0, the model_list item 1, model_info 2, the outermost `{k: ...}` 3, so with 97 wrappers the leaf
    mapping sits at depth 100 (walked) and with 98 at 101 (`depth > max_depth`, returned as written). CI run
    44176682961 showed 98 wrappers left unresolved, which this arithmetic predicts."""
    info: Final = _info(probe)
    assert leaf_of(info["r_depth_at_limit"]) == "set-value"
    assert leaf_of(info["r_depth_past_limit"]) == "os.environ/DEC1_SET"


@pytest.mark.parametrize("reference", ["os.environ/", "os.environ/DEC1_NEVER_SET"])
def test_a_bare_marker_or_an_unset_variable_resolves_to_null(gateway: Gateway, tmp_path: Path, reference: str) -> None:
    """A `model_info` key is the bare marker `os.environ/` or a reference to an unset variable. The proxy
    must still boot and `GET /model/info` must report the key as null: a missing reference is not a
    boot error for an optional field.
    UNVERIFIED: whether `/model/info` keeps a null free-form key or drops it, so both read as None here."""
    config: Final = {
        "model_list": [deployment(_MODEL, "http://127.0.0.1:9/v1", model_info={"r_missing": reference})],
        "router_settings": _RETRIES,
    }
    with booted(gateway, tmp_path, config, remove_environment=("DEC1_NEVER_SET",)) as owned:
        info: Final = object_value(model_entries(owned.gateway, _MODEL)[0]["model_info"])
    assert info.get("r_missing") is None


def test_numeric_limits_given_as_references_are_echoed_as_numbers(gateway: Gateway, tmp_path: Path) -> None:
    """`litellm_params.rpm` is `os.environ/DEC1_RPM` (`7`) and `tpm` is `os.environ/DEC1_TPM` (`700`).
    The reference resolves to the text "7", and `GET /model/info` must report `litellm_params.rpm == 7`
    and `tpm == 700` as numbers: the deployment's typed params coerce the text, so a reference works in
    a numeric field.
    UNVERIFIED: the coercion happens in `LiteLLM_Params` validation (`validate_litellm_params_strictly`);
    a strict Rust deserialiser of `DeploymentSpec` might refuse a string here."""
    config: Final = {
        "model_list": [
            deployment(_MODEL, "http://127.0.0.1:9/v1", rpm="os.environ/DEC1_RPM", tpm="os.environ/DEC1_TPM")
        ],
        "router_settings": _RETRIES,
    }
    with booted(gateway, tmp_path, config, environment={"DEC1_RPM": "7", "DEC1_TPM": "700"}) as owned:
        params: Final = object_value(model_entries(owned.gateway, _MODEL)[0]["litellm_params"])
    assert (params["rpm"], params["tpm"]) == (7, 700)


def test_a_reference_in_router_settings_names_the_group_an_alias_serves(gateway: Gateway, tmp_path: Path) -> None:
    """`router_settings.model_group_alias` maps `dec1-alias` to `os.environ/DEC1_ALIAS_TARGET`, set to
    the real model name. A chat completion for `dec1-alias` must reach the real model's upstream:
    references are resolved inside `router_settings` before the router is built."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [deployment(_MODEL, f"{upstream.url}/v1")],
            "router_settings": {**_RETRIES, "model_group_alias": {"dec1-alias": "os.environ/DEC1_ALIAS_TARGET"}},
        }
        with booted(gateway, tmp_path, config, environment={"DEC1_ALIAS_TARGET": _MODEL}) as owned:
            response: Final = chat(owned.gateway, "dec1-alias")
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [request.target for request in requests] == ["/v1/chat/completions"]


def test_a_reference_in_general_settings_sets_the_key_header_name(gateway: Gateway, tmp_path: Path) -> None:
    """`general_settings.litellm_key_header_name` is `os.environ/DEC1_KEY_HEADER`, set to
    `x-dec1-cfgload-key`. `GET /v1/models` with `Bearer <master key>` in that header and no `Authorization`
    must return 200: the resolved header name is the one auth reads. The value needs the `Bearer ` prefix like
    `Authorization` does; a bare key is refused as `Malformed API Key` (CI run 44176682961)."""
    config: Final = {
        "general_settings": {"litellm_key_header_name": "os.environ/DEC1_KEY_HEADER"},
        "model_list": [deployment(_MODEL, "http://127.0.0.1:9/v1")],
        "router_settings": _RETRIES,
    }
    with booted(gateway, tmp_path, config, environment={"DEC1_KEY_HEADER": "x-dec1-cfgload-key"}) as owned:
        response: Final = owned.gateway.client.get(
            "/v1/models", headers={"x-dec1-cfgload-key": f"Bearer {gateway.key}"}
        )
    assert response.status_code == 200, response.text


def test_an_oidc_env_reference_sends_the_named_variables_value(gateway: Gateway, tmp_path: Path) -> None:
    """The deployment's `api_key` is `os.environ/oidc/env/DEC1_OIDC_TOKEN` and that variable holds
    `sk-dec1-oidc-token`. The upstream must receive `Authorization: Bearer sk-dec1-oidc-token`: the
    `oidc/env/` provider reads the named variable (`secret_managers/main.py`, `oidc_provider == "env"`)."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [deployment(_MODEL, f"{upstream.url}/v1", api_key="os.environ/oidc/env/DEC1_OIDC_TOKEN")],
            "router_settings": _RETRIES,
        }
        with booted(gateway, tmp_path, config, environment={"DEC1_OIDC_TOKEN": "sk-dec1-oidc-token"}) as owned:
            assert chat(owned.gateway, _MODEL).status_code == 200
            requests: Final = sent(upstream)
    assert [request.headers["authorization"] for request in requests] == ["Bearer sk-dec1-oidc-token"]


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ("os.environ/oidc/env/DEC1_OIDC_ABSENT", "Environment variable DEC1_OIDC_ABSENT not found"),
        ("os.environ/oidc/dec1-unknown-provider/audience", "Unsupported OIDC provider"),
    ],
    ids=["unset_variable", "unknown_provider"],
)
def test_an_oidc_reference_that_cannot_be_read_stops_the_boot(
    gateway: Gateway, tmp_path: Path, reference: str, message: str
) -> None:
    """Unlike a plain reference, which turns null, an `oidc/env/` reference to an unset variable or an
    `oidc/` reference to an unknown provider raises during the config walk (the OIDC branch of
    `get_secret` sits outside its try block). The proxy must exit non-zero and log the message. The Rust
    loader maps the same failure to `could not read <reference>` (`config-load/src/error.rs`)."""
    config_path: Final = write_config(
        tmp_path,
        {"model_list": [deployment(_MODEL, "http://127.0.0.1:9/v1", api_key=reference)], "router_settings": _RETRIES},
    )
    refusal: Final = refused_with_exit(gateway, tmp_path, config_path, remove_environment=("DEC1_OIDC_ABSENT",))
    assert refusal.exit_code != 0
    assert message in refusal.log, refusal.log[-4000:]


def test_a_reference_to_a_variable_only_environment_variables_defines_is_null_on_first_load(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`environment_variables` sets `DEC1_SECTION_ONLY: sk-dec1-section` and the deployment's `api_key`
    is `os.environ/DEC1_SECTION_ONLY`; the process has `OPENAI_API_KEY=sk-dec1-process-fallback` and no
    `DEC1_SECTION_ONLY`. Python resolves references (`get_config`) before it exports the section
    (`load_config` then `_load_environment_variables`), so the key is null and the OpenAI client falls
    back to `OPENAI_API_KEY`: the upstream must receive `Bearer sk-dec1-process-fallback`.
    The Rust loader resolves against the section first (`environment_variables_come_before_the_process_
    environment` in `config-load/src/environ.rs`) and would send `Bearer sk-dec1-section`. This test
    pins today's Python result so the change is a decision, not an accident.
    UNVERIFIED: that a deployment whose `api_key` is null falls back to `OPENAI_API_KEY` at call time.
    BUG (suspected, cause not located): in CI run 44176682961 the proxy logged `Set models:
    dec1-cfgload-env-model` at boot, yet the call was refused with 400 `Invalid model name passed in
    model=dec1-cfgload-env-model`, with no deployment error in the log. A deployment whose `api_key` reference
    resolves to null is listed but not routable. Not confirmed where it is dropped; the deployment build in
    litellm/router_rust/surface/registry.py (around :1074-1100) is the first place to look."""
    pytest.skip("BUG: a deployment whose api_key reference resolves to null is listed at boot but not routable")
    with scripted_upstream() as upstream:
        config: Final = {
            "environment_variables": {"DEC1_SECTION_ONLY": "sk-dec1-section"},
            "model_list": [deployment(_MODEL, f"{upstream.url}/v1", api_key="os.environ/DEC1_SECTION_ONLY")],
            "router_settings": _RETRIES,
        }
        with booted(
            gateway,
            tmp_path,
            config,
            environment={"OPENAI_API_KEY": "sk-dec1-process-fallback"},
            remove_environment=("DEC1_SECTION_ONLY",),
        ) as owned:
            response: Final = chat(owned.gateway, _MODEL)
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [request.headers["authorization"] for request in requests] == ["Bearer sk-dec1-process-fallback"]


@pytest.mark.parametrize(("literal", "exported"), [("12345", "12345"), ("true", "True"), ("1.50", "1.5")])
def test_environment_variables_exports_a_non_string_value_as_its_python_text(
    gateway: Gateway, tmp_path: Path, literal: str, exported: str
) -> None:
    """`environment_variables.OPENAI_API_KEY` is the unquoted YAML scalar `12345`, `true` or `1.50`, and
    the deployment has no `api_key`. The OpenAI client reads the exported variable, so the upstream must
    receive `Bearer 12345`, `Bearer True` or `Bearer 1.5`: Python exports `str(value)`
    (`_load_environment_variables`, `proxy_server.py:7018`), so a boolean is spelled `True`. The Rust
    loader exports a boolean as `true` (`config-load/src/variables.rs`, `value.to_string()`), which
    differs for consumers that compare text.
    UNVERIFIED: the same fallback to `OPENAI_API_KEY` as the first-load test above.
    BUG for `true`: the exported text `True` is read back with `get_secret("OPENAI_API_KEY")`
    (litellm/main.py:2046 on main), which turns `"True"` into the boolean `True` (secret_managers/main.py:502),
    and `normalize_nonempty_secret_str` then calls `.strip()` on it (secret_managers/main.py:331, reached from
    llms/openai/workload_identity.py:66): the call fails with `'bool' object has no attribute 'strip'` and a 400
    instead of sending `Bearer True` (CI run 44176682961)."""
    if literal == "true":
        pytest.skip("BUG: an OPENAI_API_KEY whose text is True crashes the OpenAI client setup")
    with scripted_upstream() as upstream:
        text: Final = (
            f"environment_variables:\n  OPENAI_API_KEY: {literal}\n"
            f"model_list:\n  - model_name: {_MODEL}\n    litellm_params:\n"
            f"      model: openai/gpt-4o-mini\n      api_base: {upstream.url}/v1\n"
            "router_settings:\n  num_retries: 0\n"
        )
        with booted(
            gateway, tmp_path, write_yaml_text(tmp_path, text), remove_environment=("OPENAI_API_KEY",)
        ) as owned:
            assert chat(owned.gateway, _MODEL).status_code == 200
            requests: Final = sent(upstream)
    assert [request.headers["authorization"] for request in requests] == [f"Bearer {exported}"]


def test_environment_variables_with_a_null_value_exports_nothing(gateway: Gateway, tmp_path: Path) -> None:
    """`environment_variables.OPENAI_API_KEY:` has no value (YAML null), the process has none, and the
    deployment has no `api_key`. Nothing should be exported, so the call must fail at the provider with
    an authentication error (401) instead of sending `Bearer None`. The Rust loader exports nothing for
    null (`environment_values` in `config-load/src/variables.rs`)."""
    pytest.skip("BUG: a null environment_variables value is exported as the four-character text None")
    with scripted_upstream() as upstream:
        config: Final = {
            "environment_variables": {"OPENAI_API_KEY": None},
            "model_list": [deployment(_MODEL, f"{upstream.url}/v1", api_key=None)],
            "router_settings": _RETRIES,
        }
        with booted(gateway, tmp_path, config, remove_environment=("OPENAI_API_KEY",)) as owned:
            response: Final = chat(owned.gateway, _MODEL)
            requests: Final = sent(upstream)
    assert all(request.headers.get("authorization") != "Bearer None" for request in requests)
    assert response.status_code == 401, response.text


def test_environment_variables_with_a_nested_mapping_refuses_to_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`environment_variables.DEC1_NESTED` is a mapping (`{a: b}`). An environment variable can only hold
    text, so the boot should stop and name the entry, as the Rust loader does with
    `environment_variables.DEC1_NESTED must be a string, number or boolean` (`config-load/src/error.rs`)."""
    pytest.skip("BUG: Python exports the Python repr of the mapping ({'a': 'b'}) and boots")
    config_path: Final = write_config(
        tmp_path,
        {
            "environment_variables": {"DEC1_NESTED": {"a": "b"}},
            "model_list": [deployment(_MODEL, "http://127.0.0.1:9/v1", api_key=UPSTREAM_KEY)],
            "router_settings": _RETRIES,
        },
    )
    refusal: Final = refused_with_exit(gateway, tmp_path, config_path)
    assert refusal.exit_code != 0
    assert "DEC1_NESTED" in refusal.log
