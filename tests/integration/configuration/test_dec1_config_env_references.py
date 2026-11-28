"""`os.environ/NAME` references and the `environment_variables` section of the config file.

The contract: any string value in the config that starts with `os.environ/` is replaced, at load time,
with the value of that environment variable. Resolution walks nested mappings and mappings inside lists,
but not bare strings inside a list and not strings that merely contain the marker. A value of `true` or
`false` (any case) arrives as a boolean. A missing variable resolves to null. The `environment_variables`
section writes its entries into the process environment before the rest of the config is applied; a few
dangerous names are refused, and a value that replaces one already present is announced in the log.

The Rust loader will own this substitution, so each rule is pinned by something a caller can see: the
key the upstream receives, a header on the outbound request, or a field of a discovery endpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.integration._support.client import Gateway, object_value
from tests.integration.configuration.dec1_config_support import (
    UPSTREAM_KEY,
    booted,
    chat,
    deployment,
    model_entries,
    refused,
    scripted_upstream,
    sent,
)

_MODEL: Final = "dec1-env-model"
_UI_CONFIG: Final = "/.well-known/litellm-ui-config"


def test_an_api_key_reference_sends_the_environment_value_to_the_provider(gateway: Gateway, tmp_path: Path) -> None:
    """The deployment's `api_key` is `os.environ/DEC1_PROVIDER_KEY` and the proxy is started with that
    variable set. A chat completion must reach the upstream with `Authorization: Bearer <the variable's
    value>`, never with the literal `os.environ/...` text."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [deployment(_MODEL, f"{upstream.url}/v1", api_key="os.environ/DEC1_PROVIDER_KEY")]
        }
        with booted(gateway, tmp_path, config, environment={"DEC1_PROVIDER_KEY": "sk-dec1-from-environment"}) as owned:
            response: Final = chat(owned.gateway, _MODEL)
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [request.headers["authorization"] for request in requests] == ["Bearer sk-dec1-from-environment"]


def test_an_api_base_reference_routes_the_call_to_the_environment_url(gateway: Gateway, tmp_path: Path) -> None:
    """The deployment's `api_base` is `os.environ/DEC1_PROVIDER_BASE`, set to the scripted upstream's URL.
    A chat completion must arrive at that upstream as `POST /v1/chat/completions`."""
    with scripted_upstream() as upstream:
        config: Final = {"model_list": [deployment(_MODEL, "os.environ/DEC1_PROVIDER_BASE")]}
        with booted(gateway, tmp_path, config, environment={"DEC1_PROVIDER_BASE": f"{upstream.url}/v1"}) as owned:
            response: Final = chat(owned.gateway, _MODEL)
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [(request.method, request.target) for request in requests] == [("POST", "/v1/chat/completions")]


def test_a_reference_nested_inside_a_mapping_is_resolved(gateway: Gateway, tmp_path: Path) -> None:
    """`litellm_params.extra_headers` is a mapping whose value is `os.environ/DEC1_HEADER_VALUE`. The
    outbound provider request must carry `x-dec1-config: <the variable's value>`: resolution descends
    into nested mappings, not only top-level strings.
    UNVERIFIED: that `extra_headers` in `litellm_params` is forwarded verbatim to an OpenAI-compatible
    upstream; see `litellm/llms/openai` request building."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [
                deployment(
                    _MODEL,
                    f"{upstream.url}/v1",
                    extra_headers={"x-dec1-config": "os.environ/DEC1_HEADER_VALUE"},
                )
            ]
        }
        with booted(gateway, tmp_path, config, environment={"DEC1_HEADER_VALUE": "nested-resolved"}) as owned:
            response: Final = chat(owned.gateway, _MODEL)
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [request.headers.get("x-dec1-config") for request in requests] == ["nested-resolved"]


def test_a_reference_as_a_bare_string_inside_a_list_is_left_as_written(gateway: Gateway, tmp_path: Path) -> None:
    """`model_info` carries a scalar `os.environ/DEC1_INFO` and a list holding the same string.
    `GET /model/info` must show the scalar resolved to the variable's value and the list item still as the
    literal `os.environ/DEC1_INFO`: only mappings inside lists are walked, bare list strings are not.
    UNVERIFIED: that custom `model_info` keys are echoed unchanged by `/model/info`."""
    config: Final = {
        "model_list": [
            deployment(
                _MODEL,
                "http://127.0.0.1:9/v1",
                model_info={"dec1_scalar": "os.environ/DEC1_INFO", "dec1_list": ["os.environ/DEC1_INFO"]},
            )
        ]
    }
    with booted(gateway, tmp_path, config, environment={"DEC1_INFO": "resolved-info"}) as owned:
        entries: Final = model_entries(owned.gateway, _MODEL)
    info: Final = object_value(entries[0]["model_info"])
    assert info["dec1_scalar"] == "resolved-info"
    assert info["dec1_list"] == ["os.environ/DEC1_INFO"]


def test_the_marker_in_the_middle_of_a_string_is_not_a_reference(gateway: Gateway, tmp_path: Path) -> None:
    """A `model_info` value reads `see os.environ/DEC1_INFO`. `GET /model/info` must return it unchanged:
    only a value that starts with `os.environ/` is a reference, there is no inline interpolation."""
    config: Final = {
        "model_list": [
            deployment(_MODEL, "http://127.0.0.1:9/v1", model_info={"dec1_note": "see os.environ/DEC1_INFO"})
        ]
    }
    with booted(gateway, tmp_path, config, environment={"DEC1_INFO": "resolved-info"}) as owned:
        entries: Final = model_entries(owned.gateway, _MODEL)
    assert object_value(entries[0]["model_info"])["dec1_note"] == "see os.environ/DEC1_INFO"


@pytest.mark.parametrize(("flag", "detailed"), [("true", True), ("TRUE", True), ("false", False)])
def test_a_reference_to_a_true_or_false_variable_arrives_as_a_boolean(
    gateway: Gateway, tmp_path: Path, flag: str, detailed: bool
) -> None:
    """`general_settings.allow_public_health_readiness_details` is `os.environ/DEC1_FLAG`. The setting
    only acts when it is the boolean true, so with the variable set to `true` or `TRUE` the public
    `GET /health/readiness` must include `litellm_version`, and with `false` it must not: the text of
    the variable is converted to a boolean during resolution."""
    config: Final = {"general_settings": {"allow_public_health_readiness_details": "os.environ/DEC1_FLAG"}}
    with booted(gateway, tmp_path, config, environment={"DEC1_FLAG": flag}) as owned:
        body: Final = owned.gateway.client.get("/health/readiness").json()
    assert ("litellm_version" in body) is detailed, body


def test_a_master_key_reference_to_an_unset_variable_refuses_to_boot_and_blames_the_config_file(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`general_settings.master_key` is `os.environ/DEC1_NEVER_SET` and that variable does not exist,
    while `LITELLM_MASTER_KEY` is set. The reference resolves to null, which shadows the environment key,
    so the proxy must exit non-zero and log that `general_settings.master_key in <config path> is blank,
    or points at an environment variable that is not set.`"""
    log: Final = refused(gateway, tmp_path, {"general_settings": {"master_key": "os.environ/DEC1_NEVER_SET"}})
    assert "LiteLLM proxy refused to start: no master key is set" in log
    assert "is blank, or points at an environment variable that is not set." in log


def test_an_upperbound_reference_is_resolved_and_enforced_as_a_number(gateway: Gateway, tmp_path: Path) -> None:
    """`litellm_settings.upperbound_key_generate_params.max_budget` is `os.environ/DEC1_MAX_BUDGET`, set to
    `5`. `POST /key/generate` with `max_budget: 10` must be refused with status 400 and a message
    containing `max_budget is over max limit set in config - user_value=10.0; max_value=5.0`: the text
    from the environment is turned into the number 5."""
    config: Final = {
        "litellm_settings": {"upperbound_key_generate_params": {"max_budget": "os.environ/DEC1_MAX_BUDGET"}}
    }
    with booted(gateway, tmp_path, config, environment={"DEC1_MAX_BUDGET": "5"}) as owned:
        response: Final = owned.gateway.request("POST", "/key/generate", {"max_budget": 10})
    assert response.status_code == 400, response.text
    assert "max_budget is over max limit set in config - user_value=10.0; max_value=5.0" in response.text


def test_environment_variables_section_sets_a_literal_value_in_the_process(gateway: Gateway, tmp_path: Path) -> None:
    """`environment_variables.PROXY_BASE_URL` is a literal URL and the process starts without that
    variable. The unauthenticated `GET /.well-known/litellm-ui-config` must report `proxy_base_url` as
    that URL, proving the section wrote it into the worker's environment."""
    config: Final = {"environment_variables": {"PROXY_BASE_URL": "https://dec1-literal.example"}}
    with booted(gateway, tmp_path, config, remove_environment=("PROXY_BASE_URL",)) as owned:
        body: Final = owned.gateway.client.get(_UI_CONFIG).json()
    assert body["proxy_base_url"] == "https://dec1-literal.example"


def test_environment_variables_section_copies_a_referenced_variable_under_a_new_name(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`environment_variables.PROXY_BASE_URL` is `os.environ/DEC1_BASE_SOURCE`, which the process has.
    `GET /.well-known/litellm-ui-config` must report `proxy_base_url` as the source variable's value."""
    config: Final = {"environment_variables": {"PROXY_BASE_URL": "os.environ/DEC1_BASE_SOURCE"}}
    with booted(
        gateway,
        tmp_path,
        config,
        environment={"DEC1_BASE_SOURCE": "https://dec1-referenced.example"},
        remove_environment=("PROXY_BASE_URL",),
    ) as owned:
        body: Final = owned.gateway.client.get(_UI_CONFIG).json()
    assert body["proxy_base_url"] == "https://dec1-referenced.example"


def test_environment_variables_reference_to_an_unset_variable_leaves_the_target_unset(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`environment_variables.PROXY_BASE_URL` is `os.environ/DEC1_NEVER_SET`. The target must stay unset,
    so `GET /.well-known/litellm-ui-config` must report `proxy_base_url` as null. Today the reference is
    resolved to null before the section is applied and the null is then written as the text `None`."""
    pytest.skip(
        "BUG: environment_variables.X: os.environ/UNSET sets X to the string 'None' "
        "(get_config resolves the reference to null, _load_environment_variables then writes str(None))"
    )
    config: Final = {"environment_variables": {"PROXY_BASE_URL": "os.environ/DEC1_NEVER_SET"}}
    with booted(gateway, tmp_path, config, remove_environment=("PROXY_BASE_URL",)) as owned:
        body: Final = owned.gateway.client.get(_UI_CONFIG).json()
    assert body["proxy_base_url"] is None


def test_environment_variables_section_replaces_a_process_value_and_logs_that_it_did(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The process starts with `PROXY_BASE_URL` set and the config sets a different value for it. The
    config value must win (`proxy_base_url` in the UI discovery document), and the log must contain
    `environment_variables.PROXY_BASE_URL from the config file replaces the value already set in the
    process environment`."""
    config: Final = {"environment_variables": {"PROXY_BASE_URL": "https://dec1-from-config.example"}}
    with booted(
        gateway, tmp_path, config, environment={"PROXY_BASE_URL": "https://dec1-from-process.example"}
    ) as owned:
        body: Final = owned.gateway.client.get(_UI_CONFIG).json()
        log: Final = owned.log.read_text()
    assert body["proxy_base_url"] == "https://dec1-from-config.example"
    assert (
        "environment_variables.PROXY_BASE_URL from the config file replaces the value already set in the "
        "process environment"
    ) in log


def test_environment_variables_section_stringifies_a_yaml_boolean(gateway: Gateway, tmp_path: Path) -> None:
    """`environment_variables.LITELLM_UI_USE_SI_TERMINOLOGY: true` is a YAML boolean, not a string. It is
    written to the environment as the text `True`, so `GET /.well-known/litellm-ui-config` must report
    `use_si_terminology` as true."""
    config: Final = {"environment_variables": {"LITELLM_UI_USE_SI_TERMINOLOGY": True}}
    with booted(gateway, tmp_path, config, remove_environment=("LITELLM_UI_USE_SI_TERMINOLOGY",)) as owned:
        body: Final = owned.gateway.client.get(_UI_CONFIG).json()
    assert body["use_si_terminology"] is True


@pytest.mark.parametrize("blocked", ["PATH", "PYTHONPATH", "LD_PRELOAD", "HOME", "NO_PROXY"])
def test_environment_variables_section_refuses_names_that_change_process_execution(
    gateway: Gateway, tmp_path: Path, blocked: str
) -> None:
    """The section tries to set a name that alters how the process loads code or routes traffic (`PATH`,
    `PYTHONPATH`, `LD_PRELOAD`, `HOME`, `NO_PROXY`). The proxy must boot normally, answer readiness with
    200, and log `Skipping blocked environment variable key: <name>`; the value is never applied."""
    config: Final = {"environment_variables": {blocked: "/dec1/never/applied"}}
    with booted(gateway, tmp_path, config) as owned:
        assert owned.gateway.client.get("/health/readiness").status_code == 200
        log: Final = owned.log.read_text()
    assert f"Skipping blocked environment variable key: {blocked}" in log


def test_environment_variables_are_applied_before_model_references_are_used(gateway: Gateway, tmp_path: Path) -> None:
    """The process has no `OPENAI_API_KEY`; the config's `environment_variables` section sets it, and the
    deployment has no `api_key` of its own. A chat completion must reach the upstream with
    `Authorization: Bearer <the section's value>`: the section is applied before deployments are built,
    so provider defaults read from the environment see it.
    UNVERIFIED: that an OpenAI deployment without `api_key` falls back to `OPENAI_API_KEY` at call time."""
    with scripted_upstream() as upstream:
        config: Final = {
            "environment_variables": {"OPENAI_API_KEY": "sk-dec1-from-env-section"},
            "model_list": [deployment(_MODEL, f"{upstream.url}/v1", api_key=None)],
        }
        with booted(gateway, tmp_path, config, remove_environment=("OPENAI_API_KEY",)) as owned:
            response: Final = chat(owned.gateway, _MODEL)
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [request.headers["authorization"] for request in requests] == ["Bearer sk-dec1-from-env-section"]


def test_a_literal_api_key_is_sent_as_written(gateway: Gateway, tmp_path: Path) -> None:
    """Control for the reference cases: a deployment with a literal `api_key`. The upstream must receive
    exactly that key as the bearer token."""
    with scripted_upstream() as upstream:
        config: Final = {"model_list": [deployment(_MODEL, f"{upstream.url}/v1")]}
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _MODEL)
            assert response.status_code == 200, response.text
            requests: Final = sent(upstream)
    assert [request.headers["authorization"] for request in requests] == [f"Bearer {UPSTREAM_KEY}"]
