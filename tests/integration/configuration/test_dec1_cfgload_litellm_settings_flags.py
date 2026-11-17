"""`litellm_settings` flags that change what a call does: drop_params, request_timeout, set_verbose,
num_retries precedence and the dead `model_fallbacks` key.

`drop_params` is the most used of these and had no test in this directory. The proxy reads it with
`drop_params_flag` (`litellm/litellm_core_utils/core_helpers.py:296`), a lax boolean: `true`, `yes`,
`on`, `1` (any case, trimmed) are on, `false`, `no`, `off`, `0` are off, and anything else is off with a
warning. The process variable `LITELLM_DROP_PARAMS` is read at import with the opposite fallback (an
unrecognised value is on, with a warning), and a value in the file replaces it. A deployment's own
`litellm_params.drop_params` and a request's `drop_params` also turn it on. The Rust config model types
the field as `Spelled<bool>` (`litellm/rust/crates/config/src/settings.rs`), so every spelling below must
mean the same thing there.

The probe is an OpenAI o-series model: it accepts only `temperature: 1`, and any other temperature is
refused with 400 unless params are dropped, in which case the upstream receives no `temperature` at all
(`litellm/llms/openai/chat/o_series_transformation.py:108`).
"""

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.timing import assert_enforced_within, timed
from tests.integration._support.wire import Reply, Request
from tests.integration.configuration.dec1_cfgload_support import write_yaml_text
from tests.integration.configuration.dec1_config_support import (
    UPSTREAM_KEY,
    booted,
    chat,
    chat_reply,
    deployment,
    error_message,
    scripted_upstream,
    sent,
    sent_json,
)

_O_SERIES: Final = "dec1-cfgload-o-series"
_PLAIN: Final = "dec1-cfgload-plain"
_O_MODEL: Final = "openai/o3-mini"
_REFUSAL: Final = "O-series models don't support temperature=0.5. Only temperature=1 is supported."
_RETRIES: Final = {"num_retries": 0}


def _o_series_config(api_base: str, litellm_settings: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "model_list": [deployment(_O_SERIES, api_base, model=_O_MODEL)],
        "router_settings": _RETRIES,
        **({} if litellm_settings is None else {"litellm_settings": litellm_settings}),
    }


def test_without_drop_params_an_unsupported_temperature_is_a_400_naming_the_fix(
    gateway: Gateway, tmp_path: Path
) -> None:
    """No `drop_params` anywhere. A chat completion to an `openai/o3-mini` deployment with
    `temperature: 0.5` must return 400 whose `error.message` contains `O-series models don't support
    temperature=0.5. Only temperature=1 is supported.`, and the upstream must receive nothing. This is
    the control for every drop_params test below."""
    with (
        scripted_upstream() as upstream,
        booted(
            gateway, tmp_path, _o_series_config(f"{upstream.url}/v1"), remove_environment=("LITELLM_DROP_PARAMS",)
        ) as owned,
    ):
        response: Final = chat(owned.gateway, _O_SERIES, temperature=0.5)
        requests: Final = sent(upstream)
    assert response.status_code == 400, response.text
    assert _REFUSAL in error_message(response)
    assert requests == ()


def test_a_supported_temperature_is_sent_even_without_drop_params(gateway: Gateway, tmp_path: Path) -> None:
    """No `drop_params`. A chat completion with `temperature: 1` to the o-series deployment must return
    200 and the upstream body must carry `temperature: 1`: only unsupported values are refused or
    dropped."""
    with (
        scripted_upstream() as upstream,
        booted(
            gateway, tmp_path, _o_series_config(f"{upstream.url}/v1"), remove_environment=("LITELLM_DROP_PARAMS",)
        ) as owned,
    ):
        response: Final = chat(owned.gateway, _O_SERIES, temperature=1)
        requests: Final = sent(upstream)
    assert response.status_code == 200, response.text
    assert [sent_json(request).get("temperature") for request in requests] == [1]


@pytest.mark.parametrize("spelling", ["true", "yes", "on", "True", "'yes'", "'1'", "' TRUE '"])
def test_drop_params_spelled_as_any_lax_boolean_true_drops_the_unsupported_param(
    gateway: Gateway, tmp_path: Path, spelling: str
) -> None:
    """`litellm_settings.drop_params` is written (raw YAML) as `true`, `yes`, `on`, `True`, or the
    quoted strings `'yes'`, `'1'`, `' TRUE '`. A chat completion with `temperature: 0.5` must return 200
    and the upstream body must have no `temperature` key: YAML booleans and lax boolean text all turn
    the flag on (`lax_bool`, `core_helpers.py:279`)."""
    with scripted_upstream() as upstream:
        text: Final = (
            f"litellm_settings:\n  drop_params: {spelling}\n"
            f"model_list:\n  - model_name: {_O_SERIES}\n    litellm_params:\n"
            f"      model: {_O_MODEL}\n      api_base: {upstream.url}/v1\n      api_key: {UPSTREAM_KEY}\n"
            "router_settings:\n  num_retries: 0\n"
        )
        with booted(
            gateway, tmp_path, write_yaml_text(tmp_path, text), remove_environment=("LITELLM_DROP_PARAMS",)
        ) as owned:
            response: Final = chat(owned.gateway, _O_SERIES, temperature=0.5)
            requests: Final = sent(upstream)
    assert response.status_code == 200, response.text
    assert ["temperature" in sent_json(request) for request in requests] == [False]


def test_drop_params_set_to_text_that_is_not_a_flag_is_off_and_warned_about(gateway: Gateway, tmp_path: Path) -> None:
    """`litellm_settings.drop_params` is the string `maybe`. The boot log must contain
    `litellm_settings.drop_params='maybe' is not a flag value, treating it as off`, and a chat completion
    with `temperature: 0.5` must still be refused with 400 (`drop_params_flag`, `core_helpers.py:296`)."""
    with scripted_upstream() as upstream:
        config: Final = _o_series_config(f"{upstream.url}/v1", {"drop_params": "maybe"})
        with booted(gateway, tmp_path, config, remove_environment=("LITELLM_DROP_PARAMS",)) as owned:
            response: Final = chat(owned.gateway, _O_SERIES, temperature=0.5)
            log: Final = owned.log.read_text()
    assert response.status_code == 400, response.text
    assert "litellm_settings.drop_params='maybe' is not a flag value, treating it as off" in log


def test_drop_params_reached_through_an_environment_reference_is_a_boolean(gateway: Gateway, tmp_path: Path) -> None:
    """`litellm_settings.drop_params` is `os.environ/DEC1_DROP` and `DEC1_DROP=true`. The reference
    arrives as the boolean true, so a chat completion with `temperature: 0.5` must return 200 with no
    `temperature` in the upstream body."""
    with scripted_upstream() as upstream:
        config: Final = _o_series_config(f"{upstream.url}/v1", {"drop_params": "os.environ/DEC1_DROP"})
        with booted(
            gateway, tmp_path, config, environment={"DEC1_DROP": "true"}, remove_environment=("LITELLM_DROP_PARAMS",)
        ) as owned:
            response: Final = chat(owned.gateway, _O_SERIES, temperature=0.5)
            requests: Final = sent(upstream)
    assert response.status_code == 200, response.text
    assert ["temperature" in sent_json(request) for request in requests] == [False]


@pytest.mark.parametrize("value", ["true", "dec1-not-a-flag"])
def test_the_litellm_drop_params_variable_turns_dropping_on_when_the_file_is_silent(
    gateway: Gateway, tmp_path: Path, value: str
) -> None:
    """The file has no `drop_params` and the process has `LITELLM_DROP_PARAMS` set to `true` or to an
    unrecognised value. Both must drop the unsupported temperature (200, no `temperature` upstream): the
    process flag is read at import by `env_flag` (`core_helpers.py:306`), which treats an unrecognised
    value as on, the opposite of the file's fallback."""
    with (
        scripted_upstream() as upstream,
        booted(
            gateway, tmp_path, _o_series_config(f"{upstream.url}/v1"), environment={"LITELLM_DROP_PARAMS": value}
        ) as owned,
    ):
        response: Final = chat(owned.gateway, _O_SERIES, temperature=0.5)
        requests: Final = sent(upstream)
    assert response.status_code == 200, response.text
    assert ["temperature" in sent_json(request) for request in requests] == [False]


def test_drop_params_false_in_the_file_overrides_the_litellm_drop_params_variable(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`LITELLM_DROP_PARAMS=true` in the process and `litellm_settings.drop_params: false` in the file.
    A chat completion with `temperature: 0.5` must be refused with 400: the file's value is assigned to
    `litellm.drop_params` after the import-time default."""
    with scripted_upstream() as upstream:
        config: Final = _o_series_config(f"{upstream.url}/v1", {"drop_params": False})
        with booted(gateway, tmp_path, config, environment={"LITELLM_DROP_PARAMS": "true"}) as owned:
            response: Final = chat(owned.gateway, _O_SERIES, temperature=0.5)
    assert response.status_code == 400, response.text


def test_drop_params_on_one_deployment_drops_only_for_that_deployment(gateway: Gateway, tmp_path: Path) -> None:
    """Two o-series deployments; only `dec1-cfgload-o-series` sets `litellm_params.drop_params: true`,
    and nothing is set globally. `temperature: 0.5` must be served (200, no `temperature` upstream) by
    that deployment and refused with 400 by `dec1-cfgload-plain`: the per-deployment flag is scoped to
    its own entry.
    UNVERIFIED: that `drop_params` in `litellm_params` reaches `map_openai_params` as its `drop_params`
    argument for the proxy chat route."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [
                deployment(_O_SERIES, f"{upstream.url}/v1", model=_O_MODEL, drop_params=True),
                deployment(_PLAIN, f"{upstream.url}/v1", model=_O_MODEL),
            ],
            "router_settings": _RETRIES,
        }
        with booted(gateway, tmp_path, config, remove_environment=("LITELLM_DROP_PARAMS",)) as owned:
            dropped: Final = chat(owned.gateway, _O_SERIES, temperature=0.5)
            requests: Final = sent(upstream)
            refused: Final = chat(owned.gateway, _PLAIN, temperature=0.5)
    assert dropped.status_code == 200, dropped.text
    assert ["temperature" in sent_json(request) for request in requests] == [False]
    assert refused.status_code == 400, refused.text


def test_a_request_can_ask_for_drop_params_when_the_config_does_not(gateway: Gateway, tmp_path: Path) -> None:
    """No `drop_params` in the config. A chat completion with `temperature: 0.5` and `drop_params: true`
    in the body must return 200 and the upstream body must contain neither `temperature` nor
    `drop_params`: the request flag is honoured and never forwarded."""
    with (
        scripted_upstream() as upstream,
        booted(
            gateway, tmp_path, _o_series_config(f"{upstream.url}/v1"), remove_environment=("LITELLM_DROP_PARAMS",)
        ) as owned,
    ):
        response: Final = chat(owned.gateway, _O_SERIES, temperature=0.5, drop_params=True)
        requests: Final = sent(upstream)
    assert response.status_code == 200, response.text
    assert [("temperature" in sent_json(request), "drop_params" in sent_json(request)) for request in requests] == [
        (False, False)
    ]


def test_request_timeout_in_litellm_settings_bounds_a_call_to_a_silent_provider(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`litellm_settings.request_timeout: 2` and the deployment sets no `timeout`; the upstream accepts
    the request and never answers. The chat completion must come back as 408 well before the default
    600 seconds (between 2 and 12 seconds): the setting becomes `litellm.request_timeout`, which every
    deployment without its own timeout uses.
    UNVERIFIED: that the router reads `litellm.request_timeout` as its default for each call, and the
    408 status (existing routing tests show 408 for a deployment `timeout`)."""
    never: Final = Event()

    def silent(request: Request) -> Reply:
        if request.target.endswith("/chat/completions"):
            return Reply(body=chat_reply().body, gate_before_first=never)
        return chat_reply()

    with scripted_upstream(silent) as upstream:
        config: Final = {
            "litellm_settings": {"request_timeout": 2},
            "model_list": [deployment(_PLAIN, f"{upstream.url}/v1")],
            "router_settings": _RETRIES,
        }
        with booted(gateway, tmp_path, config) as owned:
            call: Final = timed(lambda: chat(owned.gateway, _PLAIN))
            never.set()
    assert call.value.status_code == 408, call.value.text
    assert_enforced_within(call.seconds, configured=2, slack=10, not_before=2)


def test_router_settings_num_retries_wins_over_litellm_settings_num_retries(gateway: Gateway, tmp_path: Path) -> None:
    """`litellm_settings.num_retries: 2` and `router_settings.num_retries: 0`; the upstream answers 500.
    One chat completion must reach the upstream exactly once: the router's own setting wins over the
    module default. (The harness adds `router_settings.num_retries: 0` to configs without one, so this is
    also why `litellm_settings.num_retries` alone is never observable here.)
    UNVERIFIED: the precedence is read from the router taking `num_retries` from `router_settings` and
    falling back to `litellm.num_retries` only when it is unset."""
    with scripted_upstream(lambda request: chat_reply("dec1 upstream down", status=500)) as upstream:
        config: Final = {
            "litellm_settings": {"num_retries": 2},
            "model_list": [deployment(_PLAIN, f"{upstream.url}/v1")],
            "router_settings": {"num_retries": 0},
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _PLAIN)
            requests: Final = sent(upstream)
    assert response.status_code == 500, response.text
    assert len(requests) == 1


def test_set_verbose_logs_its_deprecation_notice_on_a_call(gateway: Gateway, tmp_path: Path) -> None:
    """`litellm_settings.set_verbose: true`. After one chat completion the proxy log must contain
    ``litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug
    logs.``: the flag is still honoured and still warns on every call (`litellm/utils.py:1314`)."""
    with scripted_upstream() as upstream:
        config: Final = {
            "litellm_settings": {"set_verbose": True},
            "model_list": [deployment(_PLAIN, f"{upstream.url}/v1")],
            "router_settings": _RETRIES,
        }
        with booted(gateway, tmp_path, config) as owned:
            assert chat(owned.gateway, _PLAIN).status_code == 200
            log: Final = eventually(
                owned.log.read_text,
                lambda text: "`litellm.set_verbose` is deprecated." in text,
                seconds=15,
            )
    assert "Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs." in log


def test_model_fallbacks_in_litellm_settings_has_no_effect_and_says_where_to_put_it(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`litellm_settings.model_fallbacks` lists a backup model; the primary upstream answers 500. The boot
    log must contain `litellm_settings.model_fallbacks has no effect on the proxy, set
    router_settings.fallbacks instead` and the call must fail with 500 without reaching the backup
    (`proxy_server.py`, the `model_fallbacks` branch of `load_config`)."""
    with (
        scripted_upstream(lambda request: chat_reply("dec1 primary down", status=500)) as primary,
        scripted_upstream() as backup,
    ):
        config: Final = {
            "litellm_settings": {"model_fallbacks": [_PLAIN, "dec1-cfgload-backup"]},
            "model_list": [
                deployment(_PLAIN, f"{primary.url}/v1"),
                deployment("dec1-cfgload-backup", f"{backup.url}/v1"),
            ],
            "router_settings": _RETRIES,
        }
        with booted(gateway, tmp_path, config) as owned:
            response: Final = chat(owned.gateway, _PLAIN)
            log: Final = owned.log.read_text()
            reached_backup: Final = sent(backup)
    assert "litellm_settings.model_fallbacks has no effect on the proxy, set router_settings.fallbacks instead" in log
    assert response.status_code == 500, response.text
    assert reached_backup == ()
