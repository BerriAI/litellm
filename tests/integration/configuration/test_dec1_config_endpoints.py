"""The read-only config endpoints an operator or the admin UI calls to see what the proxy loaded.

Covered here: `GET /config/yaml` (a documentation stub that must keep answering), `GET
/get/config/callbacks` (callbacks, alert destinations and live router settings), `GET /active/callbacks`
and its alias `GET /settings` (module-level litellm state), and the fact that every one of them needs a
proxy key. There is no `/reload` route in the product: settings stored in the database are picked up by
a periodic reload (see the multi-worker reload tests in this directory), and the config file is only read
at boot, which the last test pins.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml

from tests.integration._support.client import Gateway, gateway_from_environment, held, object_value
from tests.integration._support.process import OwnedProxy
from tests.integration.configuration.dec1_config_support import (
    ITEMS,
    active_settings,
    booted,
    deployment,
    served_model_names,
    write_config,
)

_ACTIVE_KEYS: Final = frozenset(
    {
        "alerting",
        "litellm.callbacks",
        "litellm.input_callback",
        "litellm.failure_callback",
        "litellm.success_callback",
        "litellm._async_success_callback",
        "litellm._async_failure_callback",
        "litellm._async_input_callback",
        "all_litellm_callbacks",
        "num_callbacks",
        "num_alerting",
        "litellm.request_timeout",
    }
)
_CONFIG_CALLBACK_KEYS: Final = frozenset(
    {"status", "callbacks", "alerts", "active_alerting_destinations", "router_settings", "available_callbacks"}
)


@pytest.fixture(scope="module")
def plain(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    directory: Final = tmp_path_factory.mktemp("dec1-endpoints-plain")
    config: Final = {
        "model_list": [deployment("dec1-endpoint-model", "http://127.0.0.1:9/v1")],
        "general_settings": {"alerting": ["slack"]},
        "litellm_settings": {"request_timeout": 55},
    }
    with gateway_from_environment() as shared, booted(shared, directory, config) as owned:
        yield owned


def test_config_yaml_endpoint_answers_hello_world_for_a_valid_config_body(plain: OwnedProxy) -> None:
    """`GET /config/yaml` with a JSON body that is a valid (here empty) config object must return 200
    `{"hello": "world"}`. The route exists to show the config schema in Swagger and changes nothing."""
    response: Final = plain.gateway.request("GET", "/config/yaml", {})
    assert response.status_code == 200, response.text
    assert response.json() == {"hello": "world"}


def test_config_yaml_endpoint_validates_the_body_against_the_config_schema(plain: OwnedProxy) -> None:
    """`GET /config/yaml` with a body whose `litellm_settings` is a string instead of a mapping must return
    422: the body is parsed as the typed config model before the stub answers. (`model_list` cannot be used
    for this: auth refuses any body that carries it, see the next test.)"""
    response: Final = plain.gateway.request("GET", "/config/yaml", {"litellm_settings": "not-a-mapping"})
    assert response.status_code == 422, response.text


def test_config_yaml_endpoint_without_a_body_is_a_422(plain: OwnedProxy) -> None:
    """`GET /config/yaml` with no body must return 422, because the config object is a required body."""
    response: Final = plain.gateway.request("GET", "/config/yaml")
    assert response.status_code == 422, response.text


def test_config_yaml_endpoint_refuses_a_model_list_body_and_does_not_change_the_served_models(
    plain: OwnedProxy,
) -> None:
    """Sending a config with a different `model_list` to `GET /config/yaml` must leave `GET /v1/models`
    exactly as it was.

    Pinned current behaviour: the request never reaches the stub. `is_request_body_safe`
    (litellm/proxy/auth/auth_utils.py) rejects any request body that carries `model_list`, on every route, so
    the answer is 401 `auth_error` with "Authentication Error, Rejected Request: model_list is not allowed in
    the request body."."""
    before: Final = served_model_names(plain.gateway)
    body: Final = {"model_list": [{"model_name": "dec1-smuggled", "litellm_params": {"model": "openai/x"}}]}
    response: Final = plain.gateway.request("GET", "/config/yaml", body)
    assert response.status_code == 401, response.text
    error: Final = object_value(response.json()["error"])
    assert (error["type"], error["message"]) == (
        "auth_error",
        "Authentication Error, Rejected Request: model_list is not allowed in the request body.",
    ), response.text
    assert served_model_names(plain.gateway) == before


def test_get_config_callbacks_returns_the_six_top_level_sections(plain: OwnedProxy) -> None:
    """`GET /get/config/callbacks` must return 200 with `status: success` and exactly the keys
    `callbacks`, `alerts`, `active_alerting_destinations`, `router_settings`, `available_callbacks` and
    `status`."""
    body: Final = plain.gateway.get("/get/config/callbacks")
    assert frozenset(body) == _CONFIG_CALLBACK_KEYS
    assert body["status"] == "success"


def test_get_config_callbacks_always_lists_email_and_ms_teams_alert_destinations(plain: OwnedProxy) -> None:
    """The `alerts` list of `GET /get/config/callbacks` must always contain entries named `email` and
    `ms_teams`, each with a `variables` mapping, whether or not they are configured."""
    alerts: Final = {
        str(item["name"]): item for item in ITEMS.validate_python(plain.gateway.get("/get/config/callbacks")["alerts"])
    }
    assert {"email", "ms_teams"} <= set(alerts)
    assert isinstance(alerts["email"]["variables"], dict)


def test_get_config_callbacks_reports_slack_when_the_file_enables_slack_alerting(plain: OwnedProxy) -> None:
    """The file sets `general_settings.alerting: ["slack"]`. `GET /get/config/callbacks` must report
    `active_alerting_destinations` as `["slack"]` and include an `alerts` entry named `slack` carrying
    `active_alerts` and `alerts_to_webhook`."""
    body: Final = plain.gateway.get("/get/config/callbacks")
    assert body["active_alerting_destinations"] == ["slack"]
    slack: Final = next(item for item in ITEMS.validate_python(body["alerts"]) if item["name"] == "slack")
    assert {"variables", "active_alerts", "alerts_to_webhook"} <= set(slack)


def test_get_config_callbacks_returns_the_live_router_settings_as_a_mapping(plain: OwnedProxy) -> None:
    """`router_settings` in `GET /get/config/callbacks` must be a non-empty mapping of the running
    router's settings (the harness sets `num_retries: 0`, which must be reported as 0).
    UNVERIFIED: the `num_retries` key name inside the mapping."""
    settings: Final = object_value(plain.gateway.get("/get/config/callbacks")["router_settings"])
    assert settings["num_retries"] == 0


def test_active_callbacks_returns_the_documented_key_set(plain: OwnedProxy) -> None:
    """`GET /active/callbacks` must return exactly the twelve documented keys (the callback lists, the
    alerting summary, the counts and `litellm.request_timeout`), with `num_callbacks` equal to the length
    of `all_litellm_callbacks`."""
    body: Final = active_settings(plain.gateway)
    assert frozenset(body) == _ACTIVE_KEYS
    all_callbacks: Final = body["all_litellm_callbacks"]
    assert isinstance(all_callbacks, list)
    assert body["num_callbacks"] == len(all_callbacks)


def test_settings_is_an_alias_of_active_callbacks(plain: OwnedProxy) -> None:
    """`GET /settings` must return the same key set as `GET /active/callbacks` and the same
    `litellm.request_timeout` (55, from the file)."""
    body: Final = plain.gateway.get("/settings")
    assert frozenset(body) == _ACTIVE_KEYS
    assert body["litellm.request_timeout"] == 55


@pytest.mark.parametrize(
    "path",
    [
        "/get/config/callbacks",
        "/active/callbacks",
        "/settings",
        "/config/list?config_type=general_settings",
        "/config/field/info?field_name=max_request_size_mb",
        "/health/readiness/details",
    ],
)
def test_config_read_endpoints_reject_a_request_without_a_key(plain: OwnedProxy, path: str) -> None:
    """Each config read endpoint called with no `Authorization` header must return 401 and none of the
    configuration: only the public readiness and liveliness probes are open."""
    response: Final = plain.gateway.client.get(path)
    assert response.status_code == 401, response.text
    assert "request_timeout" not in response.text and "router_settings" not in response.text


def test_config_update_without_a_key_is_rejected(plain: OwnedProxy) -> None:
    """`POST /config/update` with no `Authorization` header must return 401 and store nothing."""
    response: Final = plain.gateway.client.post("/config/update", json={"litellm_settings": {"dec1_marker": 1}})
    assert response.status_code == 401, response.text


def test_config_update_from_a_non_admin_key_is_refused(plain: OwnedProxy) -> None:
    """A virtual key that is not a proxy admin calls `POST /config/update`. The call must be refused
    (not 200) and the response must not say the config was updated.
    UNVERIFIED: whether the refusal is the route-access 401/403 from auth or the endpoint's own 403
    `Only proxy admins can update config`; the test pins that it is one of those two statuses."""
    with plain.gateway.scenario() as scenario:
        key: Final = scenario.key()
        response: Final = plain.gateway.request(
            "POST", "/config/update", {"litellm_settings": {"dec1_marker": 1}}, key=key
        )
    assert response.status_code in (401, 403), response.text
    assert "Config updated successfully" not in response.text


def test_editing_the_config_file_while_the_proxy_runs_changes_nothing_until_restart(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy is booted from a file declaring one model. The file is then rewritten on disk to declare a
    different model. For the next several seconds `GET /v1/models` must keep listing the original model
    and never the new one, and a restart on the edited file must list the new one: the file is read at
    boot only and there is no hot reload of it. Uses `num_retries` in the file so the harness boots the
    file in place instead of a private copy."""
    path: Final = write_config(
        tmp_path,
        {
            "model_list": [deployment("dec1-before-edit", "http://127.0.0.1:9/v1")],
            "router_settings": {"num_retries": 0},
        },
    )
    edited: Final = {
        "model_list": [deployment("dec1-after-edit", "http://127.0.0.1:9/v1")],
        "router_settings": {"num_retries": 0},
    }
    with booted(gateway, tmp_path, path) as owned:
        path.write_text(yaml.safe_dump(edited, sort_keys=False))
        names: Final = held(
            lambda: served_model_names(owned.gateway),
            lambda seen: "dec1-before-edit" in seen and "dec1-after-edit" not in seen,
            holding=5,
            seconds=30,
        )
    assert "dec1-before-edit" in names
    with booted(gateway, tmp_path, path) as restarted:
        assert "dec1-after-edit" in served_model_names(restarted.gateway)


def _served_config(owned: OwnedProxy) -> Path:
    """The config file the proxy process was started with (a harness copy under worker isolation)."""
    arguments: Final = [str(argument) for argument in owned.process.args]
    return Path(arguments[arguments.index("--config") + 1])


def _edit_general_settings(path: Path, settings: object) -> None:
    loaded: Final = object_value(yaml.safe_load(path.read_text()))
    general: Final = object_value(loaded.get("general_settings") or {})
    path.write_text(
        yaml.safe_dump({**loaded, "general_settings": {**general, **object_value(settings)}}, sort_keys=False)
    )


def test_reading_config_callbacks_re_reads_the_settings_sections_of_the_config_file(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Surprising side effect, pinned so the migration decides about it on purpose. A proxy boots from a
    file with `general_settings.return_wildcard_routes: false`; the file is edited to true while the
    proxy runs. Plain requests keep the boot value, but after one `GET /get/config/callbacks` (which loads
    the file again to build its answer) `GET /v1/models` starts listing the wildcard model name: that GET
    re-applies the file's settings sections to the running proxy. The model list itself is not reloaded.
    Derived from `get_config` calling `_load_yaml_settings_stores` and `_apply_resolved_runtime_settings` on
    every call. Under worker isolation the harness boots a copy of the file, so the edit is made to the file
    named by the proxy's own `--config` argument, which is the one it re-reads."""
    models: Final = [deployment("dec1-wild-reread/*", "http://127.0.0.1:9/v1", model="openai/*")]
    path: Final = write_config(
        tmp_path,
        {
            "model_list": models,
            "general_settings": {"return_wildcard_routes": False},
            "router_settings": {"num_retries": 0},
        },
    )
    edited: Final = {
        "model_list": models,
        "general_settings": {"return_wildcard_routes": True},
        "router_settings": {"num_retries": 0},
    }
    with booted(gateway, tmp_path, path) as owned:
        _edit_general_settings(_served_config(owned), edited["general_settings"])
        before_read: Final = served_model_names(owned.gateway)
        assert owned.gateway.request("GET", "/get/config/callbacks").status_code == 200
        after_read: Final = served_model_names(owned.gateway)
    assert "dec1-wild-reread/*" not in before_read
    assert "dec1-wild-reread/*" in after_read
