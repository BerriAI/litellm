"""How the config file and the database combine into the configuration a proxy serves.

The contract. Models: deployments stored in the database (added through `/model/new`) are served next to
the file's `model_list`, flagged `db_model: true`, and survive a restart; file models are flagged false
and cannot be deleted through the API. Settings: for every key of `general_settings`, `litellm_settings`
and `router_settings`, a key the file declares is owned by the file. The database value for that key is
ignored (with a boot warning) and the settings APIs refuse to change or delete it, naming the file. A key
the file does not declare is owned by the database: `/config/field/update` stores it, it takes effect at
once and at the next boot, and `/config/field/delete` removes it.

Note for the migration brief: the task description says "DB-stored general settings overriding file".
The product does the opposite (see `resolve` in `litellm/proxy/config_resolvers/settings_rules.py`:
"Config wins"), and these tests pin what the product does.

Tests that restart a proxy on the same data use one scratch database for both boots.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    held,
    object_value,
)
from tests.integration._support.database import read_rows, scratch_database
from tests.integration._support.process import OwnedProxy
from tests.integration.configuration.dec1_config_support import (
    ITEMS,
    active_settings,
    booted,
    chat,
    config_rows,
    deployment,
    model_entries,
    scripted_upstream,
    sent,
    served_model_names,
)

_FILE_MODEL: Final = "dec1-file-model"
_WILDCARD: Final = "dec1-db-wild/*"
_FIELD: Final = "return_wildcard_routes"
_OWNED_BY_FILE: Final = "is set in the config file and cannot be changed here."
_UNREACHABLE: Final = "http://127.0.0.1:9/v1"


def _file_models() -> list[dict[str, JsonValue]]:
    return [
        deployment(_FILE_MODEL, _UNREACHABLE, model_info={"id": "dec1-file-id"}),
        deployment(_WILDCARD, _UNREACHABLE, model="openai/*"),
    ]


def _field_update(gateway: Gateway, name: str, value: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST", "/config/field/update", {"field_name": name, "field_value": value, "config_type": "general_settings"}
    )


def _field_delete(gateway: Gateway, name: str) -> httpx.Response:
    return gateway.request("POST", "/config/field/delete", {"field_name": name, "config_type": "general_settings"})


@contextmanager
def _same_database(gateway: Gateway, directory: Path, config: dict[str, object], url: str) -> Iterator[OwnedProxy]:
    """One boot of `config` on the scratch database at `url`, so a later boot sees what this one stored."""
    with booted(
        gateway,
        directory,
        config,
        environment={"DATABASE_URL": url},
        remove_environment=("DATABASE_URL_READ_REPLICA",),
    ) as owned:
        yield owned


@pytest.fixture(scope="module")
def unowned(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    """A proxy whose file declares models but none of the settings the tests write, so the database owns them."""
    directory: Final = tmp_path_factory.mktemp("dec1-db-unowned")
    with gateway_from_environment() as shared, booted(shared, directory, {"model_list": _file_models()}) as owned:
        yield owned


@pytest.fixture(scope="module")
def owned_by_file(tmp_path_factory: pytest.TempPathFactory) -> Iterator[OwnedProxy]:
    """A proxy whose file declares `general_settings.return_wildcard_routes: false` and a litellm timeout."""
    directory: Final = tmp_path_factory.mktemp("dec1-db-owned")
    config: Final = {
        "model_list": _file_models(),
        "general_settings": {_FIELD: False},
        "litellm_settings": {"request_timeout": 42},
    }
    with gateway_from_environment() as shared, booted(shared, directory, config) as owned:
        yield owned


def test_a_model_stored_in_the_database_is_served_next_to_the_file_models(unowned: OwnedProxy) -> None:
    """The file declares one model. An admin adds another with `POST /model/new`. `GET /v1/models` must
    list both, and a chat completion for the added model must reach the provider it was given."""
    with scripted_upstream() as upstream, unowned.gateway.scenario() as scenario:
        added: Final = scenario.model(api_base=f"{upstream.url}/v1")
        names: Final = served_model_names(unowned.gateway)
        response: Final = chat(unowned.gateway, added)
        assert response.status_code == 200, response.text
        assert len(sent(upstream)) == 1
    assert {_FILE_MODEL, added} <= names, sorted(names)


def test_model_info_flags_database_models_true_and_file_models_false(unowned: OwnedProxy) -> None:
    """After adding a model through the API, `/model/info` must report `db_model: true` for it and
    `db_model: false` for the model declared in the file."""
    with unowned.gateway.scenario() as scenario:
        added: Final = scenario.model()
        flags: Final = {
            name: object_value(model_entries(unowned.gateway, name)[0]["model_info"])["db_model"]
            for name in (added, _FILE_MODEL)
        }
    assert flags == {added: True, _FILE_MODEL: False}


def test_a_database_model_with_a_file_models_name_joins_that_models_group(unowned: OwnedProxy) -> None:
    """An admin adds a database deployment using the same `model_name` as a file model. `/model/info`
    must then hold two deployments under that name, one flagged `db_model: true` and one false: the two
    sources merge into one load-balanced group instead of one replacing the other."""
    with unowned.gateway.scenario() as scenario:
        scenario.model(name=_FILE_MODEL)
        flags: Final = sorted(
            bool(object_value(entry["model_info"])["db_model"]) for entry in model_entries(unowned.gateway, _FILE_MODEL)
        )
    assert flags == [False, True]


def test_a_file_model_cannot_be_deleted_through_the_api(unowned: OwnedProxy) -> None:
    """`POST /model/delete` with the id of a model declared in the file must be refused with 400 and
    `Model with id=dec1-file-id not found in db`, and the model must still be listed afterwards."""
    response: Final = unowned.gateway.request("POST", "/model/delete", {"id": "dec1-file-id"})
    assert response.status_code == 400, response.text
    assert "Model with id=dec1-file-id not found in db" in response.text
    assert _FILE_MODEL in served_model_names(unowned.gateway)


def test_a_database_model_is_still_served_after_a_restart_on_the_same_database(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy is booted on a scratch database, a model is added through the API, and the proxy is
    stopped. A second proxy booted with the same config file on the same database must list both the
    file model and the added model without anyone adding it again."""
    config: Final[dict[str, object]] = {"model_list": _file_models()}
    with scratch_database("dec1cfg") as url:
        with _same_database(gateway, tmp_path / "first", config, url) as first:
            created: Final = first.gateway.post(
                "/model/new",
                {
                    "model_name": "dec1-persisted-model",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-dec1", "api_base": _UNREACHABLE},
                    "model_info": {},
                },
            )
            assert object_value(created["model_info"])["id"]
        with _same_database(gateway, tmp_path / "second", config, url) as second:
            names: Final = served_model_names(second.gateway)
    assert {_FILE_MODEL, "dec1-persisted-model"} <= names, sorted(names)


def test_with_store_model_in_db_off_adding_a_model_is_refused(gateway: Gateway, tmp_path: Path) -> None:
    """`STORE_MODEL_IN_DB=False` in the environment and no `store_model_in_db` in the file. `POST
    /model/new` must be refused with 500 and `Set `'STORE_MODEL_IN_DB='True'` in your env to enable this
    feature.`, and the file model keeps being listed."""
    with booted(gateway, tmp_path, {"model_list": _file_models()}, environment={"STORE_MODEL_IN_DB": "False"}) as owned:
        response: Final = owned.gateway.request(
            "POST",
            "/model/new",
            {
                "model_name": "dec1-refused-model",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-dec1", "api_base": _UNREACHABLE},
                "model_info": {},
            },
        )
        names: Final = served_model_names(owned.gateway)
    assert response.status_code == 500, response.text
    assert "Set `'STORE_MODEL_IN_DB='True'` in your env to enable this feature." in response.text
    assert _FILE_MODEL in names and "dec1-refused-model" not in names


def test_store_model_in_db_in_the_file_enables_database_models_without_the_environment_flag(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`general_settings.store_model_in_db: true` in the file with `STORE_MODEL_IN_DB=False` in the
    environment. `POST /model/new` must succeed: either source turns the feature on."""
    config: Final = {"model_list": _file_models(), "general_settings": {"store_model_in_db": True}}
    with booted(gateway, tmp_path, config, environment={"STORE_MODEL_IN_DB": "False"}) as owned:
        response: Final = owned.gateway.request(
            "POST",
            "/model/new",
            {
                "model_name": "dec1-accepted-model",
                "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-dec1", "api_base": _UNREACHABLE},
                "model_info": {},
            },
        )
    assert response.status_code == 200, response.text


def test_updating_a_setting_the_file_does_not_declare_takes_effect_at_once_and_is_stored(
    unowned: OwnedProxy,
) -> None:
    """The file does not declare `return_wildcard_routes`. `POST /config/field/update` setting it to true
    must return 200, write `return_wildcard_routes: true` into the `general_settings` row of
    `LiteLLM_Config`, and make `GET /v1/models` list the wildcard model name on the same proxy at once.
    The field is deleted again at the end so the shared proxy returns to defaults."""
    assert _WILDCARD not in served_model_names(unowned.gateway)
    response: Final = _field_update(unowned.gateway, _FIELD, True)
    try:
        assert response.status_code == 200, response.text
        assert config_rows(unowned.gateway, "general_settings")[_FIELD] is True
        assert _WILDCARD in served_model_names(unowned.gateway)
    finally:
        _field_delete(unowned.gateway, _FIELD)


def test_deleting_a_database_owned_setting_restores_the_default(unowned: OwnedProxy) -> None:
    """After `return_wildcard_routes` is set to true through the API, `POST /config/field/delete` for it
    must return 200, remove the key from the stored `general_settings` row, and make `GET /v1/models`
    stop listing the wildcard name."""
    assert _field_update(unowned.gateway, _FIELD, True).status_code == 200
    response: Final = _field_delete(unowned.gateway, _FIELD)
    assert response.status_code == 200, response.text
    assert _FIELD not in config_rows(unowned.gateway, "general_settings")
    assert _WILDCARD not in served_model_names(unowned.gateway)


def test_field_info_reports_a_database_owned_setting_as_editable(unowned: OwnedProxy) -> None:
    """After `return_wildcard_routes` is set through the API, `GET /config/field/info?field_name=...`
    must return `field_value: true`, `source: db` and `editable: true`."""
    assert _field_update(unowned.gateway, _FIELD, True).status_code == 200
    try:
        info: Final = unowned.gateway.get("/config/field/info", {"field_name": _FIELD})
    finally:
        _field_delete(unowned.gateway, _FIELD)
    assert (info["field_value"], info["source"], info["editable"]) == (True, "db", True)


def test_field_info_for_a_setting_nobody_set_is_a_400(unowned: OwnedProxy) -> None:
    """`GET /config/field/info` for a valid field that neither the file nor the database sets must return
    400 with `Field name=max_request_size_mb is not set`."""
    response: Final = unowned.gateway.request("GET", "/config/field/info", params={"field_name": "max_request_size_mb"})
    assert response.status_code == 400, response.text
    assert "Field name=max_request_size_mb is not set" in response.text


def test_field_update_for_an_unknown_field_name_is_a_400(unowned: OwnedProxy) -> None:
    """`POST /config/field/update` naming a field that is not a general setting must return 400 with
    `Invalid field=dec1_not_a_field passed in.` and store nothing."""
    response: Final = _field_update(unowned.gateway, "dec1_not_a_field", True)
    assert response.status_code == 400, response.text
    assert "Invalid field=dec1_not_a_field passed in." in response.text
    assert "dec1_not_a_field" not in config_rows(unowned.gateway, "general_settings")


def test_field_update_with_a_value_of_the_wrong_type_is_a_400(unowned: OwnedProxy) -> None:
    """`POST /config/field/update` setting the integer field `max_request_size_mb` to the text `large`
    must return 400 with `Invalid type of field value=<class 'str'> passed in.` and store nothing."""
    response: Final = _field_update(unowned.gateway, "max_request_size_mb", "large")
    assert response.status_code == 400, response.text
    assert "Invalid type of field value=<class 'str'> passed in." in response.text
    assert "max_request_size_mb" not in config_rows(unowned.gateway, "general_settings")


def test_field_delete_with_nothing_stored_is_a_400(gateway: Gateway, tmp_path: Path) -> None:
    """On a fresh database with no stored `general_settings` row, `POST /config/field/delete` for a valid
    field must return 400 with `Field name=return_wildcard_routes not in config`."""
    with booted(gateway, tmp_path, {"model_list": []}) as owned:
        response: Final = _field_delete(owned.gateway, _FIELD)
    assert response.status_code == 400, response.text
    assert f"Field name={_FIELD} not in config" in response.text


def test_updating_a_setting_the_file_declares_is_refused_and_names_the_file(owned_by_file: OwnedProxy) -> None:
    """The file declares `return_wildcard_routes: false`. `POST /config/field/update` setting it to true
    must return 400 whose body says `general_settings key 'return_wildcard_routes' is set in the config
    file and cannot be changed here.` and tells the operator to edit the file or remove the key from it.
    The value true is never stored and `GET /v1/models` still hides the wildcard name."""
    response: Final = _field_update(owned_by_file.gateway, _FIELD, True)
    assert response.status_code == 400, response.text
    text: Final = response.text
    assert f"general_settings key '{_FIELD}' {_OWNED_BY_FILE}" in text
    assert "or remove it from the file to let the database own it" in text
    assert config_rows(owned_by_file.gateway, "general_settings").get(_FIELD) is not True
    assert _WILDCARD not in served_model_names(owned_by_file.gateway)


def test_updating_a_file_declared_setting_to_the_value_it_already_has_is_accepted(
    owned_by_file: OwnedProxy,
) -> None:
    """The file declares `return_wildcard_routes: false`. `POST /config/field/update` sending the same
    value (false) must return 200: only a write that would differ from the file's value is refused."""
    response: Final = _field_update(owned_by_file.gateway, _FIELD, False)
    assert response.status_code == 200, response.text


def test_deleting_a_setting_the_file_declares_is_refused(owned_by_file: OwnedProxy) -> None:
    """The file declares `return_wildcard_routes`. `POST /config/field/delete` for it must return 400
    whose body says the key `is set in the config file and cannot be changed here.`: a delete is refused
    whatever the stored value."""
    response: Final = _field_delete(owned_by_file.gateway, _FIELD)
    assert response.status_code == 400, response.text
    assert f"general_settings key '{_FIELD}' {_OWNED_BY_FILE}" in response.text


def test_field_info_reports_a_file_declared_setting_as_not_editable(owned_by_file: OwnedProxy) -> None:
    """`GET /config/field/info` for the file-declared `return_wildcard_routes` must return
    `field_value: false`, `source: config` and `editable: false`."""
    info: Final = owned_by_file.gateway.get("/config/field/info", {"field_name": _FIELD})
    assert (info["field_value"], info["source"], info["editable"]) == (False, "config", False)


def test_config_list_marks_file_declared_settings_as_owned_by_the_config(owned_by_file: OwnedProxy) -> None:
    """`GET /config/list?config_type=general_settings` must include an entry for `return_wildcard_routes`
    with `field_type: Boolean`, `field_value: false`, `source: config` and `editable: false`.

    `stored_in_db` only says whether the `general_settings` row holds the key
    (litellm/proxy/proxy_server.py `list_config`). This module shares one proxy, and the same value update
    accepted above writes `false` into that row, so the flag is compared with the row instead of a literal."""
    response: Final = owned_by_file.gateway.request("GET", "/config/list", params={"config_type": "general_settings"})
    assert response.status_code == 200, response.text
    entry: Final = next(item for item in ITEMS.validate_json(response.content) if item["field_name"] == _FIELD)
    assert (entry["field_type"], entry["field_value"], entry["source"], entry["editable"]) == (
        "Boolean",
        False,
        "config",
        False,
    )
    assert entry["stored_in_db"] is (_FIELD in config_rows(owned_by_file.gateway, "general_settings")), entry


def test_config_list_rejects_a_config_type_other_than_general_settings(owned_by_file: OwnedProxy) -> None:
    """`GET /config/list?config_type=litellm_settings` must return 422: `general_settings` is the only
    accepted value of the query parameter."""
    response: Final = owned_by_file.gateway.request("GET", "/config/list", params={"config_type": "litellm_settings"})
    assert response.status_code == 422, response.text


def test_config_update_refuses_a_litellm_setting_the_file_declares(owned_by_file: OwnedProxy) -> None:
    """The file declares `litellm_settings.request_timeout: 42`. `POST /config/update` trying to set it to
    99 must return 400 with a body saying `litellm_settings key 'request_timeout' is set in the config
    file and cannot be changed here.`; `GET /active/callbacks` must keep reporting 42 and no
    `litellm_settings` row may be written."""
    response: Final = owned_by_file.gateway.request(
        "POST", "/config/update", {"litellm_settings": {"request_timeout": 99}}
    )
    assert response.status_code == 400, response.text
    assert f"litellm_settings key 'request_timeout' {_OWNED_BY_FILE}" in response.text
    assert active_settings(owned_by_file.gateway)["litellm.request_timeout"] == 42
    assert config_rows(owned_by_file.gateway, "litellm_settings") == {}


def test_config_update_rejects_a_router_setting_that_does_not_exist(unowned: OwnedProxy) -> None:
    """`POST /config/update` with `router_settings: {dec1_not_a_router_setting: 1}` must return 400 with
    `Unsupported router settings: dec1_not_a_router_setting are not valid router settings` and write no
    `router_settings` row."""
    response: Final = unowned.gateway.request(
        "POST", "/config/update", {"router_settings": {"dec1_not_a_router_setting": 1}}
    )
    assert response.status_code == 400, response.text
    assert "Unsupported router settings: dec1_not_a_router_setting are not valid router settings" in response.text
    assert config_rows(unowned.gateway, "router_settings") == {}


def test_config_update_stores_only_the_sections_the_request_sends(gateway: Gateway, tmp_path: Path) -> None:
    """On a fresh database, `POST /config/update` with only `litellm_settings: {dec1_marker: 1}` must
    return `{"message": "Config updated successfully"}`, write that key to the `litellm_settings` row, and
    leave the `general_settings` and `router_settings` rows absent: values the file holds are never
    copied into the database as a side effect."""
    config: Final = {"model_list": _file_models(), "general_settings": {"user_header_name": "X-Dec1-User"}}
    with booted(gateway, tmp_path, config) as owned:
        body: Final = owned.gateway.post("/config/update", {"litellm_settings": {"dec1_marker": 1}})
        stored: Final = {
            section: config_rows(owned.gateway, section)
            for section in ("litellm_settings", "general_settings", "router_settings")
        }
    assert body == {"message": "Config updated successfully"}
    assert stored == {"litellm_settings": {"dec1_marker": 1}, "general_settings": {}, "router_settings": {}}


def test_config_update_stores_environment_variables_encrypted(gateway: Gateway, tmp_path: Path) -> None:
    """`POST /config/update` with `environment_variables: {DEC1_STORED_SECRET: <plaintext>}` must store a
    value under that name in the `environment_variables` row that is not the plaintext: secrets saved
    through the API are encrypted at rest."""
    with booted(gateway, tmp_path, {"model_list": []}) as owned:
        owned.gateway.post("/config/update", {"environment_variables": {"DEC1_STORED_SECRET": "dec1-plaintext-value"}})
        stored: Final = config_rows(owned.gateway, "environment_variables")
    assert "DEC1_STORED_SECRET" in stored
    assert stored["DEC1_STORED_SECRET"] != "dec1-plaintext-value"


def test_config_update_with_a_webhook_url_turns_slack_alerting_on(gateway: Gateway, tmp_path: Path) -> None:
    """`POST /config/update` with `general_settings.alert_to_webhook_url` on a database that has no
    `alerting` stored must write `alerting: ["slack"]` next to the webhook mapping in the
    `general_settings` row."""
    hooks: Final[dict[str, JsonValue]] = {"llm_exceptions": "https://hooks.dec1.example/alerts"}
    with booted(gateway, tmp_path, {"model_list": []}) as owned:
        owned.gateway.post("/config/update", {"general_settings": {"alert_to_webhook_url": hooks}})
        stored: Final = config_rows(owned.gateway, "general_settings")
    assert stored["alerting"] == ["slack"]
    assert stored["alert_to_webhook_url"] == hooks


def test_an_allowlisted_litellm_setting_saved_through_config_update_reaches_the_running_proxy(
    gateway: Gateway, tmp_path: Path
) -> None:
    """The file does not declare `default_internal_user_params`, one of the litellm settings the database
    may override at runtime. After `POST /config/update` stores `{max_budget: 7}` for it, a user created
    with `POST /user/new` must, within two config reload intervals and without a restart, get
    `max_budget` 7."""
    with booted(gateway, tmp_path, {"model_list": []}) as owned:
        owned.gateway.post("/config/update", {"litellm_settings": {"default_internal_user_params": {"max_budget": 7}}})

        def new_user_budget() -> JsonValue:
            created: Final = owned.gateway.post(
                "/user/new", {"user_id": f"dec1-{uuid.uuid4().hex}", "auto_create_key": False}
            )
            return created.get("max_budget")

        budget: Final = eventually(
            new_user_budget, lambda value: value == 7, seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
        )
    assert budget == 7


def test_a_litellm_setting_outside_the_runtime_allowlist_is_stored_but_not_applied_even_after_a_restart(
    gateway: Gateway, tmp_path: Path
) -> None:
    """`request_timeout` is not among the litellm settings the database may override
    (`LITELLM_SETTINGS_SAFE_DB_OVERRIDES` in litellm/constants.py). `POST /config/update` with
    `litellm_settings: {request_timeout: 99}` must store 99 in the `litellm_settings` row, yet
    `GET /active/callbacks` on the running proxy must keep reporting its boot value.

    Pinned current behaviour: a second proxy booted on the same database also reports its boot value, not 99.
    `proxy_startup_event` (litellm/proxy/proxy_server.py) runs `load_config` before the database client
    exists, so the stored section is not merged at that load, and the later database pass
    (`_apply_litellm_settings_db_values`) only sets the keys in the safe override list."""
    config: Final[dict[str, object]] = {"model_list": []}
    with scratch_database("dec1cfg") as url:
        with _same_database(gateway, tmp_path / "first", config, url) as first:
            at_boot: Final = active_settings(first.gateway)["litellm.request_timeout"]
            first.gateway.post("/config/update", {"litellm_settings": {"request_timeout": 99}})
            while_running: Final = held(
                lambda: active_settings(first.gateway)["litellm.request_timeout"],
                lambda value: value == at_boot,
                holding=5,
                seconds=30,
            )
            stored: Final = read_rows(
                'SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s',
                ("litellm_settings",),
                database_url=url,
            )
        with _same_database(gateway, tmp_path / "second", config, url) as second:
            after_restart: Final = active_settings(second.gateway)["litellm.request_timeout"]
    value: Final = stored[0]["param_value"]
    assert (JSON_OBJECT.validate_json(value) if isinstance(value, str) else object_value(value))[
        "request_timeout"
    ] == 99
    assert at_boot != 99 and while_running == at_boot
    assert after_restart == at_boot


def test_a_stored_setting_applies_at_the_next_boot_when_the_file_does_not_declare_it(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Boot one: the file has no `return_wildcard_routes`; the API stores it as true. Boot two on the same
    database with the same file: `GET /v1/models` must list the wildcard name straight after boot, with no
    API call, because the stored value is merged into the file's settings at load."""
    config: Final[dict[str, object]] = {"model_list": _file_models()}
    with scratch_database("dec1cfg") as url:
        with _same_database(gateway, tmp_path / "first", config, url) as first:
            assert _field_update(first.gateway, _FIELD, True).status_code == 200
        with _same_database(gateway, tmp_path / "second", config, url) as second:
            names: Final = served_model_names(second.gateway)
    assert _WILDCARD in names


def test_the_file_wins_at_the_next_boot_over_a_stored_value_and_the_boot_log_says_so(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Boot one: the file has no `return_wildcard_routes`; the API stores it as true. Boot two on the same
    database with a file that now declares it false: `GET /v1/models` must hide the wildcard name, the
    stored row must still hold true (it is ignored, not erased), and the boot log must warn
    `general_settings.return_wildcard_routes is set in the config file, so the config file owns it and it
    cannot be changed here. The value stored in the database for it is ignored and will never be applied.`"""
    seeded: Final[dict[str, object]] = {"model_list": _file_models()}
    declared: Final[dict[str, object]] = {"model_list": _file_models(), "general_settings": {_FIELD: False}}
    with scratch_database("dec1cfg") as url:
        with _same_database(gateway, tmp_path / "first", seeded, url) as first:
            assert _field_update(first.gateway, _FIELD, True).status_code == 200
        with _same_database(gateway, tmp_path / "second", declared, url) as second:
            names: Final = served_model_names(second.gateway)
            log: Final = second.log.read_text()
        rows: Final = read_rows(
            'SELECT param_value FROM "LiteLLM_Config" WHERE param_name = %s', ("general_settings",), database_url=url
        )
    assert _WILDCARD not in names
    stored: Final = rows[0]["param_value"]
    assert (JSON_OBJECT.validate_json(stored) if isinstance(stored, str) else object_value(stored))[_FIELD] is True
    assert (
        f"general_settings.{_FIELD} is set in the config file, so the config file owns it and it cannot be "
        "changed here. The value stored in the database for it is ignored and will never be applied."
    ) in log
