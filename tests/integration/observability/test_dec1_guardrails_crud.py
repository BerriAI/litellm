"""Contracts for the guardrail management API and for guardrails created through it taking effect at once.

`POST /guardrails`, `PUT`/`PATCH`/`DELETE /guardrails/{id}`, `GET /guardrails/{id}` and `/guardrails/{id}/info`,
`GET /guardrails/list` (config guardrails only) and `GET /v2/guardrails/list` (database and config guardrails, each
marked with `guardrail_definition_location`) are admin surfaces whose writes must also update the running guardrail
registry immediately, without a restart. Secrets in `litellm_params` are masked on every read: a value is shown as
its first two and last two characters around four asterisks (`synthetic-guardrail-key` reads `sy****ey`).

The management API and the database stores behind it are being ported to Rust, so the status codes, error texts,
masking and the immediate-effect behaviour are pinned here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from integration._support.client import eventually, object_value
from integration.observability.dec1_guardrails_support import (
    POLICY_KEY,
    POLICY_KEY_MASKED,
    Rig,
    boot_rig,
    chat,
    error_message,
    generic,
    marker,
    rails_called,
    refused_input,
)
from pydantic import JsonValue

CONFIG_RAIL: Final = "dec1-crud-config"


def _guardrails(policy_url: str) -> list[dict[str, JsonValue]]:
    return [generic(CONFIG_RAIL, "pre_call", policy_url) | {"guardrail_info": {"description": "dec1 from config"}}]


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with boot_rig(tmp_path_factory.mktemp("dec1-guardrails-crud"), _guardrails) as value:
        yield value


@pytest.fixture(autouse=True)
def _drained(rig: Rig) -> None:
    rig.drain()


def _definition(rig: Rig, name: str, mode: str = "pre_call") -> dict[str, JsonValue]:
    return generic(name, mode, rig.policy.url) | {"guardrail_info": {"description": "dec1 created by api"}}


def _create(rig: Rig, name: str, mode: str = "pre_call") -> str:
    created: Final = rig.gateway.post("/guardrails", {"guardrail": _definition(rig, name, mode)})
    identity: Final = str(created["guardrail_id"])
    rig.scenario.cleanups.callback(_delete_quietly, rig, identity)
    return identity


def _delete_quietly(rig: Rig, identity: str) -> None:
    response: Final = rig.gateway.request("DELETE", f"/guardrails/{identity}")
    assert response.status_code in (200, 404), response.text


def _rows(rig: Rig, name: str) -> list[dict[str, JsonValue]]:
    return rig.gateway.rows(
        'SELECT guardrail_id, guardrail_name, litellm_params, status FROM "LiteLLM_GuardrailsTable" '
        "WHERE guardrail_name = %s",
        (name,),
    )


def _name() -> str:
    return f"dec1-crud-{marker()}"


def test_create_returns_the_stored_guardrail_and_writes_one_active_row(rig: Rig) -> None:
    """The admin POSTs a Generic Guardrail API guardrail. The answer carries a guardrail_id, the given
    guardrail_name and guardrail_info, and the database holds exactly one row with that id and status "active"."""
    name: Final = _name()
    created: Final = rig.gateway.post("/guardrails", {"guardrail": _definition(rig, name)})
    rig.scenario.cleanups.callback(_delete_quietly, rig, str(created["guardrail_id"]))
    assert created["guardrail_name"] == name, created
    assert object_value(created["guardrail_info"]) == {"description": "dec1 created by api"}, created
    rows: Final = _rows(rig, name)
    assert [(row["guardrail_id"], row["status"]) for row in rows] == [(created["guardrail_id"], "active")], rows


def test_a_created_guardrail_runs_on_the_very_next_request(rig: Rig) -> None:
    """Right after POST /guardrails a chat names the new guardrail and carries BLOCK-IN. The guardrail must already
    be live: the caller gets 400 with "dec1 refused input by <name>" and no restart was needed."""
    name: Final = _name()
    _create(rig, name)
    response: Final = chat(rig, f"BLOCK-IN {marker()}", extra={"guardrails": [name]})
    assert response.status_code == 400, response.text
    assert error_message(response) == refused_input(name)


def test_a_created_guardrail_can_be_attached_to_a_key_and_runs_for_it(rig: Rig) -> None:
    """A guardrail created through the API is named in /key/generate guardrails (the name check finds it in the
    database). That key's plain chat makes the peer see the guardrail with input_type "request"."""
    name: Final = _name()
    _create(rig, name)
    key: Final = rig.scenario.key(guardrails=[name])
    assert chat(rig, f"attached {marker()}", key=key).status_code == 200
    assert rails_called(rig.policy_calls()) == ((name, "request"),)


def test_create_by_a_non_admin_key_is_refused_with_401(rig: Rig) -> None:
    """A plain virtual key (no user, so no proxy admin role) POSTs a guardrail. The route check refuses it before the
    handler runs (litellm/proxy/auth/route_checks.py:436, raised at :315-323): 401 whose message says "Only proxy
    admin can be used to generate, delete, update info" and names Route=/guardrails, and no row is written. The
    handler's own 403 "Admin access required to manage guardrails" is never reached for such a key."""
    name: Final = _name()
    key: Final = rig.scenario.key()
    response: Final = rig.gateway.request("POST", "/guardrails", {"guardrail": _definition(rig, name)}, key=key)
    assert response.status_code == 401, response.text
    assert "Only proxy admin can be used to generate, delete, update info" in error_message(response)
    assert "Route=/guardrails." in error_message(response)
    assert _rows(rig, name) == []


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_writes_by_a_non_admin_key_are_refused_with_401(rig: Rig, method: str) -> None:
    """A plain virtual key (no user) tries to PUT, PATCH or DELETE an existing guardrail. The route check answers
    each with 401 "Only proxy admin can be used to generate, delete, update info" naming the guardrail's route
    (litellm/proxy/auth/route_checks.py:436), and the row is unchanged."""
    name: Final = _name()
    identity: Final = _create(rig, name)
    key: Final = rig.scenario.key()
    body: Final[dict[str, JsonValue] | None] = (
        {"guardrail": _definition(rig, name, "post_call")}
        if method == "PUT"
        else {"guardrail_info": {"description": "changed"}}
        if method == "PATCH"
        else None
    )
    response: Final = rig.gateway.request(method, f"/guardrails/{identity}", body, key=key)
    assert response.status_code == 401, response.text
    assert "Only proxy admin can be used to generate, delete, update info" in error_message(response)
    assert f"Route=/guardrails/{identity}." in error_message(response)
    assert len(_rows(rig, name)) == 1


def test_create_with_custom_code_that_does_not_compile_is_a_400_and_leaves_no_row(rig: Rig) -> None:
    """The admin POSTs a custom_code guardrail whose code is not valid Python. The proxy answers 400 whose detail
    starts with "Guardrail configuration error:" and rolls the database write back, so no row remains (read from
    code: guardrail_endpoints.py create_guardrail)."""
    name: Final = _name()
    response: Final = rig.gateway.request(
        "POST",
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": name,
                "litellm_params": {
                    "guardrail": "custom_code",
                    "mode": "pre_call",
                    "custom_code": "def apply_guardrail(:",
                },
            }
        },
    )
    assert response.status_code == 400, response.text
    assert "Guardrail configuration error:" in response.text
    assert _rows(rig, name) == []


def test_info_masks_the_api_key_and_says_the_guardrail_lives_in_the_database(rig: Rig) -> None:
    """GET /guardrails/{id}/info for an API-created guardrail. litellm_params.api_key reads "sy****ey" (never the
    raw value) and guardrail_definition_location is "db"."""
    identity: Final = _create(rig, _name())
    info: Final = rig.gateway.get(f"/guardrails/{identity}/info")
    params: Final = object_value(info["litellm_params"])
    assert params["api_key"] == POLICY_KEY_MASKED, params
    assert POLICY_KEY not in str(info)
    assert info["guardrail_definition_location"] == "db", info


def test_the_bare_id_route_answers_the_same_as_info(rig: Rig) -> None:
    """GET /guardrails/{id} and GET /guardrails/{id}/info are the same handler. Both answer 200 with the same
    guardrail_id, guardrail_name and masked litellm_params."""
    identity: Final = _create(rig, _name())
    bare: Final = rig.gateway.get(f"/guardrails/{identity}")
    info: Final = rig.gateway.get(f"/guardrails/{identity}/info")
    for field in ("guardrail_id", "guardrail_name", "litellm_params", "guardrail_definition_location"):
        assert bare[field] == info[field], (field, bare, info)


def test_info_for_an_unknown_id_is_a_404(rig: Rig) -> None:
    """GET /guardrails/<random id>/info answers 404 with detail "Guardrail with ID <id> not found"."""
    identity: Final = marker()
    response: Final = rig.gateway.request("GET", f"/guardrails/{identity}/info")
    assert response.status_code == 404, response.text
    assert f"Guardrail with ID {identity} not found" in response.text


def test_v1_list_serves_config_guardrails_with_masked_secrets(rig: Rig) -> None:
    """GET /guardrails/list reads the guardrails from the loaded config. It names dec1-crud-config with its
    guardrail_info and the api_key masked as "sy****ey"."""
    listed: Final = rig.gateway.get("/guardrails/list")["guardrails"]
    assert isinstance(listed, list)
    entries: Final = [object_value(item) for item in listed if object_value(item)["guardrail_name"] == CONFIG_RAIL]
    assert len(entries) == 1, listed
    assert object_value(entries[0]["litellm_params"])["api_key"] == POLICY_KEY_MASKED, entries
    assert object_value(entries[0]["guardrail_info"]) == {"description": "dec1 from config"}, entries


def test_v1_list_does_not_serve_guardrails_created_through_the_api(rig: Rig) -> None:
    """A guardrail created through POST /guardrails is absent from GET /guardrails/list, which only reads the config
    file (read from code: guardrail_endpoints.py list_guardrails reads proxy_config.config["guardrails"])."""
    name: Final = _name()
    _create(rig, name)
    listed: Final = rig.gateway.get("/guardrails/list")["guardrails"]
    assert isinstance(listed, list)
    assert name not in [object_value(item)["guardrail_name"] for item in listed]


def test_v2_list_serves_database_and_config_guardrails_with_their_location(rig: Rig) -> None:
    """GET /v2/guardrails/list after creating one guardrail. It lists the API-created one with
    guardrail_definition_location "db" and the config one with "config", both with the api_key masked."""
    name: Final = _name()
    _create(rig, name)
    listed: Final = rig.gateway.get("/v2/guardrails/list")["guardrails"]
    assert isinstance(listed, list)
    by_name: Final = {str(object_value(item)["guardrail_name"]): object_value(item) for item in listed}
    assert by_name[name]["guardrail_definition_location"] == "db", by_name[name]
    assert by_name[CONFIG_RAIL]["guardrail_definition_location"] == "config", by_name[CONFIG_RAIL]
    for entry in (by_name[name], by_name[CONFIG_RAIL]):
        assert object_value(entry["litellm_params"])["api_key"] == POLICY_KEY_MASKED, entry


def test_put_changes_the_running_mode_immediately(rig: Rig) -> None:
    """A pre_call guardrail is replaced by PUT with the same definition in post_call mode. The next chat naming it
    makes the peer see it with input_type "response" only, so the in-memory registry was resynced from the write."""
    name: Final = _name()
    identity: Final = _create(rig, name)
    response: Final = rig.gateway.request(
        "PUT", f"/guardrails/{identity}", {"guardrail": _definition(rig, name, "post_call")}
    )
    assert response.status_code == 200, response.text
    assert chat(rig, f"after put {marker()}", extra={"guardrails": [name]}).status_code == 200
    assert rails_called(rig.policy_calls()) == ((name, "response"),)


def test_patch_updates_guardrail_info_and_keeps_litellm_params(rig: Rig) -> None:
    """PATCH /guardrails/{id} with only guardrail_info. The info read afterwards carries the new description while
    litellm_params.mode stays "pre_call" and the guardrail keeps refusing BLOCK-IN prompts."""
    name: Final = _name()
    identity: Final = _create(rig, name)
    response: Final = rig.gateway.request(
        "PATCH", f"/guardrails/{identity}", {"guardrail_info": {"description": "dec1 patched"}}
    )
    assert response.status_code == 200, response.text
    info: Final = rig.gateway.get(f"/guardrails/{identity}/info")
    assert object_value(info["guardrail_info"])["description"] == "dec1 patched", info
    assert object_value(info["litellm_params"])["mode"] == "pre_call", info
    assert chat(rig, f"BLOCK-IN {marker()}", extra={"guardrails": [name]}).status_code == 400


def test_put_with_an_invalid_definition_is_a_422_and_keeps_the_old_one_running(rig: Rig) -> None:
    """PUT replaces a working guardrail with a custom_code definition that does not compile. The proxy answers 422
    whose detail starts with "Invalid guardrail configuration, update rejected:" and the old definition still
    refuses BLOCK-IN prompts. UNVERIFIED: whether the DB row is rolled back; only the running behaviour is pinned."""
    name: Final = _name()
    identity: Final = _create(rig, name)
    response: Final = rig.gateway.request(
        "PUT",
        f"/guardrails/{identity}",
        {
            "guardrail": {
                "guardrail_name": name,
                "litellm_params": {
                    "guardrail": "custom_code",
                    "mode": "pre_call",
                    "custom_code": "def apply_guardrail(:",
                },
            }
        },
    )
    assert response.status_code == 422, response.text
    assert "Invalid guardrail configuration, update rejected:" in response.text
    refused: Final = chat(rig, f"BLOCK-IN {marker()}", extra={"guardrails": [name]})
    assert refused.status_code == 400, refused.text


def test_delete_removes_the_row_and_the_guardrail_stops_running(rig: Rig) -> None:
    """DELETE /guardrails/{id} answers 200, the database row is gone, and a chat that still names the guardrail with
    BLOCK-IN is served with 200 without the peer being called."""
    name: Final = _name()
    identity: Final = _create(rig, name)
    response: Final = rig.gateway.request("DELETE", f"/guardrails/{identity}")
    assert response.status_code == 200, response.text
    assert _rows(rig, name) == []
    served: Final = chat(rig, f"BLOCK-IN {marker()}", extra={"guardrails": [name]})
    assert served.status_code == 200, served.text
    assert rig.policy_calls() == ()


def test_delete_of_an_unknown_id_is_a_404(rig: Rig) -> None:
    """DELETE /guardrails/<random id> answers 404 with "Guardrail with ID <id> not found"."""
    identity: Final = marker()
    response: Final = rig.gateway.request("DELETE", f"/guardrails/{identity}")
    assert response.status_code == 404, response.text
    assert f"Guardrail with ID {identity} not found" in response.text


def test_a_deleted_guardrail_disappears_from_the_v2_list(rig: Rig) -> None:
    """After DELETE the guardrail's name is no longer in /v2/guardrails/list (reconciliation may lag on another
    worker, so the read is polled)."""
    name: Final = _name()
    identity: Final = _create(rig, name)
    assert rig.gateway.request("DELETE", f"/guardrails/{identity}").status_code == 200

    def names() -> list[str]:
        listed = rig.gateway.get("/v2/guardrails/list")["guardrails"]
        assert isinstance(listed, list)
        return [str(object_value(item)["guardrail_name"]) for item in listed]

    eventually(names, lambda values: name not in values, seconds=10)


def test_a_key_cannot_be_given_a_guardrail_after_it_was_deleted(rig: Rig) -> None:
    """A guardrail is created and deleted. /key/generate naming it afterwards answers 400 "Unknown guardrail(s)",
    because the name check reads the live registry and the active database rows."""
    name: Final = _name()
    identity: Final = _create(rig, name)
    assert rig.gateway.request("DELETE", f"/guardrails/{identity}").status_code == 200
    response: Final = rig.gateway.request("POST", "/key/generate", {"guardrails": [name]})
    assert response.status_code == 400, response.text
    assert f"Unknown guardrail(s): {name}." in response.text
