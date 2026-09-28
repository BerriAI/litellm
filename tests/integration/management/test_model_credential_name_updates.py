import uuid
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from tests.integration._support.database import read_rows


def _dangling_credential(gateway: Gateway, scenario: Scenario) -> str:
    name: Final = f"credential-{uuid.uuid4().hex}"
    gateway.post(
        "/credentials",
        {"credential_name": name, "credential_values": {"api_key": "synthetic-credential"}, "credential_info": {}},
    )
    scenario.cleanups.callback(_delete_credential_if_present, gateway, name)
    return name


def _delete_credential_if_present(gateway: Gateway, name: str) -> None:
    response: Final = gateway.request("DELETE", f"/credentials/{name}")
    assert response.status_code in (200, 404), response.text


def _delete_credential(gateway: Gateway, name: str) -> None:
    response: Final = gateway.request("DELETE", f"/credentials/{name}")
    assert response.status_code == 200, response.text
    assert read_rows('SELECT credential_name FROM "LiteLLM_CredentialsTable" WHERE credential_name = %s', (name,)) == []


def _model_with_credential(gateway: Gateway, scenario: Scenario, credential: str, **model_info: JsonValue) -> str:
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": f"integration-{uuid.uuid4().hex}",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "api_base": f"{gateway.upstream_url}/v1",
                "litellm_credential_name": credential,
                "rpm": 5,
            },
            "model_info": dict(model_info),
        },
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


def _stored_params(gateway: Gateway, identity: str) -> dict[str, JsonValue]:
    entries: Final = gateway.get("/model/info", {"litellm_model_id": identity})["data"]
    assert isinstance(entries, list) and len(entries) == 1, entries
    return object_value(object_value(entries[0])["litellm_params"])


def _error(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


@pytest.mark.covers("mgmt.model.update.unchanged_credential_name_is_not_revalidated")
def test_unrelated_patch_succeeds_when_resent_credential_name_is_dangling(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        credential: Final = _dangling_credential(gateway, scenario)
        identity: Final = _model_with_credential(gateway, scenario, credential)
        _delete_credential(gateway, credential)
        before: Final = _stored_params(gateway, identity)
        assert before["litellm_credential_name"] == credential
        assert before["rpm"] == 5
        patched: Final = gateway.request(
            "PATCH",
            f"/model/{identity}/update",
            {"litellm_params": {"litellm_credential_name": before["litellm_credential_name"], "rpm": 7}},
        )
        assert patched.status_code == 200, patched.text
        after: Final = _stored_params(gateway, identity)
        assert after == {**before, "rpm": 7}


@pytest.mark.covers(
    "mgmt.model.update.non_admin_detach_is_rejected",
    "mgmt.model.update.empty_credential_name_is_rejected",
)
def test_non_admin_detach_and_empty_credential_name_still_rejected(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        credential: Final = _dangling_credential(gateway, scenario)
        user: Final = scenario.user(user_role="internal_user")
        team: Final = scenario.team(members_with_roles=[{"user_id": user, "role": "admin"}])
        team_admin: Final = scenario.key(user_id=user, team_id=team)
        identity: Final = _model_with_credential(gateway, scenario, credential, team_id=team)
        before: Final = _stored_params(gateway, identity)
        detached: Final = gateway.request(
            "PATCH", f"/model/{identity}/update", {"litellm_params": {"litellm_credential_name": None}}, key=team_admin
        )
        assert detached.status_code == 403, detached.text
        assert _error(detached) == {
            "message": "Only a proxy admin can detach a stored credential (litellm_credential_name) on a model. "
            "Your role=internal_user.",
            "type": "auth_error",
            "param": "litellm_credential_name",
            "code": "403",
        }
        emptied: Final = gateway.request(
            "PATCH", f"/model/{identity}/update", {"litellm_params": {"litellm_credential_name": ""}}
        )
        assert emptied.status_code == 400, emptied.text
        assert _error(emptied) == {
            "message": "litellm_credential_name cannot be an empty string. Send null to detach the stored credential "
            "or omit the field to leave it unchanged.",
            "type": "validation_error",
            "param": "litellm_credential_name",
            "code": "400",
        }
        assert _stored_params(gateway, identity) == before


@pytest.mark.covers("mgmt.model.update.changed_missing_credential_name_is_rejected")
def test_changing_credential_name_to_missing_credential_is_rejected(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        credential: Final = _dangling_credential(gateway, scenario)
        identity: Final = _model_with_credential(gateway, scenario, credential)
        _delete_credential(gateway, credential)
        before: Final = _stored_params(gateway, identity)
        missing: Final = f"credential-{uuid.uuid4().hex}"
        rejected: Final = gateway.request(
            "PATCH", f"/model/{identity}/update", {"litellm_params": {"litellm_credential_name": missing, "rpm": 7}}
        )
        assert rejected.status_code == 400, rejected.text
        assert _error(rejected) == {
            "message": f"Credential '{missing}' not found. Create it via /credentials before attaching it to a model.",
            "type": "validation_error",
            "param": "litellm_credential_name",
            "code": "400",
        }
        assert _stored_params(gateway, identity) == before
