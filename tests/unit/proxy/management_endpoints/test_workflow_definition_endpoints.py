from collections.abc import Mapping
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter

from litellm.models.workflow_definition import DefinitionPage, WorkflowDraft
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import get_workflow_definition_repository
from litellm.proxy.management_endpoints.workflow_definition_endpoints import router
from litellm.repositories.workflow_definition_repository import WorkflowDefinitionRepository, bundle_digest
from tests.unit.proxy.workflow_definition_rows import DefinitionRows, bundle, caller


@pytest.fixture(autouse=True)
def enable_workflows(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": True})


def _client(database: DefinitionRows | None, identity: UserAPIKeyAuth) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: identity
    app.dependency_overrides[get_workflow_definition_repository] = lambda: (
        None if database is None else WorkflowDefinitionRepository(database.database)
    )
    return TestClient(app)


def _body(name: str = "example", model: str = "gateway-model") -> Mapping[str, object]:
    return {"bundle": bundle(name, model).model_dump(mode="json", by_alias=True, exclude_unset=True)}


def _register(client: TestClient, name: str = "example") -> WorkflowDraft:
    response: Final = client.post("/v1/workflows", json=_body(name))
    assert response.status_code == 201, response.text
    return WorkflowDraft.model_validate(response.json())


def test_sdk_draft_contract_round_trip_has_exact_fields_and_etag() -> None:
    database: Final = DefinitionRows()
    client: Final = _client(database, caller())
    response: Final = client.post("/v1/workflows", json=_body())
    assert response.status_code == 201, response.text
    draft: Final = WorkflowDraft.model_validate(response.json())
    assert response.json() == {
        "name": "example",
        "workflow_id": draft.workflow_id,
        "team_id": "team",
        "etag": response.headers["etag"],
        "digest": bundle_digest(bundle()),
        "bundle": _body()["bundle"],
        "published_snapshot_id": None,
    }
    fetched: Final = client.get(f"/v1/workflows/{draft.workflow_id}")
    assert fetched.json() == response.json()
    assert fetched.headers["etag"] == draft.etag


@pytest.mark.parametrize("enabled", [False, True])
def test_definition_routes_require_explicit_opt_in(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": True} if enabled else {})
    response: Final = _client(DefinitionRows(), caller()).post("/v1/workflows", json=_body())
    assert response.status_code == (201 if enabled else 404), response.text


def test_stale_put_and_publish_cannot_replace_or_approve_new_revision() -> None:
    database: Final = DefinitionRows()
    author: Final = _client(database, caller())
    reviewer: Final = _client(database, caller("reviewer"))
    draft: Final = _register(author)
    path: Final = f"/v1/workflows/{draft.workflow_id}"
    changed: Final = author.put(path, json=_body(model="other-model"), headers={"If-Match": draft.etag})
    assert changed.status_code == 200, changed.text
    assert changed.json()["etag"] != draft.etag
    assert author.put(path, json=_body(), headers={"If-Match": draft.etag}).status_code == 412
    published: Final = reviewer.post(f"{path}/publish", json={"digest": draft.digest}, headers={"If-Match": draft.etag})
    assert published.status_code == 412, published.text
    assert author.get(path).json() == changed.json()
    assert database.litellm_workflowsnapshot.rows == ()


def test_publication_requires_noncontributor_administrator_and_matching_digest() -> None:
    database: Final = DefinitionRows()
    author: Final = _client(database, caller("reviewer"))
    draft: Final = _register(author)
    path: Final = f"/v1/workflows/{draft.workflow_id}/publish"
    self_publish: Final = author.post(path, json={"digest": draft.digest}, headers={"If-Match": draft.etag})
    assert self_publish.status_code == 403
    ordinary: Final = _client(database, caller("other"))
    assert ordinary.post(path, json={"digest": draft.digest}, headers={"If-Match": draft.etag}).status_code == 403
    admin: Final = _client(
        database, UserAPIKeyAuth(user_id="independent-admin", user_role=LitellmUserRoles.PROXY_ADMIN, team_id="team")
    )
    wrong_digest: Final = admin.post(path, json={"digest": "0" * 64}, headers={"If-Match": draft.etag})
    assert wrong_digest.status_code == 412
    published: Final = admin.post(path, json={"digest": draft.digest}, headers={"If-Match": draft.etag})
    assert published.status_code == 200, published.text
    assert published.json() == {
        "workflow_id": draft.workflow_id,
        "snapshot_id": published.json()["snapshot_id"],
        "model": "workflow/example",
        "digest": draft.digest,
    }
    assert len(database.litellm_workflowsnapshot.rows) == 1


def test_missing_etag_and_invalid_cursor_are_client_errors() -> None:
    database: Final = DefinitionRows()
    client: Final = _client(database, caller())
    draft: Final = _register(client)
    path: Final = f"/v1/workflows/{draft.workflow_id}"
    assert client.put(path, json=_body()).status_code == 412
    assert client.post(f"{path}/validate").status_code == 412
    assert client.get("/v1/workflows", params={"cursor": "malformed"}).status_code == 422
    valid: Final = client.post(f"{path}/validate", headers={"If-Match": draft.etag})
    assert valid.status_code == 200, valid.text
    assert valid.json() == {"valid": True, "workflow_id": draft.workflow_id, "etag": draft.etag, "digest": draft.digest}


def test_openapi_describes_created_drafts_and_optional_empty_validation_body() -> None:
    client: Final = _client(DefinitionRows(), caller())
    spec: Final = client.get("/openapi.json")
    assert spec.status_code == 200
    assert "200" not in spec.json()["paths"]["/v1/workflows"]["post"]["responses"]
    assert spec.json()["paths"]["/v1/workflows"]["post"]["responses"]["201"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/WorkflowDraft"}
    request_body: Final = TypeAdapter(Mapping[str, object]).validate_python(
        spec.json()["paths"]["/v1/workflows/{workflow_id}/validate"]["post"]["requestBody"]
    )
    assert request_body.get("required", False) is False
    assert spec.json()["components"]["schemas"]["WorkflowValidationRequest"] == {
        "properties": {},
        "additionalProperties": False,
        "type": "object",
        "title": "WorkflowValidationRequest",
    }


@pytest.mark.parametrize("body", [b"", b"{}", b"null"])
def test_validation_accepts_the_documented_optional_empty_body(body: bytes) -> None:
    client: Final = _client(DefinitionRows(), caller())
    draft: Final = _register(client)
    response: Final = client.post(
        f"/v1/workflows/{draft.workflow_id}/validate",
        content=body,
        headers={"If-Match": draft.etag, "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "valid": True,
        "workflow_id": draft.workflow_id,
        "etag": draft.etag,
        "digest": draft.digest,
    }


def test_validation_rejects_undocumented_action_fields() -> None:
    client: Final = _client(DefinitionRows(), caller())
    draft: Final = _register(client)
    response: Final = client.post(
        f"/v1/workflows/{draft.workflow_id}/validate",
        json={"execute": True},
        headers={"If-Match": draft.etag},
    )
    assert response.status_code == 422, response.text


def test_team_key_cannot_select_another_team_or_read_its_definition() -> None:
    database: Final = DefinitionRows()
    owner: Final = _client(database, caller())
    draft: Final = _register(owner)
    outsider: Final = _client(database, caller("outsider", "other-team"))
    assert outsider.get(f"/v1/workflows/{draft.workflow_id}").status_code == 404
    assert outsider.get(f"/v1/workflows/{draft.workflow_id}", params={"team_id": "team"}).status_code == 404
    assert outsider.post("/v1/workflows", json=_body(), params={"team_id": "team"}).status_code == 404
    assert outsider.get("/v1/workflows").json() == {"workflows": [], "next_cursor": None}


def test_same_team_key_requires_roster_membership_for_draft_access() -> None:
    database: Final = DefinitionRows()
    owner: Final = _client(database, caller())
    draft: Final = _register(owner)
    outsider: Final = _client(database, caller("not-a-team-member"))
    path: Final = f"/v1/workflows/{draft.workflow_id}"
    assert outsider.get(path).status_code == 404
    assert outsider.get("/v1/workflows").status_code == 404
    assert outsider.post("/v1/workflows", json=_body("second")).status_code == 404
    assert outsider.put(path, json=_body(), headers={"If-Match": draft.etag}).status_code == 404
    assert owner.get(path).json() == draft.model_dump(mode="json", by_alias=True, exclude_unset=True)


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        (caller(None), 403),
        (UserAPIKeyAuth(user_id="author", team_id="team", user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY), 403),
        (caller(), 500),
    ],
)
def test_no_database_preserves_authorization_precedence(identity: UserAPIKeyAuth, expected: int) -> None:
    response: Final = _client(None, identity).post("/v1/workflows", json=_body())
    assert response.status_code == expected, response.text


def test_duplicate_registration_is_conflict_and_list_cursor_never_repeats_rows() -> None:
    database: Final = DefinitionRows()
    client: Final = _client(database, caller())
    first: Final = _register(client, "first")
    second: Final = _register(client, "second")
    assert client.post("/v1/workflows", json=_body("first")).status_code == 409
    page: Final = DefinitionPage.model_validate(client.get("/v1/workflows", params={"limit": 1}).json())
    tail: Final = DefinitionPage.model_validate(
        client.get("/v1/workflows", params={"limit": 1, "cursor": page.next_cursor}).json()
    )
    assert {page.workflows[0].workflow_id, tail.workflows[0].workflow_id} == {
        first.workflow_id,
        second.workflow_id,
    }
    assert page.workflows[0].workflow_id != tail.workflows[0].workflow_id
    assert tail.next_cursor is None


@pytest.mark.parametrize("field", ["api_key", "api_base", "callbacks", "headers", "router_settings"])
def test_registration_rejects_untrusted_completion_capabilities(field: str) -> None:
    client: Final = _client(DefinitionRows(), caller())
    original: Final = bundle().model_dump(mode="json", by_alias=True, exclude_unset=True)
    unsafe: Final = {**original, "nodes": [{**original["nodes"][0], field: "untrusted"}]}
    response: Final = client.post("/v1/workflows", json={"bundle": unsafe})
    assert response.status_code == 422, response.text
