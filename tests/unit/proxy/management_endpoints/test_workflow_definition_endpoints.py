from collections.abc import Mapping
from typing import Final

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from liteflow import ChatCompletion, Draft, GatewayClient, Input, Workflow, bundle_digest, compile_workflow

from litellm.models.workflow_definition import DefinitionPage
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.deps.repositories import get_workflow_definition_repository, get_workflow_execution_repository
from litellm.proxy.management_endpoints.workflow_definition_endpoints import router
from litellm.proxy.management_endpoints.workflow_execution_endpoints import router as execution_router
from litellm.repositories.workflow_definition_repository import WorkflowDefinitionRepository
from tests.unit.proxy.workflow_definition_rows import DefinitionRows, bundle, caller


@pytest.fixture(autouse=True)
def enable_workflows(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": True})


def _client(database: DefinitionRows | None, identity: UserAPIKeyAuth) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)
    app.include_router(execution_router)
    app.dependency_overrides[user_api_key_auth] = lambda: identity
    app.dependency_overrides[get_workflow_definition_repository] = lambda: (
        None if database is None else WorkflowDefinitionRepository(database.database)
    )
    app.dependency_overrides[get_workflow_execution_repository] = lambda: None
    return TestClient(app)


def _sdk(client: TestClient) -> GatewayClient:
    def send(request: httpx.Request) -> httpx.Response:
        response: Final = client.request(
            request.method, str(request.url), content=request.content, headers=dict(request.headers)
        )
        return httpx.Response(response.status_code, headers=dict(response.headers), content=response.content)

    return GatewayClient(
        base_url="http://testserver",
        api_key="synthetic",
        team_id="team",
        transport=httpx.MockTransport(send),
    )


def _body(name: str = "example", model: str = "gateway-model") -> Mapping[str, object]:
    return {"bundle": bundle(name, model).model_dump(mode="json", by_alias=True, exclude_unset=True)}


def _register(client: TestClient, name: str = "example") -> Draft:
    response: Final = client.post("/v1/workflows", json=_body(name))
    assert response.status_code == 201, response.text
    return Draft.model_validate(response.json())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("method,path,status", [
    ("POST", "/v1/workflows", 201), ("GET", "/v1/workflows/runs/missing/steps", 503),
])
def test_workflow_routes_require_opt_in(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, method: str, path: str, status: int,
) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"enable_workflows": True} if enabled else {})
    response: Final = _client(DefinitionRows(), caller()).request(method, path, json=_body())
    assert response.status_code == (status if enabled else 404), response.text


def test_real_sdk_registers_validates_replaces_and_publishes_with_an_independent_admin() -> None:
    database: Final = DefinitionRows()
    author_http: Final = _client(database, caller("reviewer"))
    publisher_http: Final = _client(
        database, UserAPIKeyAuth(user_id="publisher", user_role=LitellmUserRoles.PROXY_ADMIN, team_id="team")
    )
    workflow: Final = Workflow(name="example", nodes=[ChatCompletion(model="gateway-model", messages=Input.messages)])
    with _sdk(author_http) as author, _sdk(publisher_http) as publisher:
        draft: Final = author.register(workflow)
        assert draft.bundle == compile_workflow(workflow) and draft.digest == bundle_digest(draft.bundle)
        assert author.get(draft.workflow_id) == draft
        assert author_http.get(f"/v1/workflows/{draft.workflow_id}").headers["etag"] == draft.etag
        assert author.validate(draft) == {
            "valid": True,
            "workflow_id": draft.workflow_id,
            "etag": draft.etag,
            "digest": draft.digest,
        }
        with pytest.raises(httpx.HTTPStatusError) as denied:
            author.publish(draft)
        assert denied.value.response.status_code == 403
        replacement: Final = author.register(workflow, draft=draft)
        assert replacement.etag != draft.etag
        assert publisher.publish(replacement) == {
            "workflow_id": draft.workflow_id,
            "snapshot_id": author.get(draft.workflow_id).published_snapshot_id,
            "model": "workflow/example",
            "digest": replacement.digest,
        }


def test_stale_or_missing_revision_and_wrong_digest_cannot_publish_or_replace() -> None:
    database: Final = DefinitionRows()
    client: Final = _client(database, caller("reviewer"))
    draft: Final = _register(client)
    path: Final = f"/v1/workflows/{draft.workflow_id}"
    changed: Final = client.put(path, json=_body(model="other-model"), headers={"If-Match": draft.etag})
    assert changed.status_code == 200, changed.text
    assert client.put(path, json=_body()).status_code == 412
    assert client.put(path, json=_body(), headers={"If-Match": draft.etag}).status_code == 412
    assert client.post(f"{path}/validate").status_code == 412
    assert (
        client.post(f"{path}/publish", json={"digest": draft.digest}, headers={"If-Match": draft.etag}).status_code
        == 412
    )
    assert (
        client.post(
            f"{path}/publish", json={"digest": "0" * 64}, headers={"If-Match": changed.json()["etag"]}
        ).status_code
        == 412
    )
    assert client.get(path).json() == changed.json()


@pytest.mark.parametrize("body, status", [(b"", 200), (b"{}", 200), (b"null", 200), (b'{"execute":true}', 422)])
def test_validate_accepts_only_an_optional_empty_action(body: bytes, status: int) -> None:
    client: Final = _client(DefinitionRows(), caller())
    draft: Final = _register(client)
    response: Final = client.post(
        f"/v1/workflows/{draft.workflow_id}/validate",
        content=body,
        headers={"If-Match": draft.etag, "Content-Type": "application/json"},
    )
    assert response.status_code == status, response.text


@pytest.mark.parametrize("user, team", [("outsider", "other-team"), ("not-a-team-member", "team")])
def test_inaccessible_draft_cannot_be_read_edited_or_published(user: str, team: str) -> None:
    database: Final = DefinitionRows()
    draft: Final = _register(_client(database, caller()))
    client: Final = _client(database, caller(user, team))
    path: Final = f"/v1/workflows/{draft.workflow_id}"
    assert client.get(path).status_code == 404
    assert client.get(path, params={"team_id": "team"}).status_code == 404
    assert client.put(path, json=_body(), headers={"If-Match": draft.etag}).status_code == 404
    assert (
        client.post(f"{path}/publish", json={"digest": draft.digest}, headers={"If-Match": draft.etag}).status_code
        == 404
    )


@pytest.mark.parametrize(
    "identity, expected",
    [
        (caller(None), 403),
        (UserAPIKeyAuth(user_id="author", user_role=LitellmUserRoles.INTERNAL_USER_VIEW_ONLY), 403),
        (caller(), 500),
    ],
)
def test_no_database_preserves_authorization_precedence(identity: UserAPIKeyAuth, expected: int) -> None:
    response: Final = _client(None, identity).post("/v1/workflows", json=_body())
    assert response.status_code == expected, response.text


def test_duplicate_registration_conflicts_and_cursor_pagination_returns_each_draft_once() -> None:
    client: Final = _client(DefinitionRows(), caller())
    first: Final = _register(client, "first")
    second: Final = _register(client, "second")
    assert client.post("/v1/workflows", json=_body("first")).status_code == 409
    assert client.get("/v1/workflows", params={"cursor": "malformed"}).status_code == 422
    page: Final = DefinitionPage.model_validate(client.get("/v1/workflows", params={"limit": 1}).json())
    tail: Final = DefinitionPage.model_validate(client.get("/v1/workflows", params={"cursor": page.next_cursor}).json())
    assert {page.workflows[0].workflow_id, tail.workflows[0].workflow_id} == {first.workflow_id, second.workflow_id}
    assert page.workflows[0].workflow_id != tail.workflows[0].workflow_id and tail.next_cursor is None
