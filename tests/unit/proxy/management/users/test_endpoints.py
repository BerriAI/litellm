from collections.abc import Iterator
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1 import router

app: Final = FastAPI()


@app.exception_handler(ManagementProblem)
async def management_problem_exception_handler(request: Request, exc: ManagementProblem):
    return problem_response(exc.problem)


app.include_router(router)
client: Final = TestClient(app)

USERS_ME_PATH: Final = "/management/v1/users/me"


@pytest.fixture
def caller() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def authenticate_as(resolved: UserAPIKeyAuth) -> None:
    app.dependency_overrides[user_api_key_auth] = lambda: resolved


@pytest.mark.parametrize(
    "role",
    [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY],
)
def test_returns_the_identity_auth_resolved_for_the_caller(caller, role):
    authenticate_as(UserAPIKeyAuth(user_id="u-1", user_role=role, user_email="u1@example.com"))

    response = client.get(USERS_ME_PATH, headers={"Authorization": "Bearer k"})

    assert response.status_code == 200
    assert response.json() == {
        "user_id": "u-1",
        "user_role": role.value,
        "user_email": "u1@example.com",
        "user_alias": None,
    }


def test_current_user_returns_the_saved_alias_without_email(caller, mocker):
    user_id = "abc301ea-49c0-47c9-aee2-53b86a550582"
    authenticate_as(UserAPIKeyAuth(user_id=user_id, user_role=LitellmUserRoles.PROXY_ADMIN))
    find_unique = AsyncMock(return_value=SimpleNamespace(user_id=user_id, user_alias="Acme Developer"))
    mocker.patch(
        "litellm.proxy.proxy_server.prisma_client",
        SimpleNamespace(db=SimpleNamespace(litellm_usertable=SimpleNamespace(find_unique=find_unique))),
    )

    response = client.get(USERS_ME_PATH)

    assert response.status_code == 200, response.text
    assert response.json() == {
        "user_id": user_id,
        "user_role": "proxy_admin",
        "user_email": None,
        "user_alias": "Acme Developer",
    }
    find_unique.assert_awaited_once_with(where={"user_id": user_id})


def test_key_without_a_user_gets_a_404_problem_document(caller):
    authenticate_as(UserAPIKeyAuth(team_id="team-1", user_role=LitellmUserRoles.INTERNAL_USER))

    response = client.get(USERS_ME_PATH, headers={"Authorization": "Bearer k"})

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:litellm:error:user-not-found"
    assert response.json()["status"] == 404


def test_route_is_served_but_hidden_from_the_openapi_schema(caller):
    authenticate_as(UserAPIKeyAuth(user_id="u-1", user_role=LitellmUserRoles.INTERNAL_USER))
    paths: Final = app.openapi()["paths"]

    assert client.get(USERS_ME_PATH).status_code == 200
    assert "/management/v1/users/bulk" in paths
    assert USERS_ME_PATH not in paths
