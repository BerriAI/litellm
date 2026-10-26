from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(ps, "prisma_client", MagicMock())
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


CREATE_ENDPOINTS = (
    ("/key/generate", {}),
    ("/team/new", {}),
    ("/user/new", {}),
    ("/budget/new", {}),
    ("/customer/new", {"user_id": "customer-1"}),
)


@pytest.mark.parametrize(("path", "payload"), CREATE_ENDPOINTS, ids=[p for p, _ in CREATE_ENDPOINTS])
@pytest.mark.parametrize("field", ["rpm_limit", "tpm_limit"])
def test_create_endpoints_reject_negative_limits(
    client: TestClient, path: str, payload: dict[str, object], field: str
) -> None:
    """A negative limit is persisted as-is and then fails every request with 429.

    Reject it before the row is written instead of letting the entity be created
    in a state where it can never serve traffic.
    """
    response = client.post(path, json={**payload, field: -1})

    assert response.status_code == 400, response.text
    assert f"{field} must be non-negative" in response.text
