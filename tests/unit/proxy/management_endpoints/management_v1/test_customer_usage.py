from typing import Final

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1.customer_usage import get_customer_usage_repository, router
from litellm.repositories.customer_usage_repository import CustomerUsageRepository
from tests.unit.repositories.test_customer_usage_repository import UsageDatabase

PATH: Final = "/management/v1/customer_organization_usage"
RANGE: Final = "?start_date=2024-02-01&end_date=2024-02-29"


def client(role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)

    @app.exception_handler(ManagementProblem)
    async def problem(request: Request, exc: ManagementProblem):
        return problem_response(exc.problem)

    auth: Final = UserAPIKeyAuth(user_id="admin", user_role=role)
    database: Final = UsageDatabase()
    repository: Final = CustomerUsageRepository(database.query_rows)
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    app.dependency_overrides[get_customer_usage_repository] = lambda: repository
    return TestClient(app)


def test_paginated_totals_and_links_retain_the_date_range_and_search() -> None:
    http: Final = client()
    response: Final = http.get(PATH + RANGE + "&q=Customer&page_size=1")
    assert response.status_code == 200, response.text
    assert response.json()["data"] == [
        {"customer_org_id": "org-a", "alias": "Customer A", "spend": 8, "requests": 10, "tokens": 210}
    ]
    assert response.json()["meta"] == {"total_count": 2, "page": 1, "page_size": 1, "total_pages": 2}
    second: Final = http.get(response.json()["links"]["next"])
    assert second.status_code == 200, second.text
    assert second.json()["data"][0]["customer_org_id"] == "org-b"
    assert second.json()["data"][0]["spend"] == 7
    assert second.json()["meta"]["total_count"] == 2


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
def test_non_admins_cannot_read_other_customer_usage(role: LitellmUserRoles) -> None:
    response: Final = client(role).get(PATH + RANGE)
    assert response.status_code == 403


@pytest.mark.parametrize(
    "query,status",
    (
        ("?start_date=2024-03-01&end_date=2024-02-29", 400),
        ("?start_date=2024-02-30&end_date=2024-03-01", 422),
        ("?start_date=2024-02-01", 422),
        (RANGE + "&page=0", 422),
        (RANGE + "&page_size=101", 422),
        (RANGE + "&start_date=2024-01-01", 400),
        (RANGE + "&unknown=true", 400),
    ),
)
def test_invalid_ranges_and_parameters_are_rejected(query: str, status: int) -> None:
    response: Final = client().get(PATH + query)
    assert response.status_code == status, response.text
