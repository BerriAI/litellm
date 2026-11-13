from collections.abc import Mapping, Sequence
from dataclasses import asdict
from types import SimpleNamespace
from typing import Final, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1.customer_activity import get_customer_activity_client, router
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.common_daily_activity import SpendMetrics
from tests.unit.proxy.management_endpoints.test_common_daily_activity import _grouping_row
from tests.unit.proxy.management_helpers.test_customer_hierarchy import Database, Record, Table
from tests.unit.repositories.test_customer_activity_repository import CustomerDatabase

RANGE: Final = {"start_date": "2024-01-01", "end_date": "2024-01-31"}
LEVELS: Final = (
    ("customer_organizations", "customer_org", "org-a", "Customer A"),
    ("customer_teams", "customer_team", "team-a1", "First team"),
    ("customer_members", "customer_member", "member-a1", "First member"),
)


def row(identifier: str, *, member: str, team: str, org: str, date: str, spend: float) -> Record:
    return Record.model_validate(
        {
            **SpendMetrics().model_dump(),
            "id": identifier,
            "customer_member_id": member,
            "customer_team_id": team,
            "customer_org_id": org,
            "date": date,
            "spend": spend,
            "api_requests": 2,
            "successful_requests": 1,
            "failed_requests": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cache_read_input_tokens": 3,
            "compression_savings_spend": 0.25,
            "api_key": "key",
            "model": "model-a",
            "model_group": "group-a",
            "custom_llm_provider": "provider-a",
            "mcp_namespaced_tool_name": None,
            "endpoint": "/chat/completions",
            "ptu_flat_cost": 0,
        }
    )


class ActivityDatabase(Database):
    def __init__(self) -> None:
        super().__init__()
        self.aliases = CustomerDatabase()
        self.litellm_dailycustomermemberspend = Table("id", self)
        self.litellm_dailycustomermemberspend.rows = {
            "a": row("a", member="member-a1", team="team-a1", org="org-a", date="2024-01-01", spend=2),
            "b": row("b", member="deleted-member", team="deleted-team", org="org-a", date="2024-01-31", spend=3),
            "c": row("c", member="member-b", team="team-b", org="org-b", date="2024-01-15", spend=7),
            "d": row("d", member="member-a1", team="team-a1", org="org-a", date="2024-02-01", spend=100),
        }
        self.litellm_verificationtoken = Table("token", self)
        self.litellm_verificationtoken.rows = {
            "key": Record(token="key", key_alias="Test key", team_id=None, user_id=None, metadata={})
        }
        self.litellm_deletedverificationtoken = Table("token", self)
        self.litellm_usertable = Table("user_id", self)

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if "GROUPING SETS" in query:
            values: Final = {
                **asdict(_grouping_row(63, spend=500)),
                "prompt_tokens": 3000,
                "completion_tokens": 2000,
                "api_requests": 1000,
            }
            return ({**values, "group_level": 127}, values)
        if "api_key_rolled" in query:
            return (
                {
                    **asdict(_grouping_row(63, spend=500)),
                    "entity_id": "org-a",
                    "api_key_rolled": 1,
                    "prompt_tokens": 3000,
                    "completion_tokens": 2000,
                    "api_requests": 1000,
                },
            )
        return await self.aliases.query_rows(query, *args)


def client(role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)

    @app.exception_handler(ManagementProblem)
    async def problem(request: Request, exc: ManagementProblem):
        return problem_response(exc.problem)

    database: Final = ActivityDatabase()
    auth: Final = UserAPIKeyAuth(user_id="admin", user_role=role)
    prisma: Final = cast(PrismaClient, SimpleNamespace(db=database))  # cast-ok: injected Prisma database edge
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    app.dependency_overrides[get_customer_activity_client] = lambda: prisma
    return TestClient(app)


@pytest.mark.parametrize("path,field,identifier,alias", LEVELS)
def test_daily_activity_uses_recorded_entity_ids_and_resolves_aliases(
    path: str, field: str, identifier: str, alias: str
) -> None:
    response: Final = client().get(
        f"/management/v1/{path}/daily/activity", params={**RANGE, f"{field}_ids": identifier}
    )
    assert response.status_code == 200, response.text
    body: Final = response.json()
    expected: Final = 5 if field == "customer_org" else 2
    assert body["metadata"]["total_spend"] == expected
    assert body["metadata"]["total_tokens"] == (30 if field == "customer_org" else 15)
    entity: Final = body["results"][-1]["breakdown"]["entities"][identifier]
    assert entity["metadata"] == {"alias": alias}
    assert entity["metrics"]["compression_savings_spend"] == 0.25
    assert entity["metrics"]["cache_read_input_tokens"] == 3
    assert entity["metrics"]["failed_requests"] == 1


def test_filtering_happens_before_paging_and_deleted_members_keep_historical_usage() -> None:
    http: Final = client()
    path: Final = "/management/v1/customer_members/daily/activity"
    params: Final = {
        **RANGE,
        "exclude_customer_member_ids": "member-b",
        "page_size": 1,
        "model": "model-a",
        "api_key": "key",
    }
    first: Final = http.get(path, params=params)
    assert first.status_code == 200, first.text
    assert first.json()["metadata"]["total_pages"] == 2
    assert first.json()["metadata"]["has_more"] is True
    assert first.json()["results"][0]["breakdown"]["entities"]["deleted-member"]["metrics"]["spend"] == 3
    second: Final = http.get(path, params={**params, "page": 2})
    assert second.json()["metadata"]["total_spend"] == 2
    assert second.json()["metadata"]["has_more"] is False
    absent: Final = http.get(path, params={**params, "model": "other-model"})
    assert absent.json()["results"] == []
    assert absent.json()["metadata"]["total_spend"] == 0


def test_aggregated_activity_returns_complete_database_totals_as_one_page() -> None:
    response: Final = client().get(
        "/management/v1/customer_organizations/daily/activity/aggregated", params={**RANGE, "customer_org_ids": "org-a"}
    )
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["metadata"]["total_spend"] == 500
    assert body["metadata"]["total_api_requests"] == 1000
    assert body["metadata"]["total_tokens"] == 5000
    assert body["metadata"]["has_more"] is False
    assert body["metadata"]["total_pages"] == 1
    assert body["results"][0]["breakdown"]["entities"]["org-a"]["metrics"]["spend"] == 500


@pytest.mark.parametrize(
    "path",
    (*tuple(level[0] + "/daily/activity" for level in LEVELS), "customer_organizations/daily/activity/aggregated"),
)
@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
def test_non_admins_cannot_read_customer_usage(path: str, role: LitellmUserRoles) -> None:
    response: Final = client(role).get(f"/management/v1/{path}", params=RANGE)
    assert response.status_code == 403


@pytest.mark.parametrize(
    "query,status",
    (
        ("start_date=2024-02-01&end_date=2024-01-01", 400),
        ("start_date=2024-02-30&end_date=2024-03-01", 400),
        ("start_date=2024-01-01", 400),
        ("start_date=2024-01-01&end_date=2024-01-31&page=0", 422),
        ("start_date=2024-01-01&end_date=2024-01-31&unknown=true", 422),
        ("start_date=2024-01-01&end_date=2024-01-31&page=1&page=2", 400),
    ),
)
def test_invalid_ranges_and_parameters_are_rejected(query: str, status: int) -> None:
    response: Final = client().get("/management/v1/customer_organizations/daily/activity?" + query)
    assert response.status_code == status, response.text


@pytest.mark.parametrize(
    "params,status",
    (
        ({**RANGE, "customer_org_ids": ""}, 422),
        ({**RANGE, "customer_org_ids": ",".join(str(index) for index in range(101))}, 422),
        ({"start_date": "2020-01-01", "end_date": "2024-01-01", "customer_org_ids": "org-a"}, 400),
    ),
)
def test_aggregate_work_is_bounded(params: Mapping[str, str], status: int) -> None:
    response: Final = client().get("/management/v1/customer_organizations/daily/activity/aggregated", params=params)
    assert response.status_code == status, response.text
