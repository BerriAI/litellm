from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import reduce
from typing import Final, cast

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.list_api.common import ManagementProblem, problem_response
from litellm.proxy.management_endpoints.management_v1.customer_usage import get_customer_usage_prisma, router
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.common_daily_activity import SpendMetrics
from tests.unit.proxy.management_endpoints.test_common_daily_activity import _grouping_row


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class DailyRecord(SpendMetrics):
    model_config = ConfigDict(frozen=True)

    id: str
    date: str
    customer_org_id: str
    customer_team_id: str
    customer_member_id: str
    app_id: str = ""
    api_key: str
    model: str | None = None
    model_group: str | None = None
    custom_llm_provider: str | None = None
    mcp_namespaced_tool_name: str | None = None
    endpoint: str | None = None


def matches(values: Mapping[str, object], where: Mapping[str, object]) -> bool:
    for key, condition in where.items():
        if key == "OR":
            if not any(matches(values, clause) for clause in condition):
                return False
        elif key == "AND":
            if not all(matches(values, clause) for clause in condition):
                return False
        elif isinstance(condition, Mapping):
            if "in" in condition and values.get(key) not in condition["in"]:
                return False
            if "not" in condition and matches(values, {key: condition["not"]}):
                return False
            if "gte" in condition and str(values.get(key)) < str(condition["gte"]):
                return False
            if "lte" in condition and str(values.get(key)) > str(condition["lte"]):
                return False
        elif values.get(key) != condition:
            return False
    return True


@dataclass(frozen=True, slots=True)
class Table:
    rows: tuple[Record | DailyRecord, ...] = ()

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        take: int | None = None,
        skip: int = 0,
        order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
    ) -> tuple[Record | DailyRecord, ...]:
        selected: Final = tuple(row for row in self.rows if matches(row.model_dump(), where or {}))
        orders: Final = (order,) if isinstance(order, Mapping) else tuple(order or ())
        ordered: Final = reduce(
            lambda rows, item: tuple(
                sorted(
                    rows,
                    key=lambda row: str(row.model_dump()[next(iter(item))]),
                    reverse=next(iter(item.values())) == "desc",
                )
            ),
            reversed(orders),
            selected,
        )
        return ordered[skip:] if take is None else ordered[skip : skip + take]

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> Record | DailyRecord | None:
        return next((row for row in self.rows if matches(row.model_dump(), where)), None)

    async def count(self, where: Mapping[str, object] | None = None) -> int:
        return len(await self.find_many(where=where))


@dataclass(frozen=True, slots=True)
class Database:
    litellm_dailycustomermemberspend: Table
    litellm_customerorganizationtable: Table = Table(
        (
            Record(customer_org_id="org-a", alias="Organization A"),
            Record(customer_org_id="org-b", alias="Organization B"),
        )
    )
    litellm_customerteamtable: Table = Table(
        (Record(customer_team_id="team-a", alias="Product"), Record(customer_team_id="team-b", alias="Operations"))
    )
    litellm_customermembertable: Table = Table(
        (
            Record(customer_member_id="member-a", alias="Member A"),
            Record(customer_member_id="member-b", alias="Member B"),
        )
    )
    litellm_verificationtoken: Table = Table(
        (
            Record(token="hash-a", key_alias="Support", team_id=None, user_id=None, metadata=None),
            Record(token="hash-b", key_alias="Search", team_id=None, user_id=None, metadata=None),
        )
    )
    litellm_apptable: Table = Table(
        (
            Record(app_id="chat", resource_type="team", resource_id="chat-team"),
            Record(app_id="search", resource_type="team", resource_id="search-team"),
        )
    )
    litellm_deletedverificationtoken: Table = Table()
    litellm_usertable: Table = Table()
    litellm_teamtable: Table = Table()
    litellm_organizationtable: Table = Table()
    litellm_organizationmembership: Table = Table()
    litellm_objectpermissiontable: Table = Table()


@dataclass(frozen=True, slots=True)
class Client:
    db: Database


def usage_row(
    *,
    id: str = "usage-a",
    date: str = "2026-01-01",
    customer_org_id: str = "org-a",
    customer_team_id: str = "team-a",
    customer_member_id: str = "member-a",
    api_key: str = "hash-a",
    model: str = "model-a",
    model_group: str = "alias-a",
    provider: str = "provider-a",
    spend: float = 2.0,
    app_id: str = "",
) -> DailyRecord:
    return DailyRecord(
        id=id,
        date=date,
        customer_org_id=customer_org_id,
        customer_team_id=customer_team_id,
        customer_member_id=customer_member_id,
        app_id=app_id,
        api_key=api_key,
        model=model,
        model_group=model_group,
        custom_llm_provider=provider,
        spend=spend,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        api_requests=3,
        successful_requests=2,
        failed_requests=1,
    )


def client(
    rows: tuple[DailyRecord, ...] = (),
    role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN,
    database: Database | None = None,
) -> TestClient:
    app: Final = FastAPI()
    app.include_router(router)

    @app.exception_handler(ManagementProblem)
    async def problem(request: Request, exc: ManagementProblem):
        return problem_response(exc.problem)

    auth: Final = UserAPIKeyAuth(user_id="admin", user_role=role)
    prisma: Final = cast(PrismaClient, Client(database or Database(Table(rows))))  # cast-ok: in-memory Prisma boundary
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    app.dependency_overrides[get_customer_usage_prisma] = lambda: prisma
    return TestClient(app)


DATES: Final = {"start_date": "2026-01-01", "end_date": "2026-01-03"}
SCOPES: Final = (
    ("customer_organizations", "customer_org_ids", "exclude_customer_org_ids", "org-a", "org-b", "Organization A"),
    ("customer_teams", "customer_team_ids", "exclude_customer_team_ids", "team-a", "team-b", "Product"),
    ("customer_members", "customer_member_ids", "exclude_customer_member_ids", "member-a", "member-b", "Member A"),
)


@pytest.mark.parametrize("resource,parameter,excluded,first,second,label", SCOPES)
def test_standard_daily_activity_filters_entities_and_preserves_aliases_and_dimensions(
    resource: str, parameter: str, excluded: str, first: str, second: str, label: str
) -> None:
    rows: Final = (
        usage_row(),
        usage_row(id="usage-c", date="2026-01-03", model_group="alias-b", provider="provider-b", spend=4),
        usage_row(
            id="usage-b", customer_org_id="org-b", customer_team_id="team-b", customer_member_id="member-b", spend=10
        ),
        usage_row(id="outside-range", date="2025-12-31", spend=50),
    )
    response: Final = client(rows).get(f"/management/v1/{resource}/daily/activity", params={**DATES, parameter: first})
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert set(body) == {"results", "metadata"}
    assert body["metadata"]["total_spend"] == 6
    assert body["metadata"]["total_tokens"] == 30
    assert body["metadata"]["total_api_requests"] == 6
    assert body["metadata"]["has_more"] is False
    day: Final = body["results"][0]
    assert day["breakdown"]["entities"][first]["metadata"]["alias"] == label
    assert day["breakdown"]["models"]["model-a"]["metrics"]["spend"] == 4
    assert day["breakdown"]["model_groups"]["alias-b"]["metrics"]["spend"] == 4
    assert day["breakdown"]["providers"]["provider-b"]["metrics"]["spend"] == 4
    assert day["breakdown"]["api_keys"]["hash-a"]["metadata"]["key_alias"] == "Support"


@pytest.mark.parametrize("resource,parameter,excluded,first,second,label", SCOPES)
def test_inclusion_and_exclusion_lists_are_comma_separated(
    resource: str, parameter: str, excluded: str, first: str, second: str, label: str
) -> None:
    rows: Final = (
        usage_row(),
        usage_row(
            id="usage-b", customer_org_id="org-b", customer_team_id="team-b", customer_member_id="member-b", spend=10
        ),
    )
    response: Final = client(rows).get(
        f"/management/v1/{resource}/daily/activity", params={**DATES, parameter: f"{first},{second}", excluded: second}
    )
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["total_spend"] == 2


@pytest.mark.parametrize("resource,parameter,excluded,first,second,label", SCOPES)
def test_standard_model_key_and_page_filters_use_the_shared_response_contract(
    resource: str, parameter: str, excluded: str, first: str, second: str, label: str
) -> None:
    rows: Final = (
        usage_row(id="page-a"),
        usage_row(id="page-b", date="2026-01-02", spend=4),
        usage_row(id="other-model", model="model-b", spend=20),
        usage_row(id="other-key", api_key="hash-b", spend=30),
    )
    response: Final = client(rows).get(
        f"/management/v1/{resource}/daily/activity",
        params={**DATES, parameter: first, "model": "model-a", "api_key": "hash-a", "page": 1, "page_size": 1},
    )
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["metadata"]["page"] == 1
    assert body["metadata"]["total_pages"] == 2
    assert body["metadata"]["has_more"] is True
    assert body["metadata"]["total_spend"] == 4
    assert body["results"][0]["date"] == "2026-01-02"
    next_page: Final = client(rows).get(
        f"/management/v1/{resource}/daily/activity",
        params={**DATES, parameter: first, "model": "model-a", "api_key": "hash-a", "page": 2, "page_size": 1},
    )
    assert next_page.status_code == 200, next_page.text
    assert next_page.json()["metadata"]["total_spend"] == 2
    assert next_page.json()["metadata"]["has_more"] is False


@pytest.mark.parametrize("resource,parameter,excluded,first,second,label", SCOPES)
@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
def test_all_hierarchy_daily_activity_routes_are_admin_only(
    resource: str, parameter: str, excluded: str, first: str, second: str, label: str, role: LitellmUserRoles
) -> None:
    response: Final = client(role=role).get(f"/management/v1/{resource}/daily/activity", params=DATES)
    assert response.status_code == 403, response.text


@pytest.mark.parametrize("params", ({}, {"start_date": "invalid", "end_date": "2026-01-03"}, {**DATES, "page": 0}))
def test_shared_helper_rejects_missing_dates_invalid_dates_and_nonpositive_pagination(
    params: Mapping[str, object],
) -> None:
    response: Final = client().get("/management/v1/customer_members/daily/activity", params=params)
    assert response.status_code == 400, response.text


def test_unmatched_entities_return_the_standard_empty_response() -> None:
    response: Final = client((usage_row(),)).get(
        "/management/v1/customer_members/daily/activity", params={**DATES, "customer_member_ids": "missing"}
    )
    assert response.status_code == 200, response.text
    assert response.json()["results"] == []
    assert response.json()["metadata"]["total_spend"] == 0


RANGE: Final = {"start_date": "2024-01-01", "end_date": "2024-01-31"}


@dataclass(frozen=True, slots=True)
class ActivityDatabase(Database):
    organization_id: str = "org-a"
    app_id: str | None = None

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if [self.organization_id] not in args:
            return ()
        if self.app_id is not None and self.app_id not in args:
            return ()
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
                    "entity_id": self.organization_id,
                    "api_key_rolled": 1,
                    "prompt_tokens": 3000,
                    "completion_tokens": 2000,
                    "api_requests": 1000,
                },
            )
        raise AssertionError(f"Unexpected query: {query}")


def test_aggregated_activity_returns_complete_database_totals_as_one_page() -> None:
    response: Final = client(database=ActivityDatabase(Table())).get(
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


@pytest.mark.parametrize("organization_id", ("org,a", " org-a ", "org/a?b=c"))
def test_exact_organization_filter_preserves_the_complete_identifier(organization_id: str) -> None:
    response: Final = client(database=ActivityDatabase(Table(), organization_id=organization_id)).get(
        "/management/v1/customer_organizations/daily/activity/aggregated",
        params={**RANGE, "customer_org_id": organization_id},
    )
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["metadata"]["total_spend"] == 500
    assert set(body["results"][0]["breakdown"]["entities"]) == {organization_id}
    assert body["results"][0]["breakdown"]["entities"][organization_id]["metrics"]["spend"] == 500


@pytest.mark.parametrize("role", (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY))
def test_non_admins_cannot_read_customer_usage(role: LitellmUserRoles) -> None:
    response: Final = client(role=role).get(
        "/management/v1/customer_organizations/daily/activity/aggregated", params=RANGE
    )
    assert response.status_code == 403


@pytest.mark.parametrize(
    "params,status",
    (
        (RANGE, 400),
        ({**RANGE, "customer_org_id": "org-a", "customer_org_ids": "org-b"}, 400),
        ({**RANGE, "customer_org_id": ""}, 422),
        ({**RANGE, "customer_org_ids": ""}, 422),
        ({**RANGE, "customer_org_ids": ",".join(str(index) for index in range(101))}, 422),
        ({"start_date": "2020-01-01", "end_date": "2024-01-01", "customer_org_ids": "org-a"}, 400),
        ({**RANGE, "start_date": "2024-02-01", "customer_org_ids": "org-a"}, 400),
        ({**RANGE, "start_date": "2024-02-30", "customer_org_ids": "org-a"}, 400),
        ({"start_date": "2024-01-01", "customer_org_ids": "org-a"}, 422),
        ({**RANGE, "customer_org_ids": "org-a", "unknown": "true"}, 400),
    ),
)
def test_aggregate_work_is_bounded(params: Mapping[str, str], status: int) -> None:
    response: Final = client(database=ActivityDatabase(Table())).get(
        "/management/v1/customer_organizations/daily/activity/aggregated", params=params
    )
    assert response.status_code == status, response.text


def test_duplicate_filters_are_rejected() -> None:
    response: Final = client(database=ActivityDatabase(Table())).get(
        "/management/v1/customer_organizations/daily/activity/aggregated"
        "?start_date=2024-01-01&start_date=2024-01-02&end_date=2024-01-31&customer_org_ids=org-a"
    )
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("resource,parameter,excluded,first,second,label", SCOPES)
def test_app_filter_keeps_the_customer_scope_and_all_apps_remain_the_default(
    resource: str, parameter: str, excluded: str, first: str, second: str, label: str
) -> None:
    rows: Final = (
        usage_row(app_id="chat", spend=2),
        usage_row(id="search", app_id="search", spend=5),
        usage_row(id="legacy", spend=3),
        usage_row(
            id="other-customer",
            app_id="chat",
            spend=99,
            customer_org_id="org-b",
            customer_team_id="team-b",
            customer_member_id="member-b",
        ),
    )
    http: Final = client(rows)
    path: Final = f"/management/v1/{resource}/daily/activity"
    global_response: Final = http.get(path, params={**DATES, parameter: first})
    selected: Final = http.get(path, params={**DATES, parameter: first, "app_id": "chat"})
    search: Final = http.get(path, params={**DATES, parameter: first, "app_id": "search"})
    assert global_response.status_code == selected.status_code == search.status_code == 200
    assert global_response.json()["metadata"]["total_spend"] == 10
    assert selected.json()["metadata"]["total_spend"] == 2
    assert search.json()["metadata"]["total_spend"] == 5
    assert set(selected.json()["results"][0]["breakdown"]["entities"]) == {first}


@pytest.mark.parametrize("resource,parameter,excluded,first,second,label", SCOPES)
@pytest.mark.parametrize("app_id,expected_status", (("", 422), ("unknown", 404)))
def test_daily_activity_rejects_empty_or_unknown_app_filters(
    resource: str, parameter: str, excluded: str, first: str, second: str, label: str, app_id: str, expected_status: int
) -> None:
    response: Final = client((usage_row(),)).get(
        f"/management/v1/{resource}/daily/activity", params={**DATES, parameter: first, "app_id": app_id}
    )
    assert response.status_code == expected_status


def test_aggregated_customer_usage_applies_the_app_filter_before_rollups() -> None:
    http: Final = client(database=ActivityDatabase(Table(), app_id="chat"))
    path: Final = "/management/v1/customer_organizations/daily/activity/aggregated"
    selected: Final = http.get(path, params={**RANGE, "customer_org_id": "org-a", "app_id": "chat"})
    other_app: Final = http.get(path, params={**RANGE, "customer_org_id": "org-a", "app_id": "search"})
    other_customer: Final = http.get(path, params={**RANGE, "customer_org_id": "org-b", "app_id": "chat"})
    assert selected.status_code == other_app.status_code == other_customer.status_code == 200
    assert selected.json()["metadata"]["total_spend"] == 500
    assert other_app.json()["metadata"]["total_spend"] == other_customer.json()["metadata"]["total_spend"] == 0


@pytest.mark.parametrize("app_id,expected_status", (("", 422), ("unknown", 404)))
def test_aggregated_customer_usage_rejects_empty_or_unknown_apps(app_id: str, expected_status: int) -> None:
    response: Final = client(database=ActivityDatabase(Table())).get(
        "/management/v1/customer_organizations/daily/activity/aggregated",
        params={**RANGE, "customer_org_id": "org-a", "app_id": app_id},
    )
    assert response.status_code == expected_status


GRANTEE_PATHS: Final = (
    "/management/v1/customer_organizations/daily/activity/aggregated",
    "/management/v1/customer_organizations/daily/activity",
    "/management/v1/customer_teams/daily/activity",
    "/management/v1/customer_members/daily/activity",
)


def granted(route: str) -> Database:
    """User "admin" holds the chat app through their own, their team's or their organization's object permission."""
    return Database(
        Table(
            (
                usage_row(app_id="chat", spend=2),
                usage_row(id="search", app_id="search", spend=5, customer_org_id="org-b", api_key="hash-b"),
                usage_row(id="legacy", spend=3),
            )
        ),
        litellm_usertable=Table(
            (
                Record(
                    user_id="admin",
                    teams=["owners"] if route == "team" else [],
                    object_permission_id="grant" if route == "user" else None,
                ),
            )
        ),
        litellm_teamtable=Table(
            (Record(team_id="owners", object_permission_id="grant" if route == "team" else None),)
        ),
        litellm_organizationtable=Table(
            (Record(organization_id="division", object_permission_id="grant" if route == "organization" else None),)
        ),
        litellm_organizationmembership=Table(
            (Record(organization_id="division", user_id="admin"),) if route == "organization" else ()
        ),
        litellm_objectpermissiontable=Table((Record(object_permission_id="grant", apps=["chat"]),)),
    )


@pytest.mark.parametrize("route", ("user", "team", "organization"))
@pytest.mark.parametrize("path", GRANTEE_PATHS[1:])
def test_a_grantee_reads_usage_for_a_granted_app_only(path: str, route: str) -> None:
    http: Final = client(role=LitellmUserRoles.INTERNAL_USER, database=granted(route))
    own: Final = http.get(path, params={**DATES, "app_id": "chat"})
    assert own.status_code == 200, own.text
    assert own.json()["metadata"]["total_spend"] == 2
    assert http.get(path, params={**DATES, "app_id": "search"}).status_code == 403
    assert http.get(path, params=DATES).status_code == 403


@pytest.mark.parametrize("route", ("user", "team", "organization"))
def test_a_grantee_cannot_aggregate_usage_outside_a_granted_app(route: str) -> None:
    http: Final = client(role=LitellmUserRoles.INTERNAL_USER, database=granted(route))
    path: Final = GRANTEE_PATHS[0]
    scope: Final = {**DATES, "customer_org_id": "org-b"}
    assert http.get(path, params={**scope, "app_id": "search"}).status_code == 403
    assert http.get(path, params=scope).status_code == 403


@pytest.mark.parametrize("path", GRANTEE_PATHS[1:])
def test_a_grantee_cannot_reach_another_apps_usage_by_naming_its_key(path: str) -> None:
    http: Final = client(role=LitellmUserRoles.INTERNAL_USER, database=granted("user"))
    response: Final = http.get(path, params={**DATES, "app_id": "chat", "api_key": "hash-b"})
    assert response.status_code == 200, response.text
    assert response.json()["metadata"]["total_spend"] == 0
