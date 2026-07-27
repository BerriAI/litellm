"""Live e2e coverage for the budget, customer/end-user, user-info and
organization-membership management routes.

Each test creates its resources under unique ids (deleted on teardown) and
asserts the recorded state the route promises: the budget table reflects a
create/update, a customer round-trips through the info route and disappears after
delete, /user/info echoes what /user/new stored, and an added org member shows up
both in the add response and in /organization/info. The budget/new admin gate is
proven by driving the route under a non-admin key and asserting it is refused.

Response bodies validate into local pydantic models (only the fields asserted are
modelled) so a shape change fails here instead of passing vacuously.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import pytest
from pydantic import BaseModel, RootModel

from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from management_client import ManagementClient
from models import KeyGenerateBody, OrgInfoParams, OrgNewBody, UserNewBody

pytestmark = pytest.mark.e2e


def _poll[T](client: ManagementClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + client.proxy.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(client.proxy.poll_interval)
    pytest.fail(failure)


# ---------- budget ----------


class BudgetNewBody(BaseModel):
    max_budget: float
    soft_budget: float | None = None
    budget_duration: str | None = None


class BudgetNewResponse(BaseModel):
    budget_id: str


class BudgetUpdateBody(BaseModel):
    budget_id: str
    max_budget: float


class BudgetInfoBody(BaseModel):
    budgets: list[str]


class BudgetRow(BaseModel):
    budget_id: str | None = None
    max_budget: float | None = None
    soft_budget: float | None = None


class BudgetInfoResponse(RootModel[list[BudgetRow]]):
    pass


class BudgetListResponse(RootModel[list[BudgetRow]]):
    """GET /budget/list answers with a bare array of budget rows, not an object
    wrapping them. Read the rows off .root."""


class BudgetDeleteBody(BaseModel):
    id: str


def _delete_budget(client: ManagementClient, budget_id: str) -> None:
    _ = client.proxy.transport.post(
        "/budget/delete",
        headers=client.proxy.transport.master,
        json=BudgetDeleteBody(id=budget_id),
        response_type=NoBody,
    )


def _create_budget(client: ManagementClient, resources: ResourceManager, body: BudgetNewBody) -> str:
    budget_id = unwrap(
        client.proxy.transport.post(
            "/budget/new",
            headers=client.proxy.transport.master,
            json=body,
            response_type=BudgetNewResponse,
        )
    ).budget_id
    resources.defer(lambda: _delete_budget(client, budget_id))
    return budget_id


def _budget_rows(client: ManagementClient, budget_id: str) -> tuple[BudgetRow, ...]:
    return tuple(
        unwrap(
            client.proxy.transport.post(
                "/budget/info",
                headers=client.proxy.transport.master,
                json=BudgetInfoBody(budgets=[budget_id]),
                response_type=BudgetInfoResponse,
            )
        ).root
    )


def _budget_list_ids(client: ManagementClient) -> tuple[str, ...]:
    return tuple(
        row.budget_id
        for row in unwrap(
            client.proxy.transport.get(
                "/budget/list",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=BudgetListResponse,
            )
        ).root
        if row.budget_id is not None
    )


_INITIAL_MAX_BUDGET = 5.5
_UPDATED_MAX_BUDGET = 91.25


class TestBudgetManagement:
    @pytest.mark.covers("mgmt.budget.list.happy_path")
    def test_created_budget_appears_in_budget_list(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        budget_id = _create_budget(client, resources, BudgetNewBody(max_budget=_INITIAL_MAX_BUDGET))

        _ = _poll(
            client,
            lambda: budget_id if budget_id in _budget_list_ids(client) else None,
            f"/budget/list never included the created budget {budget_id}",
        )

    @pytest.mark.covers("mgmt.budget.update.persists")
    def test_update_max_budget_persists_to_budget_info(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        budget_id = _create_budget(client, resources, BudgetNewBody(max_budget=_INITIAL_MAX_BUDGET))

        rows = _budget_rows(client, budget_id)
        assert rows, f"/budget/info returned nothing for the freshly created budget {budget_id}"
        initial = rows[0].max_budget
        assert initial is not None and math.isclose(initial, _INITIAL_MAX_BUDGET, rel_tol=1e-9), (
            f"/budget/info reports max_budget {initial}, created with {_INITIAL_MAX_BUDGET}"
        )

        _ = unwrap(
            client.proxy.transport.post(
                "/budget/update",
                headers=client.proxy.transport.master,
                json=BudgetUpdateBody(budget_id=budget_id, max_budget=_UPDATED_MAX_BUDGET),
                response_type=NoBody,
            )
        )

        def updated() -> BudgetRow | None:
            row = next((r for r in _budget_rows(client, budget_id) if r.budget_id == budget_id), None)
            if row is None or row.max_budget is None:
                return None
            return row if math.isclose(row.max_budget, _UPDATED_MAX_BUDGET, rel_tol=1e-9) else None

        _ = _poll(
            client,
            updated,
            f"/budget/info never reported max_budget {_UPDATED_MAX_BUDGET} for {budget_id} after /budget/update",
        )

    @pytest.mark.covers("mgmt.budget.new.admin_only")
    def test_new_is_refused_for_a_non_admin_key(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = client.proxy.generate_key(KeyGenerateBody())
        resources.defer(lambda: client.proxy.delete_key(key))

        outcome = client.proxy.transport.send(
            "/budget/new",
            headers=client.proxy.transport.bearer(key),
            json=BudgetNewBody(max_budget=1.0),
        )

        assert outcome.status_code in (401, 403), (
            f"non-admin key POSTing /budget/new must be refused 401/403, got "
            f"{outcome.status_code}: {outcome.body[:300]}"
        )
        assert "proxy admin" in outcome.body.lower() or "not allowed" in outcome.body.lower(), (
            f"/budget/new denial body must name the admin-only gate, got: {outcome.body[:300]}"
        )


# ---------- customer / end-user ----------


class CustomerNewBody(BaseModel):
    user_id: str
    max_budget: float | None = None


class CustomerNewResponse(BaseModel):
    user_id: str


class CustomerInfoParams(BaseModel):
    end_user_id: str


class CustomerInfoResponse(BaseModel):
    user_id: str


class CustomerDeleteBody(BaseModel):
    user_ids: list[str]


class CustomerDeleteResponse(BaseModel):
    deleted_customers: int


def _create_customer(
    client: ManagementClient, resources: ResourceManager, route: str, body: CustomerNewBody
) -> str:
    user_id = unwrap(
        client.proxy.transport.post(
            route,
            headers=client.proxy.transport.master,
            json=body,
            response_type=CustomerNewResponse,
        )
    ).user_id
    resources.defer(lambda: client.proxy.delete_customers([user_id]))
    return user_id


def _customer_info(client: ManagementClient, route: str, user_id: str) -> CustomerInfoResponse:
    return unwrap(
        client.proxy.transport.get(
            route,
            headers=client.proxy.transport.master,
            params=CustomerInfoParams(end_user_id=user_id),
            response_type=CustomerInfoResponse,
        )
    )


class TestCustomerManagement:
    @pytest.mark.covers("mgmt.customer.new.happy_path")
    def test_new_persists_to_customer_info(self, client: ManagementClient, resources: ResourceManager) -> None:
        customer_id = f"e2e-mgmt-cust-{unique_marker()}"
        created = _create_customer(
            client, resources, "/customer/new", CustomerNewBody(user_id=customer_id, max_budget=7.0)
        )
        assert created == customer_id, f"/customer/new echoed user_id {created!r}, created {customer_id!r}"

        info = _customer_info(client, "/customer/info", customer_id)
        assert info.user_id == customer_id, (
            f"/customer/info reports user_id {info.user_id!r} for the created customer {customer_id!r}"
        )

    @pytest.mark.covers("mgmt.customer.delete.persists")
    def test_delete_removes_the_customer(self, client: ManagementClient, resources: ResourceManager) -> None:
        """The teardown's deferred delete fires again on the already-deleted customer
        by design: it is the safety net if this test fails before the in-body delete,
        and a repeat /customer/delete is absorbed by the warn-only teardown."""
        customer_id = f"e2e-mgmt-cust-{unique_marker()}"
        _ = _create_customer(client, resources, "/customer/new", CustomerNewBody(user_id=customer_id, max_budget=3.0))

        assert _customer_info(client, "/customer/info", customer_id).user_id == customer_id, (
            f"customer {customer_id} was not readable before deletion"
        )

        deleted = unwrap(
            client.proxy.transport.post(
                "/customer/delete",
                headers=client.proxy.transport.master,
                json=CustomerDeleteBody(user_ids=[customer_id]),
                response_type=CustomerDeleteResponse,
            )
        ).deleted_customers
        assert deleted == 1, f"/customer/delete reported {deleted} rows removed for one customer"

        def gone() -> bool | None:
            return True if client.proxy.transport.probe(
                "/customer/info", params=CustomerInfoParams(end_user_id=customer_id)
            ).status_code == 404 else None

        _ = _poll(client, gone, f"customer {customer_id} still resolved on /customer/info after /customer/delete")

    @pytest.mark.covers("mgmt.end_user.new.happy_path")
    def test_end_user_new_persists_to_end_user_info(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        end_user_id = f"e2e-mgmt-euser-{unique_marker()}"
        created = _create_customer(client, resources, "/end_user/new", CustomerNewBody(user_id=end_user_id))
        assert created == end_user_id, f"/end_user/new echoed user_id {created!r}, created {end_user_id!r}"

        info = _customer_info(client, "/end_user/info", end_user_id)
        assert info.user_id == end_user_id, (
            f"/end_user/info reports user_id {info.user_id!r} for the created end user {end_user_id!r}"
        )


# ---------- user info ----------


class TestUserManagement:
    @pytest.mark.covers("mgmt.user.info.happy_path")
    def test_new_user_is_readable_via_user_info(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        email = f"e2e-mgmt-{unique_marker()}@example.com"
        user_id = client.create_user(UserNewBody(user_email=email, user_role="internal_user"))
        resources.defer(lambda: client.delete_user(user_id))

        info = client.user_info(user_id).user_info
        assert info.user_id == user_id, f"/user/info reports user_id {info.user_id!r}, created {user_id!r}"
        assert info.user_email == email, f"/user/info reports user_email {info.user_email!r}, configured {email!r}"
        assert info.user_role == "internal_user", (
            f"/user/info reports user_role {info.user_role!r}, configured 'internal_user'"
        )


# ---------- organization membership ----------


class OrgMemberEntry(BaseModel):
    role: str
    user_id: str


class OrgMemberAddBody(BaseModel):
    organization_id: str
    member: OrgMemberEntry


class OrgMembershipRow(BaseModel):
    user_id: str
    organization_id: str | None = None


class OrgMemberAddResponse(BaseModel):
    organization_id: str
    updated_organization_memberships: list[OrgMembershipRow]


class OrgInfoMembersResponse(BaseModel):
    members: list[OrgMembershipRow] = []


class TestOrganizationMembership:
    @pytest.mark.covers("mgmt.organization.member_add.happy_path")
    def test_member_add_records_membership(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        org_id = client.create_org(OrgNewBody(organization_alias=f"e2e-mgmt-org-{unique_marker()}"))
        resources.defer(lambda: client.delete_org(org_id))

        user_id = client.create_user(
            UserNewBody(user_email=f"e2e-mgmt-{unique_marker()}@example.com", user_role="internal_user")
        )
        resources.defer(lambda: client.delete_user(user_id))

        added = unwrap(
            client.proxy.transport.post(
                "/organization/member_add",
                headers=client.proxy.transport.master,
                json=OrgMemberAddBody(
                    organization_id=org_id,
                    member=OrgMemberEntry(role="internal_user", user_id=user_id),
                ),
                response_type=OrgMemberAddResponse,
            )
        )
        assert added.organization_id == org_id, (
            f"/organization/member_add echoed organization_id {added.organization_id!r}, added to {org_id!r}"
        )
        assert any(
            row.user_id == user_id and row.organization_id == org_id
            for row in added.updated_organization_memberships
        ), (
            f"/organization/member_add response does not record {user_id} in org {org_id}: "
            f"{added.updated_organization_memberships}"
        )

        def listed() -> bool | None:
            members = unwrap(
                client.proxy.transport.get(
                    "/organization/info",
                    headers=client.proxy.transport.master,
                    params=OrgInfoParams(organization_id=org_id),
                    response_type=OrgInfoMembersResponse,
                )
            ).members
            return True if any(member.user_id == user_id for member in members) else None

        _ = _poll(
            client,
            listed,
            f"/organization/info never listed member {user_id} in org {org_id} after /organization/member_add",
        )
