from typing import Final

from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from proxy_client import ProxyClient
from pydantic import BaseModel, ConfigDict, Field
from setup.wire import CustomerHeaders


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_budget: float = 1_000_000
    budget_duration: str = "1d"


class OrganizationBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_org_id: str
    alias: str
    budget: Budget = Field(default_factory=Budget)


class TeamBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_team_id: str
    parent_org_id: str
    budget: Budget = Field(default_factory=Budget)


class MemberBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_member_id: str
    parent_team_ids: tuple[str, ...]
    budget_override: Budget = Field(default_factory=Budget)


def _create(proxy: ProxyClient, resources: ResourceManager, collection: str, identifier: str, body: BaseModel) -> None:
    path: Final = f"/management/v1/{collection}"
    _ = unwrap(proxy.transport.post(path, headers=proxy.transport.master, json=body, response_type=NoBody))
    resources.defer(
        lambda: unwrap(
            proxy.transport.delete(
                f"{path}/{identifier}", headers=proxy.transport.master, json=NoBody(), response_type=NoBody
            )
        )
    )


def provision_customer_hierarchy(proxy: ProxyClient, resources: ResourceManager, marker: str) -> CustomerHeaders:
    customer: Final = CustomerHeaders(
        organization=f"perf-org-{marker}", team=f"perf-team-{marker}", member=f"perf-member-{marker}"
    )
    _create(
        proxy,
        resources,
        "customer_organizations",
        customer.organization,
        OrganizationBody(customer_org_id=customer.organization, alias=customer.organization),
    )
    _create(
        proxy,
        resources,
        "customer_teams",
        customer.team,
        TeamBody(customer_team_id=customer.team, parent_org_id=customer.organization),
    )
    _create(
        proxy,
        resources,
        "customer_members",
        customer.member,
        MemberBody(customer_member_id=customer.member, parent_team_ids=(customer.team,)),
    )
    return customer
