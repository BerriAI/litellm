from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.repositories.accounting_delivery_repository import DeliveryProjection, LedgerTransaction
from litellm.types.proxy.auth.customer_hierarchy import CustomerIdentityAssertion

_FIELDS: Final = TypeAdapter(Mapping[str, object])
_MEMBER: Final = """SELECT m.customer_member_id, t.customer_team_id, o.customer_org_id,
    o.alias AS customer_org_alias
FROM "LiteLLM_CustomerMemberTable" m
JOIN "LiteLLM_CustomerTeamTable" t ON t.customer_team_id = m.parent_team_id
JOIN "LiteLLM_CustomerOrganizationTable" o ON o.customer_org_id = t.parent_org_id
WHERE m.customer_member_id = $1::text"""
_ORG_BY_ID: Final = """SELECT customer_org_id, alias AS customer_org_alias
FROM "LiteLLM_CustomerOrganizationTable" WHERE customer_org_id = $1::text LIMIT 2"""
_ORG_BY_ALIAS: Final = """SELECT customer_org_id, alias AS customer_org_alias
FROM "LiteLLM_CustomerOrganizationTable" WHERE alias = $1::text LIMIT 2"""


class _Customer(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_org_id: str
    customer_org_alias: str | None = None
    customer_team_id: str | None = None
    customer_member_id: str | None = None

    def entities(self) -> Mapping[str, str | None]:
        return {
            "customer_org_id": self.customer_org_id,
            "customer_team_id": self.customer_team_id,
            "customer_member_id": self.customer_member_id,
        }


def _mapping(value: object) -> Mapping[str, object]:
    if isinstance(value, str):
        return _FIELDS.validate_json(value)
    return _FIELDS.validate_python(value) if isinstance(value, Mapping) else {}


async def _customer(transaction: LedgerTransaction, assertion: CustomerIdentityAssertion) -> _Customer | None:
    if assertion.customer_member_id is not None:
        members: Final = await transaction.query_raw(_MEMBER, assertion.customer_member_id)
        if members:
            member: Final = _Customer.model_validate(members[0])
            if assertion.key_customer_org_id is not None:
                if member.customer_org_id == assertion.key_customer_org_id:
                    return member
            elif assertion.customer_org_alias is None or member.customer_org_alias == assertion.customer_org_alias:
                return member
            else:
                return None
    organization: Final = assertion.key_customer_org_id or assertion.customer_org_alias
    if organization is None:
        return None
    rows: Final = await transaction.query_raw(
        _ORG_BY_ID if assertion.key_customer_org_id is not None else _ORG_BY_ALIAS, organization
    )
    return _Customer.model_validate(rows[0]) if len(rows) == 1 else None


async def resolve_customer_delivery(transaction: LedgerTransaction, delivery: DeliveryProjection) -> DeliveryProjection:
    metadata: Final = _mapping(delivery.row.get("metadata"))
    asserted: Final = metadata.get("user_api_key_asserted_customer")
    if asserted is None:
        return delivery
    entities: Final = _mapping(metadata.get("accounting_entities"))
    if isinstance(entities.get("customer_org_id"), str):
        return delivery
    customer: Final = await _customer(transaction, CustomerIdentityAssertion.model_validate(asserted))
    if customer is None:
        return delivery
    return DeliveryProjection(
        {
            **delivery.row,
            "metadata": {**metadata, "accounting_entities": {**entities, **customer.entities()}},
        }
    )
