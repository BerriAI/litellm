from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.types.integrations.slack_alerting import Litellm_EntityType

CustomerScopeKey: TypeAlias = Literal["customer_org", "customer_team", "customer_member"]


class CustomerScopeAuth(BaseModel):
    """One level of the customer hierarchy a request runs under, with the limits its budget row grants."""

    model_config = ConfigDict(frozen=True)

    key: CustomerScopeKey
    entity_id: str
    spend: float = 0.0
    max_budget: float | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    allowed_models: tuple[str, ...] | None = None
    model_tpm_limit: Mapping[str, int] | None = None
    model_rpm_limit: Mapping[str, int] | None = None


class CustomerHierarchyAuth(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_member_id: str
    customer_team_id: str
    customer_org_id: str
    scopes: tuple[CustomerScopeAuth, ...]


@dataclass(frozen=True, slots=True)
class CustomerHierarchyIds:
    """The three rows a request's spend is charged to, as the request metadata recorded them."""

    customer_member_id: str
    customer_team_id: str
    customer_org_id: str

    def entities(self) -> Mapping[str, str]:
        return {
            "customer_member_id": self.customer_member_id,
            "customer_team_id": self.customer_team_id,
            "customer_org_id": self.customer_org_id,
        }

    def spend_entities(self) -> tuple[tuple[Litellm_EntityType, str], ...]:
        return (
            (Litellm_EntityType.CUSTOMER_ORG, self.customer_org_id),
            (Litellm_EntityType.CUSTOMER_TEAM, self.customer_team_id),
            (Litellm_EntityType.CUSTOMER_MEMBER, self.customer_member_id),
        )


_FIELDS: Final = TypeAdapter(Mapping[str, object])


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return _FIELDS.validate_python(value)
    except ValidationError:
        return {}


def _stamped_metadata(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    """The proxy stamps auth ids into ``litellm_metadata`` on routes whose body owns ``metadata``, else into ``metadata``."""
    buckets: Final = tuple(_mapping(litellm_params.get(name)) for name in ("litellm_metadata", "metadata"))
    return next((bucket for bucket in buckets if "user_api_key_customer_member_id" in bucket), {})


def customer_hierarchy_ids_from_request(kwargs: Mapping[str, object]) -> CustomerHierarchyIds | None:
    """The ids auth stamped into the request's proxy metadata, or None when the request named no customer member."""
    metadata: Final = _stamped_metadata(_mapping(kwargs.get("litellm_params")))
    ids: Final = tuple(metadata.get(f"user_api_key_customer_{level}_id") for level in ("member", "team", "org"))
    if not all(isinstance(value, str) and value for value in ids):
        return None
    member_id, team_id, org_id = ids
    return CustomerHierarchyIds(str(member_id), str(team_id), str(org_id))
