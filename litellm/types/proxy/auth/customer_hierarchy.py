from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.types.integrations.slack_alerting import Litellm_EntityType

CustomerScopeKey: TypeAlias = Literal["customer_org", "customer_team", "customer_member"]


@dataclass(frozen=True, slots=True)
class CustomerRequestIdentity:
    """What a request's headers said about the customer it runs for: the member id, the organization alias, or both."""

    customer_member_id: str | None = None
    customer_org_alias: str | None = None


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
    """The levels a request is charged to. A request that named only an organization alias has no team or member."""

    model_config = ConfigDict(frozen=True)

    customer_org_id: str
    customer_team_id: str | None = None
    customer_member_id: str | None = None
    scopes: tuple[CustomerScopeAuth, ...]


@dataclass(frozen=True, slots=True)
class CustomerHierarchyIds:
    """The rows a request's spend is charged to, as the request metadata recorded them. Team and member are absent
    when the request named only an organization."""

    customer_org_id: str
    customer_team_id: str | None = None
    customer_member_id: str | None = None

    def entities(self) -> Mapping[str, str | None]:
        return {
            "customer_member_id": self.customer_member_id,
            "customer_team_id": self.customer_team_id,
            "customer_org_id": self.customer_org_id,
        }

    def spend_entities(self) -> tuple[tuple[Litellm_EntityType, str], ...]:
        levels: Final = (
            (Litellm_EntityType.CUSTOMER_ORG, self.customer_org_id),
            (Litellm_EntityType.CUSTOMER_TEAM, self.customer_team_id),
            (Litellm_EntityType.CUSTOMER_MEMBER, self.customer_member_id),
        )
        return tuple((entity_type, entity_id) for entity_type, entity_id in levels if entity_id)


_FIELDS: Final = TypeAdapter(Mapping[str, object])
_NO_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return _FIELDS.validate_python(value)
    except ValidationError:
        return _NO_FIELDS


def _stamped_metadata(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    """The proxy stamps auth ids into ``litellm_metadata`` on routes whose body owns ``metadata``, else into ``metadata``."""
    buckets: Final = tuple(_mapping(litellm_params.get(name)) for name in ("litellm_metadata", "metadata"))
    return next((bucket for bucket in buckets if "user_api_key_customer_org_id" in bucket), _NO_FIELDS)


def _stamped_id(metadata: Mapping[str, object], level: str) -> str | None:
    value: Final = metadata.get(f"user_api_key_customer_{level}_id")
    return value if isinstance(value, str) and value else None


def customer_hierarchy_ids_from_request(kwargs: Mapping[str, object]) -> CustomerHierarchyIds | None:
    """The ids auth stamped into the request's proxy metadata, or None when the request named no customer."""
    metadata: Final = _stamped_metadata(_mapping(kwargs.get("litellm_params")))
    org_id: Final = _stamped_id(metadata, "org")
    if org_id is None:
        return None
    return CustomerHierarchyIds(
        customer_org_id=org_id,
        customer_team_id=_stamped_id(metadata, "team"),
        customer_member_id=_stamped_id(metadata, "member"),
    )
