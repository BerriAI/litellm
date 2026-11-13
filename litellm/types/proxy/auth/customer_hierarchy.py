from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
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
    customer_team_id: str | None
    customer_org_id: str | None

    def entities(self) -> Mapping[str, str]:
        return {
            key: value
            for key, value in (
                ("customer_member_id", self.customer_member_id),
                ("customer_team_id", self.customer_team_id),
                ("customer_org_id", self.customer_org_id),
            )
            if value is not None
        }

    def spend_entities(self) -> tuple[tuple[Litellm_EntityType, str], ...]:
        return tuple(
            (entity_type, entity_id)
            for entity_type, entity_id in (
                (Litellm_EntityType.CUSTOMER_ORG, self.customer_org_id),
                (Litellm_EntityType.CUSTOMER_TEAM, self.customer_team_id),
                (Litellm_EntityType.CUSTOMER_MEMBER, self.customer_member_id),
            )
            if entity_id is not None
        )


_FIELDS: Final = TypeAdapter(Mapping[str, object])


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return _FIELDS.validate_python(value)
    except ValidationError:
        return {}


def customer_hierarchy_ids_from_request(kwargs: Mapping[str, object]) -> CustomerHierarchyIds | None:
    """The ids auth stamped into ``litellm_params.metadata``, or None when the request named no customer member."""
    metadata: Final = _mapping(get_litellm_metadata_from_kwargs(dict(kwargs))) or _mapping(
        _mapping(kwargs.get("standard_logging_object")).get("metadata")
    )
    ids: Final = tuple(metadata.get(f"user_api_key_customer_{level}_id") for level in ("member", "team", "org"))
    member_id, team_id, org_id = ids
    if not isinstance(member_id, str) or not member_id:
        return None
    return CustomerHierarchyIds(
        member_id,
        team_id if isinstance(team_id, str) and team_id else None,
        org_id if isinstance(org_id, str) and org_id else None,
    )
