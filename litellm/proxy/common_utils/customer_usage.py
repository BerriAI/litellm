from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from litellm.proxy.management_helpers.customer_hierarchy import effective_member_budget
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget

CustomerIdentifier: TypeAlias = Annotated[str, Field(strict=True, min_length=1, max_length=256)]
_IDENTITY_FIELDS: Final = ("customer_org_id", "customer_team_id", "customer_member_id")
CUSTOMER_USAGE_METADATA_KEY: Final = "user_api_key_customer_usage"
_FIELDS: Final = TypeAdapter(Mapping[str, object])


class CustomerUsageInput(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_org_id: CustomerIdentifier | None = None
    customer_team_id: CustomerIdentifier | None = None
    customer_member_id: CustomerIdentifier | None = None


class CustomerUsageAttribution(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_org_id: CustomerIdentifier
    customer_team_id: CustomerIdentifier | None = None
    customer_member_id: CustomerIdentifier | None = None


@dataclass(frozen=True, slots=True)
class ResolvedCustomerUsage:
    attribution: CustomerUsageAttribution
    model_scopes: tuple[tuple[str, ...], ...]


class _HierarchyBudget(Budget):
    model_config = ConfigDict(frozen=True, from_attributes=True)


class _Organization(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    budget: _HierarchyBudget | None = None


class _Team(_Organization):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    parent_org_id: str


class _Member(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    parent_team_ids: tuple[str, ...] = ()
    budget: _HierarchyBudget | None = None
    budget_override: _HierarchyBudget | None = None
    budget_override_expires_at: datetime | None = None


def _budget_models(budget: Budget | None) -> tuple[str, ...]:
    if budget is None:
        return ()
    return budget.allowed_models or ()


def _model_scopes(
    organization: _Organization, team: _Team | None, member: _Member | None
) -> tuple[tuple[str, ...], ...]:
    member_budget: Final = (
        effective_member_budget(
            member.budget, member.budget_override, member.budget_override_expires_at, datetime.now(timezone.utc)
        )
        if member is not None
        else None
    )
    scopes: Final = (
        _budget_models(organization.budget),
        _budget_models(team.budget if team is not None else None),
        _budget_models(member_budget),
    )
    return tuple(scope for scope in scopes if scope)


class _KeyMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, from_attributes=True)

    metadata: object = None


class CustomerUsageDatabase(Protocol):
    @property
    def litellm_customerorganizationtable(self) -> TableActions[object]: ...

    @property
    def litellm_customerteamtable(self) -> TableActions[object]: ...

    @property
    def litellm_customermembertable(self) -> TableActions[object]: ...


class CustomerUsageError(str, Enum):
    INVALID = "Customer usage metadata contains invalid identifiers."
    MISMATCH = "Customer usage identifiers must belong to the same organization and team."
    FORBIDDEN = "Customer usage must match the customer organization assigned to this API key."
    UNBOUND = "Assign a customer organization to the API key before tracking customer usage."
    NOT_FOUND = "The customer usage organization, team, or member does not exist."
    UNAVAILABLE = "Customer usage tracking requires a database and spend accounting ledger."
    KEY_BINDING_FORBIDDEN = "Only proxy admins can change a key's customer organization, team, or member binding."


@dataclass(frozen=True, slots=True)
class CustomerUsageFailure:
    error: CustomerUsageError


def normalize_customer_key_metadata(
    metadata: Mapping[str, object] | None,
    existing_metadata: Mapping[str, object] | None,
    *,
    administrator: bool,
) -> Mapping[str, object] | CustomerUsageFailure | None:
    if administrator:
        return metadata
    stored: Final = existing_metadata or {}
    requested: Final = metadata or {}
    if any(name in requested and requested[name] != stored.get(name) for name in _IDENTITY_FIELDS):
        return CustomerUsageFailure(CustomerUsageError.KEY_BINDING_FORBIDDEN)
    bindings: Final = {name: stored[name] for name in _IDENTITY_FIELDS if stored.get(name) is not None}
    if not bindings:
        return metadata
    return {**requested, **bindings}


def _input(metadata: Mapping[str, object], binding: Mapping[str, object]) -> CustomerUsageInput | CustomerUsageFailure:
    if any(
        name in metadata and name in binding and metadata[name] != binding[name]
        for name in _IDENTITY_FIELDS
        if binding.get(name) is not None
    ):
        return CustomerUsageFailure(CustomerUsageError.FORBIDDEN)
    try:
        return CustomerUsageInput.model_validate(
            {**metadata, **{name: binding[name] for name in _IDENTITY_FIELDS if binding.get(name) is not None}}
        )
    except ValidationError:
        return CustomerUsageFailure(CustomerUsageError.INVALID)


def _hierarchy_failure(error: CustomerUsageError, *, administrator: bool) -> CustomerUsageFailure:
    if administrator:
        return CustomerUsageFailure(error)
    return CustomerUsageFailure(CustomerUsageError.FORBIDDEN)


async def resolve_customer_usage(
    *,
    metadata: Mapping[str, object],
    key_metadata: Mapping[str, object],
    administrator: bool,
    database: CustomerUsageDatabase | None,
    ledger_enabled: bool,
) -> ResolvedCustomerUsage | CustomerUsageFailure | None:
    if not customer_usage_requested(metadata, key_metadata):
        return None
    candidate: Final = _input(metadata, key_metadata)
    if isinstance(candidate, CustomerUsageFailure):
        return candidate
    if not administrator and key_metadata.get("customer_org_id") is None:
        return CustomerUsageFailure(CustomerUsageError.UNBOUND)
    if database is None or not ledger_enabled:
        return CustomerUsageFailure(CustomerUsageError.UNAVAILABLE)
    member: Final = (
        await database.litellm_customermembertable.find_unique(
            where={"customer_member_id": candidate.customer_member_id},
            include={"budget": True, "budget_override": True},
        )
        if candidate.customer_member_id is not None
        else None
    )
    if candidate.customer_member_id is not None and member is None:
        return _hierarchy_failure(CustomerUsageError.NOT_FOUND, administrator=administrator)
    member_record: Final = _Member.model_validate(member) if member is not None else None
    member_teams: Final = member_record.parent_team_ids if member_record is not None else ()
    if candidate.customer_team_id is not None and member_teams and candidate.customer_team_id not in member_teams:
        return _hierarchy_failure(CustomerUsageError.MISMATCH, administrator=administrator)
    team_id: Final = candidate.customer_team_id or (member_teams[0] if member_teams else None)
    team: Final = (
        await database.litellm_customerteamtable.find_unique(
            where={"customer_team_id": team_id}, include={"budget": True}
        )
        if team_id is not None
        else None
    )
    if team_id is not None and team is None:
        return _hierarchy_failure(CustomerUsageError.NOT_FOUND, administrator=administrator)
    team_record: Final = _Team.model_validate(team) if team is not None else None
    team_org: Final = team_record.parent_org_id if team_record is not None else None
    if candidate.customer_org_id is not None and team_org is not None and candidate.customer_org_id != team_org:
        return _hierarchy_failure(CustomerUsageError.MISMATCH, administrator=administrator)
    organization_id: Final = candidate.customer_org_id or team_org
    if organization_id is None:
        return CustomerUsageFailure(CustomerUsageError.INVALID)
    organization: Final = await database.litellm_customerorganizationtable.find_unique(
        where={"customer_org_id": organization_id}, include={"budget": True}
    )
    if organization is None:
        return _hierarchy_failure(CustomerUsageError.NOT_FOUND, administrator=administrator)
    return ResolvedCustomerUsage(
        attribution=CustomerUsageAttribution(
            customer_org_id=organization_id,
            customer_team_id=team_id,
            customer_member_id=candidate.customer_member_id,
        ),
        model_scopes=_model_scopes(_Organization.model_validate(organization), team_record, member_record),
    )


def customer_usage_requested(metadata: Mapping[str, object], key_metadata: Mapping[str, object]) -> bool:
    return any(metadata.get(name) is not None or key_metadata.get(name) is not None for name in _IDENTITY_FIELDS)


def customer_usage_fields(value: object) -> Mapping[str, object]:
    return _FIELDS.validate_python(value) if isinstance(value, Mapping) else {}


def customer_usage_key_fields(key: object) -> Mapping[str, object]:
    return customer_usage_fields(_KeyMetadata.model_validate(key).metadata)


def customer_usage_http_status(error: CustomerUsageError) -> int:
    if error == CustomerUsageError.UNAVAILABLE:
        return 503
    if error in (CustomerUsageError.UNBOUND, CustomerUsageError.FORBIDDEN, CustomerUsageError.KEY_BINDING_FORBIDDEN):
        return 403
    return 400
