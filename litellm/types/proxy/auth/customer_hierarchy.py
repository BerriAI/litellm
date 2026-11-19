from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationError,
    model_serializer,
)

from litellm.models.budget import BudgetLimit
from litellm.types.integrations.slack_alerting import Litellm_EntityType
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    CapPolicy,
)

CustomerScopeKey: TypeAlias = Literal[
    "customer_org",
    "customer_team",
    "customer_member",
    "app",
    "app_customer_org",
    "app_customer_member",
]


@dataclass(frozen=True, slots=True)
class CustomerRequestIdentity:
    """What a request's headers said about the customer it runs for: the member id, the organization alias, or both."""

    customer_member_id: str | None = None
    customer_org_alias: str | None = None
    customer_team_id: str | None = None


class AppDurationLimit(BaseModel):
    model_config = ConfigDict(frozen=True)

    budget_duration: str
    max_budget: float = Field(ge=0, allow_inf_nan=False)
    reset_at: datetime | None = None


def app_budget_limits(
    budget: Budget | None,
) -> tuple[BudgetLimit | AppDurationLimit, ...]:
    """Charge each duration once while applying both caps and the explicit window's clock."""
    if budget is None:
        return ()
    windows: Final = budget.budget_limits or ()
    if budget.budget_duration is None or budget.max_budget is None:
        return windows
    if any(window.budget_duration == budget.budget_duration for window in windows):
        return tuple(
            window.model_copy(update={"max_budget": min(window.max_budget, budget.max_budget)})
            if window.budget_duration == budget.budget_duration
            else window
            for window in windows
        )
    return (*windows, AppDurationLimit(budget_duration=budget.budget_duration, max_budget=budget.max_budget))


class CustomerScopeAuth(BaseModel):
    """One level of the customer hierarchy a request runs under, with the limits its budget row grants."""

    model_config = ConfigDict(frozen=True)

    key: CustomerScopeKey
    entity_id: str
    alias: str | None = None
    spend: float = 0.0
    budget_reset_at: datetime | None = None
    max_budget: float | None = None
    budget_limits: tuple[BudgetLimit | AppDurationLimit, ...] = ()
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    allowed_models: tuple[str, ...] | None = None
    model_tpm_limit: Mapping[str, int] | None = None
    model_rpm_limit: Mapping[str, int] | None = None
    cap_policy: CapPolicy | None = None


_APP_FIELDS: Final = ("app_id", "app_alias", "app_customer_margin")


class CustomerHierarchyAuth(BaseModel):
    """The levels a request is charged to. A request that named only an organization alias has no team or member."""

    model_config = ConfigDict(frozen=True)

    customer_org_id: str | None = None
    customer_org_alias: str | None = None
    customer_team_id: str | None = None
    customer_team_alias: str | None = None
    customer_member_id: str | None = None
    customer_member_alias: str | None = None
    app_id: str | None = None
    app_alias: str | None = None
    app_customer_margin: float | None = Field(
        default=None, description="The app's own customer surcharge; None uses the gateway-wide one"
    )
    scopes: tuple[CustomerScopeAuth, ...]
    team_scopes: tuple[CustomerScopeAuth, ...] = Field(
        default=(),
        description="Every team the member belongs to, in parent_team_ids order; scopes holds the selected one",
    )

    @model_serializer(mode="wrap")
    def serialize_identity(self, handler: SerializerFunctionWrapHandler) -> Mapping[str, object]:
        values: Final = TypeAdapter(Mapping[str, object]).validate_python(handler(self))
        return {key: value for key, value in values.items() if key not in _APP_FIELDS or value is not None}


@dataclass(frozen=True, slots=True)
class CustomerHierarchyIds:
    """The rows a request's spend is charged to, as the request metadata recorded them. Team and member are absent
    when the request named only an organization."""

    customer_org_id: str | None = None
    customer_team_id: str | None = None
    customer_member_id: str | None = None
    app_id: str | None = None

    def entities(self) -> Mapping[str, str | None]:
        return {
            "customer_member_id": self.customer_member_id,
            "customer_team_id": self.customer_team_id,
            "customer_org_id": self.customer_org_id,
            **({"app_id": self.app_id} if self.app_id else {}),
        }

    def spend_entities(self) -> tuple[tuple[Litellm_EntityType, str], ...]:
        levels: Final = (
            (Litellm_EntityType.CUSTOMER_ORG, self.customer_org_id),
            (Litellm_EntityType.CUSTOMER_TEAM, self.customer_team_id),
            (Litellm_EntityType.CUSTOMER_MEMBER, self.customer_member_id),
            (Litellm_EntityType.APP, self.app_id),
            (
                Litellm_EntityType.APP_CUSTOMER_ORG,
                (f"{self.app_id}:{self.customer_org_id}" if self.app_id and self.customer_org_id else None),
            ),
            (
                Litellm_EntityType.APP_CUSTOMER_MEMBER,
                (f"{self.app_id}:{self.customer_member_id}" if self.app_id and self.customer_member_id else None),
            ),
        )
        return tuple((entity_type, entity_id) for entity_type, entity_id in levels if entity_id)


_FIELDS: Final = TypeAdapter(Mapping[str, object])
_NO_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})
_BUDGET_SCOPES: Final = TypeAdapter(tuple[CustomerScopeAuth, ...])


def customer_budget_scopes_metadata(hierarchy: CustomerHierarchyAuth | None) -> str | None:
    if hierarchy is None:
        return None
    return _BUDGET_SCOPES.dump_json(
        hierarchy.scopes,
        include={"__all__": {"key": True, "entity_id": True, "budget_limits": True}},
    ).decode()


def _mapping(value: object) -> Mapping[str, object]:
    try:
        return _FIELDS.validate_python(value)
    except ValidationError:
        return _NO_FIELDS


def _stamped_metadata(litellm_params: Mapping[str, object]) -> Mapping[str, object]:
    """The proxy stamps auth ids into ``litellm_metadata`` on routes whose body owns ``metadata``, else into ``metadata``."""
    buckets: Final = tuple(_mapping(litellm_params.get(name)) for name in ("litellm_metadata", "metadata"))
    return next(
        (bucket for bucket in buckets if "user_api_key_customer_org_id" in bucket or "user_api_key_app_id" in bucket),
        _NO_FIELDS,
    )


def _stamped_id(metadata: Mapping[str, object], level: str) -> str | None:
    value: Final = metadata.get(f"user_api_key_customer_{level}_id")
    return value if isinstance(value, str) and value else None


def customer_hierarchy_ids_from_request(
    kwargs: Mapping[str, object],
) -> CustomerHierarchyIds | None:
    """The ids auth stamped into the request's proxy metadata, or None when the request named no customer."""
    metadata: Final = _stamped_metadata(_mapping(kwargs.get("litellm_params")))
    org_id: Final = _stamped_id(metadata, "org")
    app: Final = metadata.get("user_api_key_app_id")
    app_id: Final = app if isinstance(app, str) and app else None
    if org_id is None and app_id is None:
        return None
    return CustomerHierarchyIds(
        customer_org_id=org_id,
        customer_team_id=_stamped_id(metadata, "team"),
        customer_member_id=_stamped_id(metadata, "member"),
        app_id=app_id,
    )


def customer_budget_scopes_from_request(
    kwargs: Mapping[str, object],
) -> tuple[CustomerScopeAuth, ...] | None:
    metadata: Final = _stamped_metadata(_mapping(kwargs.get("litellm_params")))
    snapshot: Final = metadata.get("user_api_key_customer_budget_scopes")
    if not isinstance(snapshot, str):
        return None
    scopes: Final = _BUDGET_SCOPES.validate_json(snapshot)
    ids: Final = customer_hierarchy_ids_from_request(kwargs)
    identities: Final = {kind.value: value for kind, value in ids.spend_entities()} if ids is not None else {}
    return tuple(scope for scope in scopes if identities.get(scope.key) == scope.entity_id)
