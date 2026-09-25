from collections.abc import Mapping
from datetime import datetime
from typing import Final, NoReturn, TypedDict
from uuid import uuid4

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly

from litellm.proxy.common_utils.timezone_utils import get_budget_reset_time
from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import (
    AgentBudgetConfig,
    AgentExecutionMode,
    AgentIdentityBinding,
    AgentIdentityFailure,
    AgentSubject,
    EntraIdentityConfig,
)

_MODE: Final = TypeAdapter(AgentExecutionMode)


class IdentityFields(TypedDict, total=False):
    provider: ReadOnly[str]
    tenant_id: ReadOnly[str]
    client_id: ReadOnly[str]
    issuer: ReadOnly[str]
    service_principal_id: ReadOnly[str | None]
    required_roles: ReadOnly[tuple[str, ...]]
    required_scopes: ReadOnly[tuple[str, ...]]
    active: ReadOnly[bool]
    revision: ReadOnly[str]
    last_authenticated_at: ReadOnly[datetime | None]


class IdentityUpsert(TypedDict):
    create: ReadOnly[IdentityFields]
    update: ReadOnly[IdentityFields]


class IdentityRelationWrite(TypedDict, total=False):
    create: ReadOnly[IdentityFields]
    update: ReadOnly[IdentityFields]
    upsert: ReadOnly[IdentityUpsert]


class IdentityHistoryKey(TypedDict):
    provider: ReadOnly[str]
    tenant_id: ReadOnly[str]
    client_id: ReadOnly[str]


class IdentityHistoryWhere(TypedDict):
    provider_tenant_id_client_id: ReadOnly[IdentityHistoryKey]


class IdentityHistoryEntry(IdentityHistoryKey):
    issuer: ReadOnly[str]


class IdentityHistoryConnect(TypedDict):
    where: ReadOnly[IdentityHistoryWhere]
    create: ReadOnly[IdentityHistoryEntry]


class IdentityHistoryWrite(TypedDict):
    connectOrCreate: ReadOnly[IdentityHistoryConnect]


class BudgetFields(TypedDict, total=False):
    max_budget: ReadOnly[float]
    budget_duration: ReadOnly[str | None]
    budget_reset_at: ReadOnly[datetime | None]
    updated_by: ReadOnly[str]
    created_by: ReadOnly[str]


class BudgetRelationWrite(TypedDict, total=False):
    create: ReadOnly[BudgetFields]
    update: ReadOnly[BudgetFields]
    disconnect: ReadOnly[bool]


class ManagedWriteFields(TypedDict, total=False):
    enabled: ReadOnly[bool]
    execution_mode: ReadOnly[AgentExecutionMode]
    identity_managed: ReadOnly[bool]
    identity: ReadOnly[IdentityRelationWrite]
    retired_identities: ReadOnly[IdentityHistoryWrite]
    litellm_budget_table: ReadOnly[BudgetRelationWrite]


def raise_identity_failure(failure: AgentIdentityFailure, status_code: int = 403) -> NoReturn:
    raise HTTPException(503 if failure.code == "policy_unavailable" else status_code, failure.message)


def _configuration_failure(
    identity: EntraIdentityConfig | AgentIdentityBinding | None,
    mode: AgentExecutionMode,
    enabling_without_binding: bool,
) -> AgentIdentityFailure | None:
    if identity is not None and mode != "delegated" and not identity.service_principal_id:
        return AgentIdentityFailure(
            message="Autonomous mode requires the Enterprise application service-principal object ID"
        )
    if identity is not None and mode != "autonomous" and not identity.required_scopes:
        return AgentIdentityFailure(message="Delegated mode requires at least one delegated scope")
    if enabling_without_binding and (
        identity is None or isinstance(identity, AgentIdentityBinding) and not identity.active
    ):
        return AgentIdentityFailure(message="Bind an identity before enabling this managed agent")
    return None


def managed_write_fields(
    incoming: Mapping[str, object],
    existing: AgentResponse | None,
    updated_by: str,
) -> ManagedWriteFields | AgentIdentityFailure:
    try:
        identity: Final = (
            EntraIdentityConfig.model_validate(incoming["identity"]) if incoming.get("identity") is not None else None
        )
        mode: Final = _MODE.validate_python(
            incoming.get("execution_mode", existing.execution_mode if existing else "autonomous")
        )
        current_identity: Final = identity if "identity" in incoming else existing.identity if existing else None
        failure: Final = _configuration_failure(
            current_identity,
            mode,
            incoming.get("enabled") is True
            and "identity" not in incoming
            and bool(existing and existing.identity_managed),
        )
        if failure is not None:
            return failure
        empty: Final[ManagedWriteFields] = {}
        identity_fields: Final = _identity_write(identity, existing) if "identity" in incoming else empty
        if isinstance(identity_fields, AgentIdentityFailure):
            return identity_fields
        budget_fields: Final = (
            _budget_write(incoming["budget"], existing, updated_by) if "budget" in incoming else empty
        )
        result: Final[ManagedWriteFields] = {
            **({"enabled": incoming["enabled"] is True} if "enabled" in incoming else {}),
            **({"execution_mode": mode} if "execution_mode" in incoming else {}),
            **budget_fields,
            **identity_fields,
        }
        return result
    except (ValidationError, ValueError) as exc:
        return AgentIdentityFailure(message=f"Invalid agent identity or budget configuration: {exc}")


def _identity_write(
    identity: EntraIdentityConfig | None, existing: AgentResponse | None
) -> ManagedWriteFields | AgentIdentityFailure:
    if identity is None:
        unbind: Final[ManagedWriteFields] = {
            **(
                {"identity": {"update": {"active": False, "revision": str(uuid4()), "last_authenticated_at": None}}}
                if existing and existing.identity
                else {}
            ),
            **({"identity_managed": True, "enabled": False} if existing and existing.identity_managed else {}),
        }
        return unbind
    if (
        existing
        and existing.identity
        and existing.identity.active
        and all(getattr(existing.identity, name) == value for name, value in identity.model_dump().items())
    ):
        unchanged: Final[ManagedWriteFields] = {}
        return unchanged
    binding: Final[IdentityFields] = {
        "provider": identity.provider,
        "tenant_id": identity.tenant_id,
        "client_id": identity.client_id,
        "service_principal_id": identity.service_principal_id,
        "required_roles": identity.required_roles,
        "required_scopes": identity.required_scopes,
        "issuer": identity.issuer,
        "active": True,
        "revision": str(uuid4()),
        "last_authenticated_at": None,
    }
    result: Final[ManagedWriteFields] = {
        "retired_identities": {
            "connectOrCreate": {
                "where": {
                    "provider_tenant_id_client_id": {
                        "provider": identity.provider,
                        "tenant_id": identity.tenant_id,
                        "client_id": identity.client_id,
                    }
                },
                "create": {
                    "provider": identity.provider,
                    "issuer": identity.issuer,
                    "tenant_id": identity.tenant_id,
                    "client_id": identity.client_id,
                },
            }
        },
        "identity_managed": True,
        "identity": {"upsert": {"create": binding, "update": binding}} if existing else {"create": binding},
    }
    return result


def _budget_write(raw: object, existing: AgentResponse | None, updated_by: str) -> ManagedWriteFields:
    if raw is None:
        if existing and existing.budget_id:
            disconnected: Final[ManagedWriteFields] = {"litellm_budget_table": {"disconnect": True}}
            return disconnected
        empty: Final[ManagedWriteFields] = {}
        return empty
    budget: Final = AgentBudgetConfig.model_validate(raw)
    fields: Final[BudgetFields] = {
        "max_budget": budget.max_budget,
        "budget_duration": budget.budget_duration,
        "updated_by": updated_by,
        "budget_reset_at": (
            existing.litellm_budget_table.budget_reset_at
            if existing
            and existing.litellm_budget_table
            and existing.litellm_budget_table.budget_duration == budget.budget_duration
            else get_budget_reset_time(budget.budget_duration)
            if budget.budget_duration
            else None
        ),
    }
    result: Final[ManagedWriteFields] = {
        "litellm_budget_table": (
            {"update": fields} if existing and existing.budget_id else {"create": {**fields, "created_by": updated_by}}
        )
    }
    return result


def classify_agent_subject(
    binding: AgentIdentityBinding,
    claims: Mapping[str, object],
    allowed_mode: AgentExecutionMode,
) -> AgentSubject | AgentIdentityFailure:
    if (claims.get("iss"), claims.get("tid"), claims.get("azp")) != (
        binding.issuer,
        binding.tenant_id,
        binding.client_id,
    ):
        return AgentIdentityFailure(message="Token does not match the registered Entra application")
    oid: Final = claims.get("oid")
    if not isinstance(oid, str) or not oid:
        return AgentIdentityFailure(message="Entra token must identify its object subject")
    scope: Final = claims.get("scp")
    facets: Final = claims.get("xms_sub_fct")
    if facets is not None and (not isinstance(facets, str) or "13" in facets.split()):
        return AgentIdentityFailure(message="Native agent-user authentication is not supported by this binding")
    if scope is not None and not isinstance(scope, str):
        return AgentIdentityFailure(message="Invalid delegated scope claim")
    if isinstance(scope, str) and scope:
        if allowed_mode == "autonomous" or oid == binding.service_principal_id or claims.get("idtyp") == "app":
            return AgentIdentityFailure(message="Delegated token contradicts the configured agent identity or mode")
        if not binding.required_scopes or not frozenset(binding.required_scopes).issubset(scope.split()):
            return AgentIdentityFailure(message="Token lacks the required delegated scopes")
        return AgentSubject(kind="delegated_subject", oid=oid, mode="delegated")
    if allowed_mode == "delegated" or oid != binding.service_principal_id or claims.get("idtyp") == "user":
        return AgentIdentityFailure(message="Application token contradicts the configured agent identity or mode")
    roles: Final = claims.get("roles", ())
    if not isinstance(roles, (list, tuple)) or any(not isinstance(role, str) for role in roles):
        return AgentIdentityFailure(message="Invalid application roles claim")
    if not frozenset(binding.required_roles).issubset(roles):
        return AgentIdentityFailure(message="Token lacks the required application roles")
    return AgentSubject(kind="application", oid=oid, mode="autonomous")
