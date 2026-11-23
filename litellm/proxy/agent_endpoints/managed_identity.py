from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final, NoReturn, TypedDict
from uuid import uuid4

from fastapi import HTTPException
from pydantic import TypeAdapter, ValidationError
from typing_extensions import ReadOnly

from litellm.types.agents import AgentResponse
from litellm.types.proxy.agent_identity import (
    AGENT_IDENTITY_CONFIG,
    AgentExecutionMode,
    AgentIdentityBinding,
    AgentIdentityFailure,
    AgentSubject,
    EntraIdentityConfig,
    OktaIdentityConfig,
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


class IdentityHistoryEntry(IdentityHistoryKey):
    issuer: ReadOnly[str]


class IdentityHistoryWrite(TypedDict):
    create: ReadOnly[IdentityHistoryEntry]


class ManagedWriteFields(TypedDict, total=False):
    enabled: ReadOnly[bool]
    execution_mode: ReadOnly[AgentExecutionMode]
    identity_managed: ReadOnly[bool]
    identity: ReadOnly[IdentityRelationWrite]
    retired_identities: ReadOnly[IdentityHistoryWrite]


def raise_identity_failure(failure: AgentIdentityFailure, status_code: int = 403) -> NoReturn:
    raise HTTPException(503 if failure.code == "policy_unavailable" else status_code, failure.message)


def _service_principal(identity: EntraIdentityConfig | OktaIdentityConfig | AgentIdentityBinding | None) -> str | None:
    return None if identity is None or isinstance(identity, OktaIdentityConfig) else identity.service_principal_id


def _configuration_failure(
    identity: EntraIdentityConfig | OktaIdentityConfig | AgentIdentityBinding | None,
    mode: AgentExecutionMode,
    enabling_without_binding: bool,
) -> AgentIdentityFailure | None:
    provider: Final = identity.provider if identity is not None else None
    if provider == "microsoft_entra" and mode != "delegated" and not _service_principal(identity):
        return AgentIdentityFailure(
            message="Autonomous mode requires the Enterprise application service-principal object ID"
        )
    if enabling_without_binding and (
        identity is None or isinstance(identity, AgentIdentityBinding) and not identity.active
    ):
        return AgentIdentityFailure(message="Bind an identity before enabling this managed agent")
    return None


def managed_write_fields(
    incoming: Mapping[str, object],
    existing: AgentResponse | None,
    updated_by: str,
    okta_orgs: Mapping[str, str] = MappingProxyType({}),
) -> ManagedWriteFields | AgentIdentityFailure:
    try:
        identity: Final = (
            AGENT_IDENTITY_CONFIG.validate_python(incoming["identity"])
            if incoming.get("identity") is not None
            else None
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
        identity_fields: Final = _identity_write(identity, existing, okta_orgs) if "identity" in incoming else empty
        result: Final[ManagedWriteFields] = {
            **({"enabled": incoming["enabled"] is True} if "enabled" in incoming else {}),
            **({"execution_mode": mode} if "execution_mode" in incoming else {}),
            **identity_fields,
        }
        return result
    except (ValidationError, ValueError) as exc:
        return AgentIdentityFailure(message=f"Invalid agent identity configuration: {exc}")


def _identity_fields(identity: EntraIdentityConfig | OktaIdentityConfig, tenant: str) -> Mapping[str, object]:
    return {**identity.model_dump(), "tenant_id": tenant}


def _tenant(identity: EntraIdentityConfig | OktaIdentityConfig, okta_orgs: Mapping[str, str]) -> str:
    if isinstance(identity, EntraIdentityConfig):
        return identity.tenant_id
    org: Final = okta_orgs.get(identity.issuer)
    if org is None:
        raise ValueError(f"issuer {identity.issuer} is not opted in with agent_identity_provider: okta")
    return org


def _identity_write(
    identity: EntraIdentityConfig | OktaIdentityConfig | None,
    existing: AgentResponse | None,
    okta_orgs: Mapping[str, str],
) -> ManagedWriteFields:
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
        and all(
            getattr(existing.identity, name) == value
            for name, value in _identity_fields(identity, _tenant(identity, okta_orgs)).items()
        )
    ):
        unchanged: Final[ManagedWriteFields] = {}
        return unchanged
    tenant: Final = _tenant(identity, okta_orgs)
    entra: Final = identity if isinstance(identity, EntraIdentityConfig) else None
    binding: Final[IdentityFields] = {
        "provider": identity.provider,
        "tenant_id": tenant,
        "client_id": identity.client_id,
        "service_principal_id": entra.service_principal_id if entra else None,
        "required_roles": entra.required_roles if entra else (),
        "required_scopes": identity.required_scopes,
        "issuer": identity.issuer,
        "active": True,
        "revision": str(uuid4()),
        "last_authenticated_at": None,
    }
    result: Final[ManagedWriteFields] = {
        "retired_identities": {
            "create": {
                "provider": identity.provider,
                "issuer": identity.issuer,
                "tenant_id": tenant,
                "client_id": identity.client_id,
            }
        },
        "identity_managed": True,
        "identity": {"upsert": {"create": binding, "update": binding}} if existing else {"create": binding},
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
        granted_scopes: Final = frozenset(scope.split())
        if not granted_scopes or not frozenset(binding.required_scopes).issubset(granted_scopes):
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
