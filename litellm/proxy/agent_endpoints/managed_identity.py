from collections.abc import Mapping
from typing import Final

from litellm.types.proxy.agent_identity import (
    AgentExecutionMode,
    AgentIdentityBinding,
    AgentIdentityFailure,
    AgentSubject,
)


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
