"""Compose extractors into a single ``IdentityContext`` per request.

This entrypoint is intentionally not yet wired into the proxy auth chain;
it exists so Phase 2 can switch over and so unit tests can exercise the
full path end-to-end.
"""

from typing import Any, Dict, Optional

from litellm.constants import (
    LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
    LITTELM_CLI_SERVICE_ACCOUNT_NAME,
    LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
)
from litellm.identity.context import AuditInfo, ClientInfo, IdentityContext
from litellm.identity.extractors.api_key import extract_api_key_principal
from litellm.identity.extractors.client import extract_client_info
from litellm.identity.extractors.end_user import extract_end_user_id
from litellm.identity.extractors.header import extract_audit_changed_by
from litellm.identity.extractors.jwt import extract_jwt_principal
from litellm.identity.principal import (
    AnonymousPrincipal,
    Principal,
    ServiceAccountPrincipal,
)

_SERVICE_ACCOUNT_API_KEYS = frozenset(
    {
        LITTELM_INTERNAL_HEALTH_SERVICE_ACCOUNT_NAME,
        LITTELM_CLI_SERVICE_ACCOUNT_NAME,
        LITELLM_INTERNAL_JOBS_SERVICE_ACCOUNT_NAME,
    }
)


def _resolve_principal(api_key: Optional[str]) -> Principal:
    if api_key and api_key in _SERVICE_ACCOUNT_API_KEYS:
        return ServiceAccountPrincipal(name=api_key)

    jwt_principal = extract_jwt_principal(api_key)
    if jwt_principal is not None:
        return jwt_principal

    api_key_principal = extract_api_key_principal(api_key)
    if api_key_principal is not None:
        return api_key_principal

    return AnonymousPrincipal()


def resolve_identity(
    *,
    api_key: Optional[str] = None,
    request: Any = None,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    general_settings: Optional[Dict[str, Any]] = None,
) -> IdentityContext:
    principal = _resolve_principal(api_key)
    end_user_id = extract_end_user_id(body=body, headers=headers)
    audit = AuditInfo(changed_by=extract_audit_changed_by(headers))
    client: ClientInfo
    if request is not None:
        client = extract_client_info(
            request=request, general_settings=general_settings
        )
    else:
        client = ClientInfo()

    return IdentityContext(
        principal=principal,
        end_user_id=end_user_id,
        audit=audit,
        client=client,
    )
