from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from uuid import UUID

from litellm.types.proxy.agent_identity import MicrosoftInteractiveSubject


def microsoft_interactive_subject(
    tenant: str | None,
    response: Mapping[str, object],
    endpoints: Mapping[str, str | None],
) -> MicrosoftInteractiveSubject | None:
    if tenant is None:
        return None
    try:
        tenant_id: Final = str(UUID(tenant))
        object_id: Final = response.get("id")
        if not isinstance(object_id, str):
            return None
        oid: Final = str(UUID(object_id))
    except ValueError:
        return None
    expected: Final = MappingProxyType(
        {
            "MICROSOFT_AUTHORIZATION_ENDPOINT": f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize",
            "MICROSOFT_TOKEN_ENDPOINT": f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            "MICROSOFT_USERINFO_ENDPOINT": "https://graph.microsoft.com/v1.0/me",
        }
    )
    if any(value and value != expected.get(name) for name, value in endpoints.items()):
        return None
    return MicrosoftInteractiveSubject(
        issuer=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
        tenant_id=tenant_id,
        oid=oid,
    )


async def enroll_microsoft_subject(subject: object, user_id: object, client: object) -> None:
    from litellm.proxy.agent_endpoints.identity_store import AgentIdentityStore
    from litellm.proxy.agent_endpoints.managed_identity import raise_identity_failure
    from litellm.types.proxy.agent_identity import AgentIdentityFailure

    if not isinstance(subject, MicrosoftInteractiveSubject) or not isinstance(user_id, str) or not user_id:
        return
    result: Final = await AgentIdentityStore.from_client(client).enroll_interactive_human(subject, user_id)
    if isinstance(result, AgentIdentityFailure):
        raise_identity_failure(result)
