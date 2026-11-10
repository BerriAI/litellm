from typing import Final

from litellm.proxy.auth.core.host import OAuth2IntrospectionHost, OAuth2ProxyHost
from litellm.proxy.auth.core.identity import (
    DisallowedProxyMappings,
    InactiveToken,
    IntrospectionNotConfigured,
    OAuth2Denial,
    OAuth2Identity,
    OAuth2ProxyDenial,
    OAuth2ProxyIdentity,
    ProxyMappingsMissing,
    TrustedProxyRangesMissing,
    UntrustedProxy,
)
from litellm.proxy.auth.core.inputs import OAuth2IntrospectionSettings, OAuth2ProxySettings
from litellm.proxy.auth.core.network import ip_in_networks

# Header trust asserts identity only. Mapping any other UserAPIKeyAuth field (role, budgets, limits, permissions)
# would let a caller forge its own enforcement by sending the matching header, and an allowlist stays safe as
# fields are added. A trusted upstream that must assert more should use JWT auth, which checks a signature.
ALLOWED_OAUTH2_PROXY_FIELDS: Final = frozenset({"user_id", "user_email", "team_id", "team_alias", "org_id", "models"})


async def introspect(
    host: OAuth2IntrospectionHost, token: str, settings: OAuth2IntrospectionSettings
) -> OAuth2Identity | OAuth2Denial:
    if not settings.endpoint_configured:
        return IntrospectionNotConfigured()
    answer: Final = await host.introspect(token)
    if settings.requires_active and not answer.active:
        return InactiveToken()
    return OAuth2Identity(token=token, user_id=answer.user_id, role=answer.role, team_id=answer.team_id)


def _proxy_identity(host: OAuth2ProxyHost, settings: OAuth2ProxySettings) -> OAuth2ProxyIdentity:
    def value(field: str) -> str | None:
        header: Final = settings.header_mappings.get(field)
        return (host.header(header) or None) if header is not None else None

    models: Final = value("models")
    return OAuth2ProxyIdentity(
        user_id=value("user_id"),
        user_email=value("user_email"),
        team_id=value("team_id"),
        team_alias=value("team_alias"),
        org_id=value("org_id"),
        models=None if models is None else tuple(model.strip() for model in models.split(",")),
    )


def proxy(host: OAuth2ProxyHost, settings: OAuth2ProxySettings) -> OAuth2ProxyIdentity | OAuth2ProxyDenial:
    if not settings.trusted_networks:
        return TrustedProxyRangesMissing()
    client_ip: Final = host.client_ip()
    if not ip_in_networks(client_ip, settings.trusted_networks):
        return UntrustedProxy(client_ip=client_ip)
    if not settings.header_mappings:
        return ProxyMappingsMissing()
    disallowed: Final = tuple(sorted(settings.header_mappings.keys() - ALLOWED_OAUTH2_PROXY_FIELDS))
    if disallowed:
        return DisallowedProxyMappings(fields=disallowed)
    return _proxy_identity(host, settings)
