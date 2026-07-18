"""The IdP OAuth provider config for delegated-OBO consent capture (Path B, item 2.2).

The mint arm (item 3) exchanges a user's stored IdP grant; this is where that grant comes from: a
first-time consent flow runs an ``authorization_code + offline_access`` grant against the user's IdP
(e.g. Okta) so the gateway captures and stores the user's refresh token. The token_exchange server
config carries the IdP's token endpoint and the gateway's client credentials, but not the authorize
endpoint or the offline_access scope the capture needs, so those live here.

One provider serves every ``token_exchange`` upstream fronted by the same IdP, so a provider is keyed
by its token endpoint - the same anchor the mint arm's grant store uses (``idp_grant_key``) - and the
consent flow stores the captured grant under that key, ready for the mint arm to read.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, SecretStr, TypeAdapter

from litellm.proxy._experimental.mcp_server.outbound_credentials.idp_subject_source import (
    idp_grant_key,
)


class IdpOAuthProvider(BaseModel):
    """One IdP's OAuth config for the consent-capture ``authorization_code`` flow.

    ``token_url`` is the IdP's token endpoint; it is also the anchor the captured grant is keyed by
    (via ``idp_grant_key``), so it must match the ``token_exchange_endpoint`` of the servers this IdP
    fronts. ``scopes`` must include an offline-access scope for the IdP to return a refresh_token; the
    default covers the OIDC + Okta case.
    """

    model_config = ConfigDict(frozen=True)

    token_url: str
    authorize_url: str
    client_id: str
    client_secret: SecretStr
    scopes: tuple[str, ...] = ("openid", "offline_access")

    @property
    def grant_key(self) -> str:
        """The (user, idp) storage key's idp component the captured grant is stored under."""
        return idp_grant_key(self.token_url)


class IdpOAuthProviderRegistry:
    """Immutable lookup of the configured IdP providers, keyed by their grant key.

    Built once from config at startup. A provider is resolved by the ``idp`` the consent flow targets
    (the token endpoint), so the captured grant lands under the exact key the mint arm reads.
    """

    def __init__(self, providers: tuple[IdpOAuthProvider, ...]) -> None:
        self._by_key: dict[str, IdpOAuthProvider] = {p.grant_key: p for p in providers}

    def get(self, grant_key: str) -> IdpOAuthProvider | None:
        return self._by_key.get(grant_key)

    def get_by_token_url(self, token_url: str) -> IdpOAuthProvider | None:
        return self._by_key.get(idp_grant_key(token_url))

    def __len__(self) -> int:
        return len(self._by_key)


def load_idp_oauth_providers(raw_providers: object) -> IdpOAuthProviderRegistry:
    """Build the registry from the ``mcp_idp_oauth_providers`` config block (a list of dicts).

    Each entry is validated into an ``IdpOAuthProvider``; a malformed entry raises at load time (fail
    fast at startup) rather than surfacing as a broken consent flow later. A missing/empty block yields
    an empty registry, so the consent endpoints simply 404 until an IdP is configured.
    """
    if not isinstance(raw_providers, list):
        return IdpOAuthProviderRegistry(())
    return IdpOAuthProviderRegistry(_PROVIDERS_ADAPTER.validate_python(raw_providers))


_PROVIDERS_ADAPTER: TypeAdapter[tuple[IdpOAuthProvider, ...]] = TypeAdapter(tuple[IdpOAuthProvider, ...])


_registry = IdpOAuthProviderRegistry(())


def set_idp_oauth_registry(registry: IdpOAuthProviderRegistry) -> None:
    """Install the process-wide registry (called once from config load at startup)."""
    global _registry  # noqa: PLW0603  # single install point for the startup-built registry
    _registry = registry


def get_idp_oauth_registry() -> IdpOAuthProviderRegistry:
    return _registry
