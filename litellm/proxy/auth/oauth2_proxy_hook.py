from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fastapi import Request
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.core.identity import (
    DisallowedProxyMappings,
    OAuth2ProxyIdentity,
    ProxyMappingsMissing,
    TrustedProxyRangesMissing,
    UntrustedProxy,
)
from litellm.proxy.auth.core.inputs import OAuth2ProxySettings
from litellm.proxy.auth.core.oauth2 import ALLOWED_OAUTH2_PROXY_FIELDS
from litellm.proxy.auth.network import parse_trusted_proxy_ranges
from litellm.proxy.auth.selected_core import AUTH_CORE
from litellm.proxy.auth.trusted_proxy_utils import (
    TRUSTED_PROXY_RANGES_KEY,
    direct_client_ip,
    trusted_proxy_ranges_missing,
    untrusted_proxy,
)

_FEATURE_NAME: Final = "OAuth2 proxy auth"
_MAPPINGS: Final = TypeAdapter(dict[str, str])


class _RequestHeaders:
    def __init__(self, request: Request) -> None:
        self._request: Final = request

    def client_ip(self) -> str | None:
        return direct_client_ip(self._request)

    def header(self, name: str) -> str | None:
        return self._request.headers.get(name)


def _settings(general_settings: Mapping[str, object]) -> OAuth2ProxySettings:
    mappings: Final = _MAPPINGS.validate_python(general_settings.get("oauth2_config_mappings") or {})
    return OAuth2ProxySettings(
        trusted_networks=tuple(
            parse_trusted_proxy_ranges(
                general_settings.get(TRUSTED_PROXY_RANGES_KEY), setting_name=TRUSTED_PROXY_RANGES_KEY
            )
        ),
        header_mappings=MappingProxyType(mappings),
    )


def _user_api_key_auth(identity: OAuth2ProxyIdentity) -> UserAPIKeyAuth:
    fields: Final[dict[str, object]] = {
        "user_id": identity.user_id,
        "user_email": identity.user_email,
        "team_id": identity.team_id,
        "team_alias": identity.team_alias,
        "org_id": identity.org_id,
        "models": None if identity.models is None else list(identity.models),
    }
    asserted: Final = {field: value for field, value in fields.items() if value is not None}
    verbose_proxy_logger.debug("Auth data before creating UserAPIKeyAuth object: keys=%s", list(asserted))
    return UserAPIKeyAuth.model_validate(asserted)


async def handle_oauth2_proxy_request(request: Request) -> UserAPIKeyAuth:
    """
    Resolve a ``UserAPIKeyAuth`` from the identity headers a trusted OAuth2 reverse proxy (oauth2-proxy,
    Authelia) injects, per the admin-set ``oauth2_config_mappings``. Only the identity fields in
    ``ALLOWED_OAUTH2_PROXY_FIELDS`` may be mapped.
    """
    from litellm.proxy.proxy_server import general_settings

    verbose_proxy_logger.debug("Handling oauth2 proxy request")
    result: Final = AUTH_CORE.oauth2().proxy(_RequestHeaders(request), _settings(general_settings))
    match result:
        case OAuth2ProxyIdentity():
            return _user_api_key_auth(result)
        case TrustedProxyRangesMissing():
            raise trusted_proxy_ranges_missing(_FEATURE_NAME)
        case UntrustedProxy(client_ip=client_ip):
            raise untrusted_proxy(_FEATURE_NAME, client_ip)
        case ProxyMappingsMissing():
            raise ValueError("Oauth2 config mappings not found in general_settings")
        case DisallowedProxyMappings(fields=fields):
            raise ValueError(
                "Oauth2 proxy auth refuses to map non-identity UserAPIKeyAuth "
                f"fields from request headers: {list(fields)}. Only identity "
                f"fields are accepted ({sorted(ALLOWED_OAUTH2_PROXY_FIELDS)}); "
                "anything else (privileges, budgets, rate limits, metadata) "
                "would let a caller forge enforcement parameters by spoofing "
                "the matching header. If you need a trusted upstream to "
                "assert anything beyond identity, use JWT auth "
                "(signature-validated) instead of header-trust."
            )
