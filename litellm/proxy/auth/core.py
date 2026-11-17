"""What a gateway-auth-core call is made of on the Python side: the request and host facts it reads, and the
proxy's settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import litellm
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.rust_bridge._native import RustAuthSettings


@dataclass(frozen=True, slots=True)
class HostFacts:
    """What the proxy knows that `general_settings` does not say. The defaults are a proxy that configured
    none of them."""

    premium: bool
    metrics_require_auth: bool = True


@dataclass(frozen=True, slots=True)
class AuthRequest:
    route: str
    token: str | None
    is_llm_api_route: bool
    is_info_route: bool
    pass_through_open: bool = False
    from_custom_auth: bool = False


def request(
    token: str | None,
    route: str,
    *,
    pass_through_open: bool = False,
    from_custom_auth: bool = False,
) -> AuthRequest:
    return AuthRequest(
        route=route,
        token=token,
        is_llm_api_route=RouteChecks.is_llm_api_route(route=route),
        is_info_route=RouteChecks.is_info_route(route=route),
        pass_through_open=pass_through_open,
        from_custom_auth=from_custom_auth,
    )


def proxy_settings() -> RustAuthSettings:
    """The auth settings as the proxy's globals stand now. A config load or a settings endpoint can change them
    while the proxy serves, so a request builds its own."""
    from litellm.proxy import proxy_server  # noqa: PLC0415  # proxy_server imports this module

    general_settings: Final = proxy_server.current_general_settings()
    return RustAuthSettings(
        general_settings,
        proxy_server.jwt_handler.litellm_jwtauth if general_settings.get("enable_jwt_auth") is True else None,
        HostFacts(
            premium=proxy_server.premium_user is True,
            metrics_require_auth=litellm.require_auth_for_metrics_endpoint is not False,
        ),
    )
