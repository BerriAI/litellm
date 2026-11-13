"""What every gateway-auth-core call is made of on the Python side: the request and host facts it reads, the
proxy's settings, and the loop that runs a call to its outcome by answering each step it parks on from a
`PythonAuthHost`."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import litellm
from litellm.proxy.auth.host.answer import answer
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.rust_bridge._native import (
    RustAuth,
    RustAuthOutcome,
    RustAuthSettings,
    RustAuthStep,
)

if TYPE_CHECKING:
    from litellm.proxy.auth.host.python import PythonAuthHost

TEAM_HEADER: Final = "x-litellm-team-id"


@dataclass(frozen=True, slots=True)
class HostFacts:
    """What the proxy knows that `general_settings` does not say, field for field with `HostFacts` in
    `gateway-auth-core/src/settings.rs`. The defaults are a proxy that configured none of them."""

    master_key: str | None
    premium: bool
    has_db: bool
    ui_login: bool = True
    proxy_budget: bool = False
    metrics_require_auth: bool = True


@dataclass(frozen=True, slots=True)
class AuthRequest:
    route: str
    method: str | None
    token: str | None
    model: str | None
    team_header: str | None
    is_llm_api_route: bool
    auth_enforced_pass_through: bool
    forwarded_routes: tuple[str, ...]
    is_info_route: bool
    now: float
    pass_through_open: bool = False
    from_custom_auth: bool = False
    end_user_id: str | None = None


def team_header(headers: Mapping[str, str] | None) -> str | None:
    """`x-litellm-team-id`, matched case-insensitively."""
    if not headers:
        return None
    return {name.lower(): value for name, value in headers.items()}.get(TEAM_HEADER)


def request(
    token: str | None,
    route: str,
    method: str | None,
    *,
    headers: Mapping[str, str] | None = None,
    model: str | None = None,
    end_user_id: str | None = None,
    pass_through_open: bool = False,
    from_custom_auth: bool = False,
) -> AuthRequest:
    return AuthRequest(
        route=route,
        method=method,
        token=token,
        model=model,
        end_user_id=end_user_id,
        team_header=team_header(headers),
        is_llm_api_route=RouteChecks.is_llm_api_route(route=route),
        auth_enforced_pass_through=RouteChecks.is_auth_enforced_pass_through_route(
            route=route, method=method.upper() if isinstance(method, str) else None
        ),
        forwarded_routes=RouteChecks.forwarded_pass_through_routes(route),
        is_info_route=RouteChecks.is_info_route(route=route),
        now=time.time(),
        pass_through_open=pass_through_open,
        from_custom_auth=from_custom_auth,
    )


def proxy_settings() -> RustAuthSettings:
    """The auth settings as the proxy's globals stand now. A config load or a settings endpoint can change them
    while the proxy serves, so a request builds its own once and passes it to every call it makes."""
    from litellm.proxy import proxy_server  # noqa: PLC0415  # proxy_server imports this module

    return RustAuthSettings(
        proxy_server.current_general_settings(),
        proxy_server.litellm_jwtauth,
        HostFacts(
            master_key=proxy_server.master_key,
            premium=proxy_server.premium_user,
            has_db=proxy_server.prisma_client is not None,
            ui_login=proxy_server.ui_login_enabled,
            proxy_budget=litellm.max_budget > 0,
            metrics_require_auth=litellm.require_auth_for_metrics_endpoint is not False,
        ),
    )


async def drive(call: RustAuth, host: PythonAuthHost) -> RustAuthOutcome:
    """A host step that raises is reported to the core, which skips a step whose failure the proxy
    tolerated. When the core gives up instead, the step's exception propagates as the Python auth path raised
    it."""
    failure: Exception | None = None  # rebind-ok: the latest host failure, re-raised if the core gives up on it
    step = call.resume()  # rebind-ok: each resume returns the next step
    while not isinstance(step, RustAuthStep.Done):
        try:
            await answer(call, step, host)
        except Exception as error:
            failure = error
            call.fail()
        try:
            step = call.resume()
        except RuntimeError:
            if failure is None:
                raise
            raise failure
    return step.outcome
